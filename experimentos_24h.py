"""Cinco frentes: harmonização, tuning no treino, resumos, ablação e benchmark.

O teste interno já foi consultado no desenvolvimento anterior: resultados novos
são exploratórios. Nenhuma escolha usa o eICU. Protocolos e grades ficam salvos.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from avaliar_protocolo_24h import (modelos, avaliar_classificador, avaliar_regressor,
                                  explicar, comparar_rede_xgboost)
from variaveis_protocolo import colunas

# v4 = protocolo adotado (painel ampliado + contexto de 24h). v3 fica preservado para
# comparação e para os experimentos exploratórios que foram feitos sobre ele.
ROOT=Path('artefatos_protocolo_24h/experimentos_v4')
V3=Path('artefatos_protocolo_24h/experimentos_v3')
NOVAS=['pas_24h','aki_creatinina_estagio_24h','aki_creatinina_24h','sofa_24h',
       'sofa_componentes_observados','arritmia_yang_24h','arritmia_ventricular_24h',
       'roncos_24h','sibilos_24h','estertores_24h','estridor_24h','murmurio_diminuido_24h',
       'ausculta_alterada_24h','ausculta_registrada_24h','valvulopatia_previa','cateterismo_cardiaco_24h']
TODAS_V3=colunas('completo')+NOVAS
# Contexto conhecido até 24h (contexto_24h.py): tipo de UTI, origem, tempo pré-UTI e tendências.
CONTEXTO=[f'unidade_{u}' for u in ['coronariana','cardiotoracica','clinica','cirurgica','mista','neuro']]+    [f'origem_{o}' for o in ['pronto_socorro','centro_cirurgico','enfermaria','outra_uti','direta_externa']]+    ['horas_pre_uti']+[f'{n}_{s}_24h' for n in ['creatinina','bun','lactato'] for s in ['ultimo','delta']]
TODAS=TODAS_V3+CONTEXTO
IO=[c for c in TODAS if any(s in c for s in ['noradrenalina','vasopress','adrenalina',
    'fenilefrina','dopamina','inotropico','diuretic','furosemida','volume_infundido','diurese',
    'balanco','oliguria','bolus_cristaloide'])]+['sofa_24h','sofa_componentes_observados']
YANG=['arritmia_yang_retro','aki_creatinina_estagio_24h','sofa_24h','ausculta_alterada_24h',
      'valvulopatia_retro','pas_24h','diuretico_alca_48h','bun_max_24h','spo2_24h','cateterismo_cardiaco_48h']
CATS=['genero','tipo_admissao']
SEARCH_VERSION=1

class CoberturaNumerica(TransformerMixin,BaseEstimator):
    def __init__(self,max_missing=1.): self.max_missing=max_missing
    def fit(self,X,y=None):
        self.feature_names_in_=np.asarray(X.columns,dtype=object)
        self.keep_=X.columns[X.isna().mean().le(self.max_missing)]
        if not len(self.keep_): raise ValueError('Nenhuma coluna numérica após filtro de cobertura')
        return self
    def transform(self,X): return X[self.keep_]
    def get_feature_names_out(self,input_features=None): return np.asarray(self.keep_,dtype=object)

def preparo(features,indicadores=False,max_missing=1.):
    num=[c for c in features if c not in CATS]; cat=[c for c in features if c in CATS]
    parts=[('num',Pipeline([('cobertura',CoberturaNumerica(max_missing)),
        ('impute',SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=indicadores)),
        ('scale',StandardScaler())]),num)]
    if cat: parts.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat))
    return ColumnTransformer(parts)

def dividir_preservando(d):
    original=pd.read_csv('artefatos_protocolo_24h/completo/divisao.csv')
    grupos=original.drop_duplicates('patient_id').set_index('patient_id').particao
    p=d.patient_id.map(grupos)
    # Pacientes novos do benchmark de estadias curtas recebem partição determinística.
    new=d.patient_id.map(lambda s:'teste' if int(hashlib.sha256(str(s).encode()).hexdigest()[:8],16)%5==0 else 'treino')
    p=p.fillna(new)
    tr,te=d[p.eq('treino')].copy(),d[p.eq('teste')].copy()
    assert not set(tr.patient_id)&set(te.patient_id)
    return tr,te

def hash_dados(d,features):
    return hashlib.sha256(pd.util.hash_pandas_object(d[['patient_id','stay_id','los_dias']+features],index=False).values.tobytes()).hexdigest()

def pipeline(features,nome,regressao=False,indicadores=False,max_missing=1.):
    cls,regs=modelos()
    model=clone((regs if regressao else cls)[nome])
    if nome=='rede_neural': model.set_params(max_iter=200,n_iter_no_change=15)
    if regressao: model=TransformedTargetRegressor(regressor=model,func=np.log1p,inverse_func=np.expm1)
    return Pipeline([('preparo',preparo(features,indicadores,max_missing)),('modelo',model)])

def grade(nome,regressao=False):
    pre='modelo__regressor__' if regressao else 'modelo__'
    if nome=='xgboost':
        raw={'n_estimators':[200,350],'max_depth':[2,3,5],'learning_rate':[.03,.07],
             'min_child_weight':[1,5,10],'reg_lambda':[1,5,15],'reg_alpha':[0,.5],
             'subsample':[.8,1.],'colsample_bytree':[.8,1.]}
    else:
        raw={'hidden_layer_sizes':[(32,),(64,32)],'alpha':[.1,1.,10.],
             'learning_rate_init':[.0005,.001],'batch_size':[128,256]}
    return {pre+k:v for k,v in raw.items()}

def otimizar(tr,features,nome,corte,folder,regressao=False):
    folder.mkdir(parents=True,exist_ok=True)
    signature=hash_dados(tr,features)
    cfg={'version':SEARCH_VERSION,'signature':signature,'features':features,'nome':nome,
         'corte':corte,'regressao':regressao,'grade':grade(nome,regressao),
         'n_iter':4 if nome=='rede_neural' else (4 if regressao else 8),'cv':3,'seed':42}
    cfg=json.loads(json.dumps(cfg))
    if (folder/'ajuste.joblib').exists() and json.loads((folder/'config.json').read_text())==cfg:
        print('Reutilizando busca:',folder,flush=True);return joblib.load(folder/'ajuste.joblib')
    (folder/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    y=tr.los_dias if regressao else tr.los_dias.gt(corte).astype(int)
    cv=list((GroupKFold(3) if regressao else StratifiedGroupKFold(3,shuffle=True,random_state=42)).split(tr,y,tr.patient_id))
    for a,b in cv: assert not set(tr.iloc[a].patient_id)&set(tr.iloc[b].patient_id)
    search=RandomizedSearchCV(pipeline(features,nome,regressao),grade(nome,regressao),
        n_iter=cfg['n_iter'],scoring='neg_mean_absolute_error' if regressao else 'roc_auc',
        cv=cv,n_jobs=1,random_state=42,refit=True,error_score='raise',return_train_score=False,verbose=1)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning)
        search.fit(tr[features],y)
        # OOF pós-seleção serve ao limiar; não é estimativa não enviesada do processo de busca.
        oof=None if regressao else cross_val_predict(clone(search.best_estimator_),tr[features],y,
            cv=cv,method='predict_proba',n_jobs=1)[:,1]
    pd.DataFrame(search.cv_results_).to_csv(folder/'busca_cv.csv',index=False)
    pd.DataFrame({'aviso':[str(w.message) for w in ws]}).to_csv(folder/'avisos.csv',index=False)
    threshold=None
    if oof is not None:
        fpr,tpr,t=roc_curve(y,oof); ok=np.isfinite(t);threshold=float(t[ok][np.argmax((tpr-fpr)[ok])])
    result={'modelo':search.best_estimator_,'params':search.best_params_,'score_cv':search.best_score_,
            'limiar':threshold,'corte':corte,'features':features,'pacientes_treino':set(tr.patient_id)}
    joblib.dump(result,folder/'ajuste.joblib')
    return result

def harmonizar(m,e,todas=None,root=None):
    todas=TODAS if todas is None else todas; root=ROOT if root is None else root
    # Exclusões motivadas pela fonte; jamais escolhidas pela AUC externa.
    excluded={'feve':'Sem extração temporal equivalente de ecocardiograma no eICU',
              'tipo_admissao':'Categorias de origem/admissão não equivalem às oito categorias MIMIC',
              'furosemida_dose_total_24h':'Snapshots e ordens não comprovam dose acumulada administrada'}
    for col in todas:
        if col not in m or col not in e or m[col].notna().sum()==0 or e[col].notna().sum()==0:
            excluded[col]='Nenhum valor recuperado em pelo menos uma base; não imputar coluna inteira'
    if 'nt_probnp_max_24h' in excluded: excluded['nt_probnp_medido']='Sem NT-proBNP equivalente; BNP não é substituto'
    if 'troponina_t_max_24h' in excluded: excluded['troponina_medida']='Sem troponina T equivalente; não substituir por I'
    features=[c for c in todas if c not in excluded]
    rows=[]
    for col in todas:
        rows.append({'variavel':col,'incluida':col in features,'motivo':excluded.get(col,'Definição harmonizada, com limitações documentadas'),
                     'ausente_mimic':float(m[col].isna().mean()) if col in m else 1.,
                     'ausente_eicu':float(e[col].isna().mean()) if col in e else 1.})
    root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(root/'harmonizacao.csv',index=False)
    return features

def avaliar(result,d,folder,bootstrap=500,shap=False,regressao=False):
    if set(d.patient_id)&result['pacientes_treino']: raise ValueError('Paciente de treino na avaliação')
    x=d[result['features']]
    if regressao: return avaliar_regressor(d,result['modelo'].predict(x),folder)
    r=avaliar_classificador(d,result['modelo'].predict_proba(x)[:,1],result['corte'],result['limiar'],folder,bootstrap)
    if shap: explicar(result['modelo'],x,folder)
    return r

def principais(m,e,features,perfil,bootstrap=500):
    tr,te=dividir_preservando(m)
    folder=ROOT/perfil;folder.mkdir(parents=True,exist_ok=True)
    pd.concat([tr[['patient_id','stay_id']].assign(particao='treino'),te[['patient_id','stay_id']].assign(particao='teste')]).to_csv(folder/'divisao.csv',index=False)
    rows=[]
    for corte in [3.55,7.]:
        for nome in ['xgboost','rede_neural']:
            print(f'BUSCA {perfil} / {nome} / >{corte}d',flush=True)
            fit=otimizar(tr,features,nome,corte,folder/'buscas'/f'{nome}_{corte:g}d')
            r=avaliar(fit,te,folder/'interno'/f'{nome}_{corte:g}d',bootstrap,shap=nome=='xgboost')
            rows.append(dict(perfil=perfil,modelo=nome,corte=corte,base='mimic',auc_cv_selecao=fit['score_cv'],**r))
            if e is not None:
                r=avaliar(fit,e,folder/'externo'/f'{nome}_{corte:g}d',bootstrap)
                rows.append(dict(perfil=perfil,modelo=nome,corte=corte,base='eicu',auc_cv_selecao=fit['score_cv'],**r))
            pd.DataFrame(rows).to_csv(folder/'classificacao.csv',index=False)
    comparar_rede_xgboost(folder/'interno',bootstrap)
    if e is not None: comparar_rede_xgboost(folder/'externo',bootstrap)
    regs=[]
    for nome in ['xgboost','rede_neural']:
        fit=otimizar(tr,features,nome,None,folder/'buscas'/f'regressao_{nome}',regressao=True)
        for base,data in [('mimic',te)]+([] if e is None else [('eicu',e)]):
            r=avaliar(fit,data,folder/base/f'regressao_{nome}',regressao=True)
            regs.append(dict(modelo=nome,base=base,**r))
    pd.DataFrame(regs).to_csv(folder/'regressao.csv',index=False)

def ablacoes(m,bootstrap=500):
    # XGBoost mostrou resultados distintos entre limites OpenMP 2 e 3 nesta
    # instalação. Preserve a mesma configuração numérica do treino principal.
    with threadpool_limits(limits=3):
        return _ablacoes(m,bootstrap)

def _ablacoes(m,bootstrap=500):
    tr,te=dividir_preservando(m)
    rows=[]
    configs={'completo_ampliado':(TODAS,False,1.,False),
        'sem_novas_variaveis':(colunas('completo'),False,1.,False),
        'sem_contexto':(TODAS_V3,False,1.,False),
        'sem_entradas_saidas':([c for c in TODAS if c not in IO],False,1.,False),
        'sem_mais_30pct_ausencia':(TODAS,False,.3,False),
        'indicadores_de_ausencia':(TODAS,True,1.,False),
        'laboratorios_maximos':(TODAS,False,1.,True)}
    for corte in [3.55,7.]:
        best=joblib.load(ROOT/'completo_ampliado/buscas'/f'xgboost_{corte:g}d/ajuste.joblib')
        y=tr.los_dias.gt(corte).astype(int)
        cv=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(tr,y,tr.patient_id))
        for label,(features,ind,missing,maximos) in configs.items():
            dest=ROOT/'ablacoes'/f'{label}_{corte:g}d';dest.mkdir(parents=True,exist_ok=True)
            signature=hash_dados(m,TODAS+[s+'_max_24h' for s in ['creatinina','ureia','sodio','potassio','glicose','ptt']])
            if (dest/'resultado.json').exists() and json.loads((dest/'resultado.json').read_text())['signature']==signature:
                rows.append(json.loads((dest/'resultado.json').read_text())['resultado']);continue
            print('ABLAÇÃO',label,corte,flush=True)
            a,b=tr.copy(),te.copy()
            if maximos:
                for stem in ['creatinina','ureia','sodio','potassio','glicose','ptt']:
                    a[stem+'_24h']=a[stem+'_max_24h'];b[stem+'_24h']=b[stem+'_max_24h']
            model=pipeline(features,'xgboost',indicadores=ind,max_missing=missing).set_params(**best['params'])
            pred=cross_val_predict(model,a[features],y,cv=cv,method='predict_proba')[:,1]
            fpr,tpr,t=roc_curve(y,pred);ok=np.isfinite(t);threshold=float(t[ok][np.argmax((tpr-fpr)[ok])])
            model.fit(a[features],y)
            fit={'modelo':model,'features':features,'pacientes_treino':set(a.patient_id),'corte':corte,'limiar':threshold}
            r=avaliar(fit,b,dest,bootstrap)
            r.update(variante=label,corte=corte,auc_cv=roc_auc_score(y,pred),n_preditores_transformados=len(model.named_steps['preparo'].get_feature_names_out()))
            rows.append(r)
            joblib.dump(fit,dest/'modelo.joblib')
            (dest/'resultado.json').write_text(json.dumps({'signature':signature,'resultado':r},indent=2))
    pd.DataFrame(rows).to_csv(ROOT/'ablacoes/comparacao.csv',index=False)
    # Decisão baseada apenas no treino. Teste é exibido para todas as variantes, exploratoriamente.
    pd.DataFrame(rows).sort_values('auc_cv',ascending=False).drop_duplicates('corte').to_csv(ROOT/'ablacoes/escolha_por_cv.csv',index=False)
    deltas=[]
    for corte in [3.55,7.]:
        ref=pd.read_csv(ROOT/'ablacoes'/f'completo_ampliado_{corte:g}d/previsoes.csv')
        for label in configs:
            if label=='completo_ampliado': continue
            other=pd.read_csv(ROOT/'ablacoes'/f'{label}_{corte:g}d/previsoes.csv')
            deltas.append(dict(variante=label,corte=corte,**diferenca_pareada(ref,other,bootstrap)))
    pd.DataFrame(deltas).to_csv(ROOT/'ablacoes/diferencas_pareadas.csv',index=False)

def diferenca_pareada(a,b,bootstrap=500):
    """AUC de a menos b, reamostrando os mesmos pacientes em ambos."""
    a,b=[d.sort_values('stay_id').reset_index(drop=True) for d in [a,b]]
    if not a[['stay_id','patient_id','alvo']].equals(b[['stay_id','patient_id','alvo']]):
        raise ValueError('Comparação pareada exige pacientes, estadias e alvos iguais')
    y=a.alvo.to_numpy();pa=a.probabilidade.to_numpy();pb=b.probabilidade.to_numpy()
    groups=list(a.groupby('patient_id',sort=True).indices.values())
    rng=np.random.default_rng(42);values=[]
    for _ in range(bootstrap):
        idx=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        if len(np.unique(y[idx]))==2: values.append(roc_auc_score(y[idx],pa[idx])-roc_auc_score(y[idx],pb[idx]))
    lo,hi=np.quantile(values,[.025,.975]) if values else (np.nan,np.nan)
    return {'delta_completo_menos_variante':roc_auc_score(y,pa)-roc_auc_score(y,pb),'ic95_inf':lo,'ic95_sup':hi}

AKI={'creatinina':'aki_creatinina_estagio_24h','kdigo24':'aki_kdigo_estagio_24h','kdigo48':'aki_kdigo_estagio_48h'}

def benchmark(bootstrap=500,ic_max_seq=1,sem_curtas=False,aki='creatinina',revisar=()):
    m=pd.read_csv('data/processados/mimic_ampliado_todas_estadias.csv')
    e=pd.read_csv('data/processados/eicu_ampliado_todas_estadias.csv')
    # Recalcula a evidencia de IO em 48h mesmo ao reutilizar paineis antigos.
    from painel_ampliado import fluidos_eicu
    indexed=e.set_index('stay_id')
    io48=fluidos_eicu(indexed.index,indexed.peso_kg,{},horas=48)
    positivos=io48[['furosemida_io','bumetanida_io','torsemida_io']].max(axis=1)
    e['diuretico_alca_48h']=np.maximum(e.diuretico_alca_48h,e.stay_id.map(positivos))
    if (e.diuretico_alca_24h>e.diuretico_alca_48h).any():
        raise ValueError('Diuretico 24h ausente em 48h')
    if aki!='creatinina':
        # Colunas de aki_kdigo.py; kdigo48 consulta dados posteriores a 24h (retrospectivo).
        m=m.merge(pd.read_csv('data/processados/mimic_aki_kdigo.csv'),on='stay_id',how='left',validate='one_to_one')
        e=e.merge(pd.read_csv('data/processados/eicu_aki_kdigo.csv'),on='stay_id',how='left',validate='one_to_one')
    feats=[AKI[aki] if f=='aki_creatinina_estagio_24h' else f for f in YANG]
    if revisar:
        # Colunas de preditores_yang.py: definições da Tabela 1 de Yang, em 24h.
        m=m.merge(pd.read_csv('data/processados/mimic_preditores_yang.csv'),on='stay_id',how='left',validate='one_to_one')
        e=e.merge(pd.read_csv('data/processados/eicu_preditores_yang.csv'),on='stay_id',how='left',validate='one_to_one')
        troca={'arritmia':{'arritmia_yang_retro':['arritmia_maligna_24h']},
               'ausculta':{'ausculta_alterada_24h':['roncos_yang_24h','bronquico_yang_24h','estridor_yang_24h']},
               'valvulopatia':{'valvulopatia_retro':['valvulopatia_yang']}}
        mapa={k:v for r in revisar for k,v in troca[r].items()}
        feats=[g for f in feats for g in mapa.get(f,[f])]
    if ic_max_seq>1:
        # Import tardio: experimento_populacao importa este módulo.
        from experimento_populacao import ic_ate_posicao
        m=ic_ate_posicao(m,ic_max_seq)
    def select(d):
        keep=d.ic_principal.eq(True)&d.primeira_uti_internacao.eq(True)&d.obito_hospitalar.eq(0)
        if sem_curtas: keep&=d.los_dias.gt(1)
        return d[keep].copy()
    m,e=select(m),select(e)
    tr,te=dividir_preservando(m)
    nome='yang_aproximado'+(f'_ic_seq1a{ic_max_seq}' if ic_max_seq>1 else '')+('_sem_curtas' if sem_curtas else '')+('' if aki=='creatinina' else '_aki_'+aki)+''.join('_'+r for r in sorted(revisar))
    # Variantes exploratórias do benchmark ficam junto das demais análises do v3.
    folder=(ROOT if nome=='yang_aproximado' else V3)/(nome+'_tratamentos_corrigidos');folder.mkdir(parents=True,exist_ok=True)
    pop=('IC principal' if ic_max_seq==1 else f'IC (428/I50) nas posições 1-{ic_max_seq} no MIMIC; Primary no eICU')
    pop+=', primeira UTI da internação, sobreviventes hospitalares, '+('exclui estadias <=24h' if sem_curtas else 'inclui estadias <=24h')
    (folder/'limites.json').write_text(json.dumps({'populacao':pop,'n_treino_mimic':len(tr),'n_teste_mimic':len(te),
        'preditores':feats,'n_mimic':len(m),'n_eicu':len(e),
        'limites':['Aproximação de 10 preditores importantes; não reprodução exata dos 41 candidatos ou nove algoritmos',
        'MIMIC 3.1, não 2.2',('AKI somente critério de creatinina' if aki=='creatinina' else f'AKI KDIGO creatinina+diurese ({AKI[aki]})')+'; SOFA adaptado, com cobertura explícita',
        'Diagnósticos retrospectivos podem não estar disponíveis em 24h; não utilizar como modelo prospectivo',
        'Diurético e cateterismo até 48h; evidência registrada varia entre as bases',
        'Split por paciente preservado; não selecionar sobreviventes na população principal']},indent=2,ensure_ascii=False),encoding='utf-8')
    for f in feats:
        if tr[f].notna().sum()==0 or e[f].notna().sum()==0 or tr[f].nunique()<2: raise ValueError('Benchmark sem variável: '+f)
    rows=[]
    for nome in ['xgboost','rede_neural']:
        fit=otimizar(tr,feats,nome,3.55,folder/'buscas'/nome)
        for base,d in [('mimic',te),('eicu',e)]:
            r=avaliar(fit,d,folder/base/nome,bootstrap,shap=nome=='xgboost' and base=='mimic')
            rows.append(dict(modelo=nome,base=base,**r))
    pd.DataFrame(rows).to_csv(folder/'comparacao.csv',index=False)

def carregar_painel(base,janela='24h'):
    """Painel de 24h (ou todas as estadias) com as colunas de contexto juntadas por stay_id."""
    d=pd.read_csv(f'data/processados/{base}_ampliado_{"24h" if janela=="24h" else "todas_estadias"}.csv',low_memory=False)
    ctx=pd.read_csv(f'data/processados/{base}_contexto_24h.csv')
    return d.merge(ctx,on='stay_id',how='left',validate='one_to_one')

def rodar(etapa='tudo',bootstrap=500,ic_max_seq=1,sem_curtas=False,aki='creatinina',revisar=()):
    ROOT.mkdir(parents=True,exist_ok=True)
    m,e=carregar_painel('mimic'),carregar_painel('eicu')
    missing=[c for c in TODAS if c not in m]
    if missing: raise ValueError('Preditores ausentes: '+str(missing))
    common=harmonizar(m,e)
    protocol={'versao':SEARCH_VERSION,'semente':42,'janela_principal':24,'features_completo':TODAS,'features_harmonizado':common,
        'cv':'3 folds por paciente; busca somente no treino; OOF pós-seleção apenas para limiar',
        'teste':'mesma partição anteriormente consultada: avaliação exploratória, não confirmação independente nova',
        'externo':'nenhum fit, recalibração ou escolha de configuração no eICU',
        'grades':{n:grade(n) for n in ['xgboost','rede_neural']},
        'interpretacao':'AUC maior não é garantida; comparar intervalos, calibração e cobertura'}
    (ROOT/'protocolo.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    with threadpool_limits(limits=3):
        if etapa in ['tudo','principal']: principais(m,None,TODAS,'completo_ampliado',bootstrap)
        if etapa in ['tudo','externo']: principais(m,e,common,'harmonizado',bootstrap)
        if etapa in ['tudo','ablacoes']: ablacoes(m,bootstrap)
        if etapa in ['tudo','yang']: benchmark(bootstrap,ic_max_seq,sem_curtas,aki,revisar)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--etapa',default='tudo',choices=['tudo','principal','externo','ablacoes','yang']);p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--ic-max-seq',type=int,default=1);p.add_argument('--sem-curtas',action='store_true')
    p.add_argument('--aki',default='creatinina',choices=['creatinina','kdigo24','kdigo48'])
    p.add_argument('--revisar',nargs='*',default=[],choices=['arritmia','ausculta','valvulopatia'])
    a=p.parse_args()
    # Nome importável no pickle: o notebook deve abrir os mesmos modelos depois.
    from experimentos_24h import rodar as executar
    executar(a.etapa,a.bootstrap,a.ic_max_seq,a.sem_curtas,a.aki,tuple(a.revisar))
