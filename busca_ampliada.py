"""Busca exploratoria v5: selecao somente por CV agrupada no treino MIMIC.

Preserva v4. Cada perfil/alvo escolhe configuracao e familia por CV antes de
avaliar teste e eICU. Resultados continuam exploratorios: testes ja consultados.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
from experimentos_24h import carregar_painel, dividir_preservando, harmonizar, TODAS, pipeline, grade, avaliar, hash_dados

OUT=Path('artefatos_protocolo_24h/experimentos_v5')

def rodar(iteracoes=50):
    m,e=carregar_painel('mimic'),carregar_painel('eicu')
    common=harmonizar(m,e,root=OUT)
    tr,te=dividir_preservando(m)
    resultados=[]
    for perfil,features in [('completo',TODAS),('harmonizado',common)]:
        for corte in [3.55,7.]:
            dest=OUT/perfil/f'{corte:g}d'; dest.mkdir(parents=True,exist_ok=True)
            y=tr.los_dias.gt(corte).astype(int)
            folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y,tr.patient_id))
            assert all(not set(tr.iloc[a].patient_id)&set(tr.iloc[b].patient_id) for a,b in folds)
            candidates=[]
            for nome in ['xgboost','random_forest','rede_neural']:
                folder=dest/nome; folder.mkdir(exist_ok=True)
                model=pipeline(features,'rede_neural' if nome=='rede_neural' else 'xgboost')
                if nome=='random_forest':
                    model.set_params(modelo=RandomForestClassifier(n_estimators=300,n_jobs=3,random_state=42))
                    space={'modelo__max_depth':[6,12,None],'modelo__min_samples_leaf':[2,5,15],
                           'modelo__max_features':['sqrt',.5,1.]}
                    n=min(iteracoes,15)
                else:
                    space=grade(nome); n=iteracoes if nome=='xgboost' else min(iteracoes,12)
                    if nome=='xgboost':
                        space['modelo__n_estimators']=[200,350,600]
                        space['modelo__learning_rate']=[.015,.03,.07,.12]
                    else: model.set_params(modelo__max_iter=400)
                space['preparo__num__impute__add_indicator']=[False,True]
                config={'signature':hash_dados(tr,features),'corte':corte,'n_iter':n,'cv':5,
                        'grade':space,'features':features,'seed':42,'version':1}
                config=json.loads(json.dumps(config))
                cache=folder/'busca.joblib'
                if cache.exists() and json.loads((folder/'config.json').read_text())==config:
                    search=joblib.load(cache)
                else:
                    print(f'BUSCA {perfil} {corte} {nome}: {n} configuracoes x 5 folds',flush=True)
                    search=RandomizedSearchCV(model,space,n_iter=n,scoring='roc_auc',cv=folds,
                        n_jobs=1,random_state=42,error_score='raise',refit=True)
                    search.fit(tr[features],y)
                    joblib.dump(search,cache)
                    (folder/'config.json').write_text(json.dumps(config,indent=2))
                    pd.DataFrame(search.cv_results_).to_csv(folder/'cv.csv',index=False)
                candidates.append((search.best_score_,nome,search.best_estimator_))
            score,nome,best=max(candidates,key=lambda x:x[0])
            (dest/'selecao.json').write_text(json.dumps({'vencedor':nome,'auc_cv':score,
                'candidatos':{n:s for s,n,_ in candidates},'criterio':'CV treino apenas',
                'limite':'teste interno e externo previamente consultados; exploratorio'},indent=2))
            oof=cross_val_predict(best,tr[features],y,cv=folds,method='predict_proba')[:,1]
            fpr,tpr,t=roc_curve(y,oof); ok=np.isfinite(t)
            fit={'modelo':best,'features':features,'pacientes_treino':set(tr.patient_id),
                 'corte':corte,'limiar':float(t[ok][np.argmax((tpr-fpr)[ok])])}
            joblib.dump(fit,dest/'modelo_selecionado.joblib')
            for base,d in [('mimic',te)]+([('eicu',e)] if perfil=='harmonizado' else []):
                r=avaliar(fit,d,dest/base)
                resultados.append(dict(perfil=perfil,corte=corte,modelo=nome,base=base,auc_cv=score,**r))
            pd.DataFrame(resultados).to_csv(OUT/'classificacao.csv',index=False)
            print('CONCLUIDO',perfil,corte,nome,score,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--iteracoes',type=int,default=50)
    a=p.parse_args()
    with threadpool_limits(limits=3): rodar(a.iteracoes)
