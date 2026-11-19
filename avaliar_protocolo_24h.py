"""Treino MIMIC e validação externa congelada"""
import argparse
import json
import warnings
import hashlib
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef, mean_absolute_error,
    mean_squared_error, r2_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.exceptions import ConvergenceWarning

from extrair_painel_24h import NUMERICAS, CATEGORICAS
from variaveis_protocolo import PERFIS, colunas


def assinatura_dados(d, perfil):
    campos = ['patient_id', 'stay_id', 'los_dias'] + colunas(perfil)
    return hashlib.sha256(pd.util.hash_pandas_object(d[campos], index=False).to_numpy().tobytes()).hexdigest()


def preparar(perfil='completo'):
    p = PERFIS[perfil]
    return ColumnTransformer([
        ('num', Pipeline([('imputar', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('escala', StandardScaler())]), p['numericas'] + p['binarias']),
        ('cat', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), p['categoricas'])])


def modelos(usar_xgb=True):
    classificadores = {'logistica': LogisticRegression(max_iter=2000),
        'random_forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                                max_depth=8, random_state=42, n_jobs=-1)}
    regressores = {'mediana': DummyRegressor(strategy='median'), 'ridge': Ridge(alpha=10),
        'random_forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                                               max_depth=8, random_state=42, n_jobs=-1)}
    # Sem early_stopping: o split interno padrão do MLP separa linhas, não pacientes.
    mlp = dict(hidden_layer_sizes=(64, 32), alpha=1.0, max_iter=300,
               early_stopping=False, n_iter_no_change=20, random_state=42)
    classificadores['rede_neural'] = MLPClassifier(**mlp)
    regressores['rede_neural'] = MLPRegressor(**mlp)
    if usar_xgb:
        from xgboost import XGBClassifier, XGBRegressor
        kwargs = dict(n_estimators=250, max_depth=3, learning_rate=.04, subsample=.8,
                      colsample_bytree=.8, reg_lambda=5, random_state=42, n_jobs=4)
        classificadores['xgboost'] = XGBClassifier(**kwargs, eval_metric='logloss')
        regressores['xgboost'] = XGBRegressor(**kwargs, objective='reg:squarederror')
    return classificadores, regressores


def validar_tabela(d, perfil='completo'):
    obrigatorias = ['patient_id', 'stay_id', 'hospital_id', 'los_dias', 'obito_hospitalar',
                    'raca'] + colunas(perfil)
    falta = set(obrigatorias) - set(d)
    if falta: raise ValueError(f'Colunas ausentes: {sorted(falta)}')
    if d.patient_id.isna().any() or d.stay_id.isna().any() or d.stay_id.duplicated().any():
        raise ValueError('IDs ausentes ou estadias duplicadas.')
    if not (np.isfinite(d.los_dias) & d.los_dias.gt(1) & d.idade.ge(18)).all():
        raise ValueError('Painel deve conter somente adultos presentes na UTI após 24h.')
    if perfil == 'completo':
        from painel_completo_24h import VERSAO
        if 'versao_painel' not in d or not d.versao_painel.eq(VERSAO).all():
            raise ValueError('Painel completo ausente ou desatualizado: execute painel_completo_24h.py.')
        for c in PERFIS[perfil]['binarias']:
            if not d[c].dropna().isin([0, 1]).all():
                raise ValueError(f'Variável booleana inválida: {c}')


def dividir(d):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(d, groups=d.patient_id))
    assert not set(d.iloc[tr].patient_id) & set(d.iloc[te].patient_id)
    return d.iloc[tr].copy(), d.iloc[te].copy()


def metricas(y, p, limiar):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = p >= limiar
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    def div(a, b): return float(a / b) if b else np.nan
    duas = len(np.unique(y)) == 2
    return dict(n=len(y), positivos=int(y.sum()), prevalencia=float(y.mean()),
        auc=roc_auc_score(y, p) if duas else np.nan,
        auc_pr=average_precision_score(y, p) if duas else np.nan,
        brier=brier_score_loss(y, p), sensibilidade=div(tp, tp + fn),
        especificidade=div(tn, tn + fp), vpp=div(tp, tp + fp), vpn=div(tn, tn + fn),
        acuracia=div(tp + tn, len(y)), kappa=cohen_kappa_score(y, pred) if duas else np.nan,
        mcc=matthews_corrcoef(y, pred) if duas else np.nan,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), limiar=float(limiar))


def dca(y, p):
    y, p = np.asarray(y), np.asarray(p)
    linhas = []
    for t in np.linspace(.01, .99, 99):
        pred = p >= t
        linhas.append(dict(limiar=t, modelo=((pred & (y == 1)).sum() -
            (pred & (y == 0)).sum() * t / (1 - t)) / len(y),
            tratar_todos=y.mean() - (1 - y.mean()) * t / (1 - t), tratar_ninguem=0))
    return pd.DataFrame(linhas)


def auc_ic(y, p, grupos, n=500):
    # Reamostra pacientes inteiros: estadias repetidas permanecem juntas.
    rng = np.random.default_rng(42)
    grupos = np.asarray(grupos)
    unicos = np.unique(grupos)
    indices = {g: np.flatnonzero(grupos == g) for g in unicos}
    valores = []
    y, p = np.asarray(y), np.asarray(p)
    for _ in range(n):
        idx = np.concatenate([indices[g] for g in rng.choice(unicos, len(unicos), replace=True)])
        if len(np.unique(y[idx])) == 2: valores.append(roc_auc_score(y[idx], p[idx]))
    return np.quantile(valores, [.025, .975]) if valores else [np.nan, np.nan]


def avaliar_classificador(d, p, corte, limiar, pasta, bootstrap=500):
    pasta.mkdir(parents=True, exist_ok=True)
    y = d.los_dias.gt(corte).astype(int).to_numpy()
    resultado = metricas(y, p, limiar)
    resultado['auc_ic95_inf'], resultado['auc_ic95_sup'] = auc_ic(y, p, d.patient_id, bootstrap)
    pd.DataFrame([resultado]).to_csv(pasta / 'metricas.csv', index=False)
    pred = d[['patient_id', 'stay_id', 'los_dias', 'obito_hospitalar']].copy()
    pred['alvo'] = y
    pred['probabilidade'] = p
    pred.to_csv(pasta / 'previsoes.csv', index=False)
    obs, prob = calibration_curve(y, p, n_bins=10, strategy='quantile')
    pd.DataFrame({'probabilidade_media': prob, 'frequencia_observada': obs}).to_csv(pasta / 'calibracao.csv', index=False)
    dca(y, p).to_csv(pasta / 'dca.csv', index=False)
    subgrupos = d[['genero', 'raca', 'hospital_id']].copy()
    subgrupos['faixa_etaria'] = pd.cut(d.idade, [17, 65, 80, np.inf], labels=['18-65', '66-80', '>80'])
    linhas = []
    for campo in subgrupos:
        for grupo in subgrupos[campo].dropna().unique():
            mask = subgrupos[campo].eq(grupo).to_numpy()
            m = metricas(y[mask], p[mask], limiar)
            m.update(campo=campo, grupo=str(grupo),
                     interpretacao='amostra pequena' if mask.sum() < 50 else 'descritiva')
            linhas.append(m)
    pd.DataFrame(linhas).to_csv(pasta / 'subgrupos.csv', index=False)
    sens = []
    for valor, rotulo in [(0, 'sobreviventes'), (1, 'obitos_hospitalares')]:
        mask = d.obito_hospitalar.eq(valor).to_numpy()
        if mask.any(): sens.append(dict(grupo=rotulo, **metricas(y[mask], p[mask], limiar)))
    pd.DataFrame(sens).to_csv(pasta / 'sensibilidade_obitos.csv', index=False)
    return resultado


def avaliar_regressor(d, p, pasta):
    pasta.mkdir(parents=True, exist_ok=True)
    p = np.maximum(p, 0)
    r = dict(n=len(d), mae=mean_absolute_error(d.los_dias, p),
             rmse=float(np.sqrt(mean_squared_error(d.los_dias, p))),
             r2=r2_score(d.los_dias, p) if len(d) > 1 else np.nan)
    pd.DataFrame([r]).to_csv(pasta / 'metricas.csv', index=False)
    saida = d[['patient_id', 'stay_id', 'los_dias']].copy()
    saida['dias_previstos'] = p
    saida.to_csv(pasta / 'previsoes.csv', index=False)
    return r


def explicar(modelo, d, pasta):
    import shap
    x = modelo.named_steps['preparo'].transform(d)
    x = pd.DataFrame(x, columns=modelo.named_steps['preparo'].get_feature_names_out())
    x = x.sample(min(500, len(x)), random_state=42)
    valores = shap.TreeExplainer(modelo.named_steps['modelo'])(x)
    pd.DataFrame({'variavel': x.columns, 'shap_absoluto_medio': np.abs(valores.values).mean(axis=0)}
                 ).sort_values('shap_absoluto_medio', ascending=False).to_csv(pasta / 'shap.csv', index=False)


def treinar(d, pasta, usar_xgb=True, bootstrap=500, usar_shap=True, perfil='completo'):
    validar_tabela(d, perfil)
    numericas = PERFIS[perfil]['numericas'] + PERFIS[perfil]['binarias']
    tr, te = dividir(d)
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    pd.concat([tr[['patient_id', 'stay_id']].assign(particao='treino'),
               te[['patient_id', 'stay_id']].assign(particao='teste')]).to_csv(pasta / 'divisao.csv', index=False)
    tr[numericas].isna().mean().rename('fracao_ausente_treino').to_csv(pasta / 'ausencias_treino.csv')
    if tr[numericas].notna().sum().eq(0).any():
        raise ValueError('Há variável inteiramente ausente no treino. Confira unidades no JSON de extração.')
    classificadores, regressores = modelos(usar_xgb)
    pacote = {'classificadores': {}, 'regressores': {}, 'features': colunas(perfil), 'perfil': perfil,
              'versao_protocolo': 2, 'pacientes_treino': set(tr.patient_id),
              'pacientes_teste': set(te.patient_id), 'assinatura_dados': assinatura_dados(d, perfil)}
    resultados = []
    avisos = []
    for corte in [3.55, 7.0]:
        y = tr.los_dias.gt(corte).astype(int)
        cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(tr, y, tr.patient_id))
        for it, iv in cv:
            if y.iloc[it].nunique() < 2 or y.iloc[iv].nunique() < 2:
                raise ValueError(f'Poucos pacientes/classes para CV de 5 partes no corte {corte}.')
        for nome, estimador in classificadores.items():
            print(f'Treinando {nome}, corte {corte} dias...', flush=True)
            m = Pipeline([('preparo', preparar(perfil)), ('modelo', clone(estimador))])
            # Pré-processamento reajustado DENTRO de cada fold.
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always', ConvergenceWarning)
                oof = cross_val_predict(m, tr, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
                m.fit(tr, y)
            avisos.extend(dict(modelo=nome, alvo=str(corte), aviso=str(w.message)) for w in ws)
            fpr, tpr, limiares = roc_curve(y, oof)
            finitos = np.isfinite(limiares)
            limiar = float(limiares[finitos][np.argmax((tpr - fpr)[finitos])])
            chave = f'{nome}_{corte:g}d'
            pacote['classificadores'][chave] = dict(modelo=m, limiar=limiar, corte=corte,
                                                    auc_oof=float(roc_auc_score(y, oof)))
            destino = pasta / 'interno' / chave
            r = avaliar_classificador(te, m.predict_proba(te)[:, 1], corte, limiar, destino, bootstrap)
            r.update(modelo=nome, corte=corte, auc_oof=roc_auc_score(y, oof))
            resultados.append(r)
            if nome == 'xgboost' and usar_shap: explicar(m, te, destino)
    for nome, estimador in regressores.items():
        reg = estimador if nome == 'mediana' else TransformedTargetRegressor(
            regressor=estimador, func=np.log1p, inverse_func=np.expm1)
        print(f'Treinando regressão: {nome}...', flush=True)
        m = Pipeline([('preparo', preparar(perfil)), ('modelo', reg)])
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always', ConvergenceWarning)
            m.fit(tr, tr.los_dias)
        avisos.extend(dict(modelo=nome, alvo='dias', aviso=str(w.message)) for w in ws)
        pacote['regressores'][nome] = m
        avaliar_regressor(te, m.predict(te), pasta / 'interno' / f'regressao_{nome}')
    pd.DataFrame(resultados).to_csv(pasta / 'comparacao_interna.csv', index=False)
    pd.DataFrame(avisos, columns=['modelo', 'alvo', 'aviso']).to_csv(pasta / 'avisos_ajuste.csv', index=False)
    comparar_rede_xgboost(pasta / 'interno', bootstrap)
    # A referência é fixada pelo treino antes de consultar o eICU.
    pacote['referencia_por_corte'] = {str(c): max(
        (k for k, v in pacote['classificadores'].items() if v['corte'] == c),
        key=lambda k: pacote['classificadores'][k]['auc_oof']) for c in [3.55, 7.]}
    joblib.dump(pacote, pasta / 'modelos.joblib')
    (pasta / 'protocolo.json').write_text(json.dumps({
        'semente': 42, 'janela_horas': 24, 'mimic_esperado': '3.1', 'eicu_esperado': '2.0',
        'perfil': perfil, 'versao_protocolo': 2,
        'rede_neural': 'MLP (64,32), alpha=1, 300 épocas máximas, sem early stopping com split de linhas',
        'obitos_excluidos': False, 'los_minimo_exclusivo': 1,
        'selecao': 'AUC OOF no treino; hiperparâmetros fixos',
        'referencias': pacote['referencia_por_corte'],
        'sensibilidade_obitos': 'avaliação estratificada do modelo fixo, sem reestimar',
        'risco_concorrente': 'não estimado; LOS observado inclui saídas por óbito',
        'bibliotecas': {n: version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
        'features': pacote['features']}, ensure_ascii=False, indent=2), encoding='utf-8')
    return pacote


def validar_externo(d, pacote, pasta, bootstrap=500):
    perfil = pacote.get('perfil', 'comum')
    if perfil != 'comum':
        raise ValueError('O painel completo inclui FEVE e tratamentos ainda sem equivalência validada no eICU. '
                         'Para validação externa use os modelos do perfil comum; não impute colunas inteiras inexistentes.')
    validar_tabela(d, perfil)
    numericas = PERFIS[perfil]['numericas'] + PERFIS[perfil]['binarias']
    if set(d.patient_id) & (pacote['pacientes_treino'] | pacote.get('pacientes_teste', set())):
        raise ValueError('Pacientes externos sobrepostos ao treino.')
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    d[numericas].isna().mean().rename('fracao_ausente_eicu').to_csv(pasta / 'ausencias_externas.csv')
    ausentes = d[numericas].columns[d[numericas].notna().sum().eq(0)].tolist()
    if ausentes:
        raise ValueError(f'Variáveis sem nenhum valor externo: {ausentes}. Confira mapeamentos/unidades antes de validar.')
    resultados = []
    for chave, v in pacote['classificadores'].items():
        # Nenhum fit, calibração ou escolha de limiar na base externa.
        p = v['modelo'].predict_proba(d)[:, 1]
        r = avaliar_classificador(d, p, v['corte'], v['limiar'], pasta / chave, bootstrap)
        resultados.append(dict(modelo=chave, **r))
    for nome, m in pacote['regressores'].items():
        avaliar_regressor(d, m.predict(d), pasta / f'regressao_{nome}')
    pd.DataFrame(resultados).to_csv(pasta / 'comparacao_externa.csv', index=False)
    comparar_rede_xgboost(pasta, bootstrap)


def comparar_rede_xgboost(pasta, bootstrap=500):
    """IC da diferença de AUC com o MESMO sorteio de pacientes para ambos."""
    linhas = []
    for corte in [3.55, 7.0]:
        arquivos = [pasta / f'{m}_{corte:g}d' / 'previsoes.csv' for m in ['rede_neural', 'xgboost']]
        if not all(p.exists() for p in arquivos): continue
        a, b = [pd.read_csv(p).sort_values('stay_id').reset_index(drop=True) for p in arquivos]
        if not a[['stay_id','patient_id','alvo']].equals(b[['stay_id','patient_id','alvo']]):
            raise ValueError('Comparação pareada exige as mesmas estadias, pacientes e alvos.')
        grupos = a.patient_id.to_numpy()
        unicos = np.unique(grupos)
        indices = {g: np.flatnonzero(grupos == g) for g in unicos}
        y, pn, px = a.alvo.to_numpy(), a.probabilidade.to_numpy(), b.probabilidade.to_numpy()
        rng, deltas = np.random.default_rng(42), []
        for _ in range(bootstrap):
            idx = np.concatenate([indices[g] for g in rng.choice(unicos, len(unicos), replace=True)])
            if len(np.unique(y[idx])) == 2:
                deltas.append(roc_auc_score(y[idx],pn[idx])-roc_auc_score(y[idx],px[idx]))
        lo, hi = np.quantile(deltas,[.025,.975]) if deltas else [np.nan,np.nan]
        linhas.append(dict(corte=corte, auc_rede=roc_auc_score(y,pn), auc_xgboost=roc_auc_score(y,px),
            delta_auc=roc_auc_score(y,pn)-roc_auc_score(y,px), delta_ic95_inf=lo, delta_ic95_sup=hi))
    if linhas: pd.DataFrame(linhas).to_csv(pasta / 'rede_neural_vs_xgboost.csv',index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('etapa', choices=['treinar', 'externo'])
    p.add_argument('--dados', type=Path)
    p.add_argument('--saida', type=Path)
    p.add_argument('--perfil', choices=PERFIS, default='completo')
    p.add_argument('--sem-xgboost', action='store_true', help='Somente baselines; não é execução completa.')
    p.add_argument('--sem-shap', action='store_true')
    p.add_argument('--bootstrap', type=int, default=500)
    a = p.parse_args()
    base = 'mimic' if a.etapa == 'treinar' else 'eicu'
    saida = a.saida or Path('artefatos_protocolo_24h') / a.perfil
    nome = 'mimic_completo_24h.csv' if a.perfil == 'completo' else f'{base}_24h.csv'
    d = pd.read_csv(a.dados or Path('data/processados') / nome)
    if a.etapa == 'treinar':
        treinar(d, saida, not a.sem_xgboost, a.bootstrap, not a.sem_shap, a.perfil)
    else:
        validar_externo(d, joblib.load(saida / 'modelos.joblib'), saida / 'externo', a.bootstrap)
