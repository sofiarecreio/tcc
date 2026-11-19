"""Efeito isolado dos filtros de população de Yang sobre a AUC (exploratório).

Mantém fixos: painel de 24h, preditores, hiperparâmetros do XGBoost escolhidos
na busca principal (somente treino), partição por paciente e bootstrap.
Muda somente a população, um filtro por vez e os três juntos.

Duas leituras por filtro, sempre nos MESMOS pacientes de teste filtrados:
- modelo_base: treinado no protocolo principal, avaliado no subgrupo filtrado
  (quanto o subgrupo é "mais fácil" sem retreinar);
- modelo_filtrado: retreinado só no subgrupo (efeito de treinar numa população
  homogênea). A diferença entre eles é pareada por paciente.

`sobreviventes` usa um desfecho só conhecido depois da alta: é reprodução
retrospectiva do artigo, não modelo prospectivo de 24h.
Opcional (`--rebusca`): refaz a busca de hiperparâmetros dentro de cada
subgrupo, no treino, como sensibilidade.
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits
from experimentos_24h import (ROOT, TODAS, harmonizar, dividir_preservando, pipeline,
                              avaliar, otimizar, diferenca_pareada)

OUT = ROOT / 'populacao'
FILTROS = {
    'base': [],
    'ic_principal': ['ic_principal'],
    'primeira_uti': ['primeira_uti'],
    'sobreviventes': ['sobreviventes'],
    'tres_filtros': ['ic_principal', 'primeira_uti', 'sobreviventes'],
}


def filtrar(d, filtros):
    """Desconhecido não passa: óbito ausente não conta como sobrevivente."""
    mask = pd.Series(True, index=d.index)
    for f in filtros:
        if f == 'ic_principal': mask &= d.ic_principal.eq(True)
        elif f == 'primeira_uti': mask &= d.primeira_uti_internacao.eq(True)
        elif f == 'sobreviventes': mask &= d.obito_hospitalar.eq(0)
        else: raise ValueError('Filtro desconhecido: ' + f)
    return d[mask].copy()


def ic_ate_posicao(m, max_seq, raiz='data/mimic-iv'):
    """Redefine IC principal no MIMIC: código 428/I50 entre as `max_seq` primeiras posições.

    Pelas regras de codificação, cardiopatia hipertensiva com IC (I11.0/I13.x, 402/404)
    ou o evento agudo costumam ocupar a posição 1, com I50 na 2. A definição de posição 1
    produz ~1/4 da coorte publicada; posições 1-2 aproximam contagem e distribuição de LOS.
    """
    from caminhos_dados import localizar_tabela
    s = pd.read_csv(localizar_tabela(raiz, 'icustays'), usecols=['stay_id', 'hadm_id'])
    d = pd.read_csv(localizar_tabela(raiz, 'diagnoses_icd'), usecols=['hadm_id', 'seq_num', 'icd_code', 'icd_version'])
    code = d.icd_code.astype(str).str.upper().str.replace('.', '', regex=False)
    hf = (d.icd_version.eq(9) & code.str.startswith('428')) | (d.icd_version.eq(10) & code.str.startswith('I50'))
    hadm = set(d.loc[hf & d.seq_num.le(max_seq), 'hadm_id'])
    flag = m.stay_id.map(s.set_index('stay_id').hadm_id).isin(hadm)
    if m.ic_principal.eq(True).gt(flag).any(): raise ValueError('Definição nova deveria conter a de posição 1')
    return m.assign(ic_principal=flag)


def ajustar_fixo(tr, features, params, corte):
    y = tr.los_dias.gt(corte).astype(int)
    cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(tr, y, tr.patient_id))
    model = pipeline(features, 'xgboost').set_params(**params)
    oof = cross_val_predict(model, tr[features], y, cv=cv, method='predict_proba')[:, 1]
    fpr, tpr, t = roc_curve(y, oof); ok = np.isfinite(t)
    model.fit(tr[features], y)
    return {'modelo': model, 'features': features, 'pacientes_treino': set(tr.patient_id),
            'corte': corte, 'limiar': float(t[ok][np.argmax((tpr - fpr)[ok])]),
            'auc_cv': roc_auc_score(y, oof)}


def prever(fit, d):
    if set(d.patient_id) & fit['pacientes_treino']: raise ValueError('Paciente de treino na avaliação')
    p = d[['patient_id', 'stay_id']].copy()
    p['alvo'] = d.los_dias.gt(fit['corte']).astype(int).to_numpy()
    p['probabilidade'] = fit['modelo'].predict_proba(d[fit['features']])[:, 1]
    return p


def descrever(d, corte):
    return dict(estadias=len(d), pacientes=d.patient_id.nunique(),
                prevalencia=float(d.los_dias.gt(corte).mean()),
                obitos=float(d.obito_hospitalar.eq(1).mean()))


def rodar(bootstrap=500, rebusca=False, ic_max_seq=1):
    global OUT
    m = pd.read_csv('data/processados/mimic_ampliado_24h.csv', low_memory=False)
    if ic_max_seq > 1:
        m = ic_ate_posicao(m, ic_max_seq)
        OUT = ROOT / f'populacao_ic_seq1a{ic_max_seq}'
    e = pd.read_csv('data/processados/eicu_ampliado_24h.csv', low_memory=False)
    perfis = {'completo_ampliado': TODAS, 'harmonizado': harmonizar(m, e)}
    tr_all, te_all = dividir_preservando(m)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ns = [], []
    for corte in [3.55, 7.]:
        for perfil, feats in perfis.items():
            busca = joblib.load(ROOT / perfil / 'buscas' / f'xgboost_{corte:g}d' / 'ajuste.joblib')
            assert busca['features'] == feats, 'Preditores diferem da busca principal'
            bases = [('mimic', te_all)] + ([('eicu', e)] if perfil == 'harmonizado' else [])
            base_fit = None
            for nome, filtros in FILTROS.items():
                tr, te = filtrar(tr_all, filtros), filtrar(te_all, filtros)
                dest = OUT / perfil / f'{nome}_{corte:g}d'
                print(f'POPULAÇÃO {perfil} / {nome} / >{corte}d: treino {len(tr)}', flush=True)
                fit = ajustar_fixo(tr, feats, busca['params'], corte)
                if nome == 'base': base_fit = fit
                for base, data in bases:
                    d = filtrar(data, filtros)
                    ns.append(dict(corte=corte, perfil=perfil, filtro=nome, conjunto=base + '_avaliacao', **descrever(d, corte)))
                    r = avaliar(fit, d, dest / base, bootstrap)
                    rb = avaliar(base_fit, d, dest / f'{base}_modelo_base', bootstrap)
                    delta = ({} if nome == 'base' else
                             diferenca_pareada(prever(fit, d), prever(base_fit, d), bootstrap))
                    rows.append(dict(
                        corte=corte, perfil=perfil, filtro=nome, base=base, n=r['n'],
                        prevalencia=r['prevalencia'], auc_cv_treino_filtrado=fit['auc_cv'],
                        auc_modelo_filtrado=r['auc'], ic95_inf=r['auc_ic95_inf'], ic95_sup=r['auc_ic95_sup'],
                        auc_modelo_base_no_subgrupo=rb['auc'], base_ic95_inf=rb['auc_ic95_inf'], base_ic95_sup=rb['auc_ic95_sup'],
                        delta_filtrado_menos_base=delta.get('delta_completo_menos_variante', 0.),
                        delta_ic95_inf=delta.get('ic95_inf', np.nan), delta_ic95_sup=delta.get('ic95_sup', np.nan),
                        brier_modelo_filtrado=r['brier']))
                ns.append(dict(corte=corte, perfil=perfil, filtro=nome, conjunto='mimic_treino', **descrever(tr, corte)))
                pd.DataFrame(rows).to_csv(OUT / 'resultados.csv', index=False)
    pd.DataFrame(ns).drop_duplicates().to_csv(OUT / 'tamanhos.csv', index=False)
    if rebusca:
        sens = []
        with threadpool_limits(limits=3):
            for nome, filtros in FILTROS.items():
                tr, te = filtrar(tr_all, filtros), filtrar(te_all, filtros)
                fit = otimizar(tr, TODAS, 'xgboost', 3.55, OUT / 'rebusca' / nome)
                r = avaliar(fit, te, OUT / 'rebusca' / nome / 'mimic', bootstrap)
                sens.append(dict(filtro=nome, auc_cv_busca=fit['score_cv'], **r, **{'params': json.dumps(fit['params'])}))
                pd.DataFrame(sens).to_csv(OUT / 'rebusca' / 'resultados.csv', index=False)
    (OUT / 'protocolo.json').write_text(json.dumps({
        'objetivo': 'Isolar o efeito dos filtros populacionais de Yang na AUC',
        'fixos': 'painel 24h, preditores, hiperparâmetros da busca principal, partição por paciente, bootstrap por paciente',
        'variantes': FILTROS,
        'comparacao_pareada': 'modelo filtrado vs modelo base nos mesmos pacientes filtrados',
        'atencao': ['sobreviventes usa desfecho posterior à previsão: retrospectivo',
                    'teste interno já consultado: exploratório',
                    'subgrupos pequenos têm IC largos; comparar IC, não só ponto'],
        'rebusca': rebusca,
        'ic_principal_mimic': f'428/I50 nas posições 1 a {ic_max_seq}; eICU mantém diagnosispriority=Primary'}, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--rebusca', action='store_true')
    p.add_argument('--ic-max-seq', type=int, default=1)
    a = p.parse_args()
    from experimento_populacao import rodar as executar
    with threadpool_limits(limits=3):
        executar(a.bootstrap, a.rebusca, a.ic_max_seq)
