"""Contexto e tendências no modelo principal de 24h, e APACHE IV como referência no eICU.

Mesmo desenho do experimento de população: painel de 24h, partição por paciente,
hiperparâmetros da busca principal (fixos) e diferenças pareadas nos mesmos
pacientes de teste. A variante com todos os grupos também é reajustada com nova
busca no treino (sensibilidade). Nada é escolhido pelo teste ou pelo eICU.
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from caminhos_dados import localizar_tabela
from avaliar_protocolo_24h import auc_ic
# Comparação feita sobre o protocolo v3, antes de adotar o contexto (v4).
from experimentos_24h import V3 as ROOT, TODAS_V3 as TODAS, harmonizar, dividir_preservando, avaliar, otimizar, diferenca_pareada
from experimento_populacao import ajustar_fixo, prever
from contexto_24h import UNIDADES_UTI, ORIGENS, LABS_TENDENCIA

OUT = ROOT / 'contexto'
GRUPOS = {'unidade': [f'unidade_{u}' for u in UNIDADES_UTI],
          'origem': [f'origem_{o}' for o in ORIGENS],
          'pre_uti': ['horas_pre_uti'],
          'tendencias': [f'{n}_{s}_24h' for n in LABS_TENDENCIA for s in ['ultimo', 'delta']]}
VARIANTES = {'base': [], **{'+' + g: [g] for g in GRUPOS}, '+todos': list(GRUPOS)}


def carregar(base):
    d = pd.read_csv(f'data/processados/{base}_ampliado_24h.csv', low_memory=False)
    return d.merge(pd.read_csv(f'data/processados/{base}_contexto_24h.csv'), on='stay_id', how='left', validate='one_to_one')


def apache(e, fits, bootstrap):
    a = pd.read_csv(localizar_tabela('data/eicu', 'apachePatientResult'),
                    usecols=['patientunitstayid', 'apacheversion', 'predictediculos'])
    a = a[a.apacheversion.eq('IVa') & a.predictediculos.gt(0)].drop_duplicates('patientunitstayid')
    d = e.merge(a.rename(columns={'patientunitstayid': 'stay_id'}), on='stay_id')
    rows = []
    for corte, fit in fits.items():
        y = d.los_dias.gt(corte).astype(int)
        ap = d[['patient_id', 'stay_id']].assign(alvo=y.to_numpy(), probabilidade=d.predictediculos.to_numpy())
        nosso = prever(fit, d)
        lo, hi = auc_ic(y.to_numpy(), d.predictediculos.to_numpy(), d.patient_id, bootstrap)
        dd = diferenca_pareada(nosso, ap, bootstrap)
        rows.append(dict(corte=corte, n=len(d), cobertura_apache=len(d) / len(e), auc_apache_iva=roc_auc_score(y, d.predictediculos),
                         apache_ic95_inf=lo, apache_ic95_sup=hi, auc_modelo_harmonizado=roc_auc_score(y, nosso.probabilidade),
                         delta_modelo_menos_apache=dd['delta_completo_menos_variante'], delta_ic95_inf=dd['ic95_inf'], delta_ic95_sup=dd['ic95_sup']))
    pd.DataFrame(rows).to_csv(OUT / 'apache_iva_eicu.csv', index=False)


def rodar(bootstrap=500):
    m, e = carregar('mimic'), carregar('eicu')
    perfis = {'completo_ampliado': TODAS, 'harmonizado': harmonizar(m, e, TODAS, ROOT)}
    tr, te = dividir_preservando(m)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, base_harm = [], {}
    for corte in [3.55, 7.]:
        for perfil, feats in perfis.items():
            busca = joblib.load(ROOT / perfil / 'buscas' / f'xgboost_{corte:g}d' / 'ajuste.joblib')
            assert busca['features'] == feats, 'Preditores diferem da busca principal'
            bases = [('mimic', te)] + ([('eicu', e)] if perfil == 'harmonizado' else [])
            for nome, grupos in VARIANTES.items():
                cols = feats + [c for g in grupos for c in GRUPOS[g]]
                print(f'CONTEXTO {perfil} / {nome} / >{corte}d', flush=True)
                fit = ajustar_fixo(tr, cols, busca['params'], corte)
                if nome == 'base':
                    ref = fit
                    if perfil == 'harmonizado': base_harm[corte] = fit
                for base, d in bases:
                    r = avaliar(fit, d, OUT / perfil / f'{nome}_{corte:g}d' / base, bootstrap)
                    dd = {} if nome == 'base' else diferenca_pareada(prever(fit, d), prever(ref, d), bootstrap)
                    rows.append(dict(corte=corte, perfil=perfil, variante=nome, base=base, n=r['n'],
                                     auc_cv_treino=fit['auc_cv'], auc=r['auc'], ic95_inf=r['auc_ic95_inf'], ic95_sup=r['auc_ic95_sup'],
                                     brier=r['brier'], delta_vs_base=dd.get('delta_completo_menos_variante', 0.),
                                     delta_ic95_inf=dd.get('ic95_inf', np.nan), delta_ic95_sup=dd.get('ic95_sup', np.nan)))
                pd.DataFrame(rows).to_csv(OUT / 'resultados.csv', index=False)
            # Sensibilidade: nova busca de hiperparâmetros só no treino, com todos os grupos.
            cols = feats + [c for g in GRUPOS for c in GRUPOS[g]]
            fit = otimizar(tr, cols, 'xgboost', corte, OUT / perfil / 'rebusca' / f'todos_{corte:g}d')
            for base, d in bases:
                r = avaliar(fit, d, OUT / perfil / 'rebusca' / f'todos_{corte:g}d' / base, bootstrap, shap=base == 'mimic')
                dd = diferenca_pareada(prever(fit, d), prever(ref, d), bootstrap)
                rows.append(dict(corte=corte, perfil=perfil, variante='+todos (nova busca)', base=base, n=r['n'],
                                 auc_cv_treino=fit['score_cv'], auc=r['auc'], ic95_inf=r['auc_ic95_inf'], ic95_sup=r['auc_ic95_sup'],
                                 brier=r['brier'], delta_vs_base=dd['delta_completo_menos_variante'],
                                 delta_ic95_inf=dd['ic95_inf'], delta_ic95_sup=dd['ic95_sup']))
            pd.DataFrame(rows).to_csv(OUT / 'resultados.csv', index=False)
    apache(e, base_harm, bootstrap)
    (OUT / 'protocolo.json').write_text(json.dumps({
        'objetivo': 'Ganho de preditores de contexto e tendência conhecidos até 24h',
        'grupos': GRUPOS, 'fixos': 'painel 24h, partição por paciente, hiperparâmetros da busca principal',
        'sensibilidade': 'nova busca no treino para +todos', 'apache': 'APACHE IVa predictediculos, eICU, referência não externa',
        'atencao': ['teste interno já consultado: exploratório',
                    'APACHE foi desenvolvido com dados de UTIs americanas semelhantes ao eICU']}, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--bootstrap', type=int, default=500)
    from experimento_contexto import rodar as executar
    with threadpool_limits(limits=3):
        executar(a.parse_args().bootstrap)
