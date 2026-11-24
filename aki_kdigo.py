"""AKI KDIGO com creatinina E diurese, em 24h (prospectivo) e 48h (retrospectivo).

Não altera os painéis: salva data/processados/{base}_aki_kdigo.csv, que os
experimentos juntam pelo stay_id quando pedido. A coluna de 24h pode entrar no
protocolo de previsão em 24h; as de 48h usam dados posteriores ao instante da
previsão e servem apenas para aproximar o artigo de Yang.
"""
import argparse
import json
import numpy as np
import pandas as pd
from extrair_fontes_ampliadas import partes, carregar
from extrair_painel_24h import FAIXAS, UNIDADES
from painel_completo_24h import URINA
from criterios_clinicos import aki_creatinina, aki_diurese, aki_kdigo

SAIDA = 'data/processados'
HORAS = [24, 48]


def creatinina(base, ids):
    xs = []
    for b in partes(base, 'labevents' if base == 'mimic' else 'lab'):
        if base == 'mimic':
            b = b[b.itemid.isin([50912, 52546, 52024])].rename(columns={'valuenum': 'valor', 'valueuom': 'unidade'})
        else:
            b = b[b.labname.str.lower().str.strip().eq('creatinine')].copy()
            b['unidade'] = b.labmeasurenamesystem.fillna(b.labmeasurenameinterface)
            b = b.rename(columns={'labresult': 'valor'})
        b = b[b.stay_id.isin(ids)].copy()
        b['valor'] = pd.to_numeric(b.valor, errors='coerce')
        u = b.unidade.fillna('').str.lower().str.replace(' ', '', regex=False)
        xs.append(b.loc[b.valor.between(*FAIXAS['creatinina']) & u.isin(UNIDADES['creatinina']),
                        ['stay_id', 'offset', 'disponivel', 'valor']])
    return pd.concat(xs, ignore_index=True)


def urina(base, ids):
    if base == 'mimic':
        b = carregar('mimic', 'outputevents')
        b = b[b.valueuom.eq('mL') & b.value.between(0, 5000) & b.itemid.isin(URINA) & b.stay_id.isin(ids)].copy()
        b['valor'] = np.where(b.itemid.eq(227488), -b.value, b.value)  # irrigante, como mimic-code
        return b[['stay_id', 'offset', 'disponivel', 'valor']]
    # Mesmos rótulos e deduplicação do painel eICU (painel_ampliado.fluidos_eicu).
    b = carregar('eicu', 'intakeOutput')
    b = b.drop_duplicates('intakeoutputid').sort_values('disponivel').drop_duplicates(
        ['stay_id', 'offset', 'cellpath', 'celllabel'], keep='last').copy()
    path = b.cellpath.fillna('').str.lower(); label = b.celllabel.fillna('').str.lower()
    unit = path.str.contains(r'\(ml\)', regex=True) | label.str.contains(r'\(ml\)', regex=True)
    summary = label.str.contains(r'total|cumulative|balance|net|shift|24 hour|8 hr|count|occurrence|unmeasured', regex=True)
    urine = (path.str.contains(r'\|output \(ml\)\|', regex=True) & unit & ~summary &
             label.str.contains(r'urine|urinary|foley|void|nephrost', regex=True) & ~label.str.contains('irrig', regex=False))
    v = pd.to_numeric(b.cellvaluenumeric, errors='coerce')
    b['valor'] = v.where(v.between(0, 30000))
    return b.loc[urine & b.stay_id.isin(ids), ['stay_id', 'offset', 'disponivel', 'valor']]


def calcular(base):
    painel = pd.read_csv(f'{SAIDA}/{base}_ampliado_todas_estadias.csv', low_memory=False,
                         usecols=['stay_id', 'peso_kg', 'aki_creatinina_estagio_24h'])
    ids = painel.stay_id
    peso = painel.set_index('stay_id').peso_kg
    print(base + ': creatinina...', flush=True); cr = creatinina(base, set(ids))
    print(base + ': diurese...', flush=True); uo = urina(base, set(ids))
    out = pd.DataFrame(index=pd.Index(ids, name='stay_id'))
    for h in HORAS:
        print(f'{base}: estágios em {h}h...', flush=True)
        out[f'aki_creatinina_estagio_{h}h'] = aki_creatinina(cr, ids, h)
        out[f'aki_diurese_estagio_{h}h'] = aki_diurese(uo, peso, ids, h)
        out[f'aki_kdigo_estagio_{h}h'] = aki_kdigo(out[f'aki_creatinina_estagio_{h}h'], out[f'aki_diurese_estagio_{h}h'])
    # Contrato: a creatinina em 24h tem de reproduzir a coluna do painel.
    old = painel.set_index('stay_id').aki_creatinina_estagio_24h
    diff = ~((old == out.aki_creatinina_estagio_24h) | (old.isna() & out.aki_creatinina_estagio_24h.isna()))
    if diff.any(): raise ValueError(f'{base}: creatinina 24h difere do painel em {int(diff.sum())} estadias')
    out = out.drop(columns='aki_creatinina_estagio_24h').reset_index()
    out.to_csv(f'{SAIDA}/{base}_aki_kdigo.csv', index=False)
    resumo = {c: {str(k): round(float(v), 4) for k, v in out[c].value_counts(normalize=True, dropna=False).sort_index().items()}
              for c in out if c != 'stay_id'}
    with open(f'{SAIDA}/{base}_aki_kdigo.json', 'w', encoding='utf-8') as f: json.dump(resumo, f, indent=2)
    print(json.dumps(resumo, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('base', choices=['mimic', 'eicu'])
    calcular(p.parse_args().base)
