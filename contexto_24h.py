"""Preditores de contexto e tendência conhecidos até 24h, harmonizados MIMIC/eICU.

- Tipo de UTI (unidade de entrada) e origem imediata antes da UTI, como indicadores.
- Horas entre a admissão hospitalar e a entrada na UTI.
- Tendências de laboratório em [0, 24h): último valor e variação (último - primeiro).
Saída: data/processados/{base}_contexto_24h.csv (não altera os painéis).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from caminhos_dados import localizar_tabela
from extrair_fontes_ampliadas import partes
from extrair_painel_24h import FAIXAS, UNIDADES
from criterios_clinicos import janela

SAIDA = Path('data/processados')
UNIDADES_UTI = ['coronariana', 'cardiotoracica', 'clinica', 'cirurgica', 'mista', 'neuro']
ORIGENS = ['pronto_socorro', 'centro_cirurgico', 'enfermaria', 'outra_uti', 'direta_externa']
# CCU-CTICU (eICU) é unidade cardíaca combinada; agrupada com coronariana por predominar IC clínica.
MAPA_UNIDADE = {
    'Coronary Care Unit (CCU)': 'coronariana', 'Cardiac ICU': 'coronariana', 'CCU-CTICU': 'coronariana',
    'Cardiac Vascular Intensive Care Unit (CVICU)': 'cardiotoracica', 'CTICU': 'cardiotoracica', 'CSICU': 'cardiotoracica',
    'Medical Intensive Care Unit (MICU)': 'clinica', 'MICU': 'clinica',
    'Surgical Intensive Care Unit (SICU)': 'cirurgica', 'Trauma SICU (TSICU)': 'cirurgica', 'SICU': 'cirurgica',
    'Medical/Surgical Intensive Care Unit (MICU/SICU)': 'mista', 'Med-Surg ICU': 'mista',
    'Neuro Intermediate': 'neuro', 'Neuro Surgical Intensive Care Unit (Neuro SICU)': 'neuro',
    'Neuro Stepdown': 'neuro', 'Neuro ICU': 'neuro'}
MAPA_ORIGEM_EICU = {
    'Emergency Department': 'pronto_socorro', 'Chest Pain Center': 'pronto_socorro',
    'Operating Room': 'centro_cirurgico', 'Recovery Room': 'centro_cirurgico', 'PACU': 'centro_cirurgico',
    'Other ICU': 'outra_uti', 'ICU': 'outra_uti', 'ICU to SDU': 'outra_uti',
    'Direct Admit': 'direta_externa', 'Other Hospital': 'direta_externa'}
LABS_TENDENCIA = {'creatinina': ({50912, 52546, 52024}, 'creatinine', FAIXAS['creatinina'], UNIDADES['creatinina']),
                  'bun': ({51006}, 'bun', FAIXAS['bun'], UNIDADES['bun']),
                  'lactato': ({50813}, 'lactate', (.2, 30), {'mmol/l'})}


def origem_mimic(careunit):
    u = careunit.fillna('')
    return np.select([u.isna() | u.eq(''), u.str.contains('Emergency'), u.isin(['PACU']),
                      u.str.contains('Intensive Care|\\(CCU\\)|\\(CVICU\\)', regex=True)],
                     ['direta_externa', 'pronto_socorro', 'centro_cirurgico', 'outra_uti'], 'enfermaria')


def indicadores(out, serie, nome, categorias):
    for c in categorias: out[f'{nome}_{c}'] = serie.eq(c).astype(int)


def tendencias(base, ids):
    xs = []
    for b in partes(base, 'labevents' if base == 'mimic' else 'lab'):
        b = b[b.stay_id.isin(ids)]
        for nome, (items, label, faixa, unidades) in LABS_TENDENCIA.items():
            if base == 'mimic':
                q = b[b.itemid.isin(items)].rename(columns={'valuenum': 'valor', 'valueuom': 'unidade'})
            else:
                q = b[b.labname.str.lower().str.strip().eq(label)].copy()
                q['unidade'] = q.labmeasurenamesystem.fillna(q.labmeasurenameinterface)
                q = q.rename(columns={'labresult': 'valor'})
            q = q.assign(nome=nome, valor=pd.to_numeric(q.valor, errors='coerce'))
            u = q.unidade.fillna('').str.lower().str.replace(' ', '', regex=False)
            xs.append(q.loc[q.valor.between(*faixa) & u.isin(unidades), ['stay_id', 'offset', 'disponivel', 'nome', 'valor']])
    d = janela(pd.concat(xs, ignore_index=True)).sort_values(['stay_id', 'nome', 'offset'])
    g = d.groupby(['stay_id', 'nome']).valor
    out = pd.DataFrame({'ultimo': g.last(), 'delta': g.last() - g.first(), 'n': g.size()}).unstack('nome')
    res = pd.DataFrame(index=pd.Index(sorted(ids), name='stay_id'))
    for nome in LABS_TENDENCIA:
        res[f'{nome}_ultimo_24h'] = out[('ultimo', nome)]
        # Variação só com duas medidas ou mais; uma medida não informa tendência.
        res[f'{nome}_delta_24h'] = out[('delta', nome)].where(out[('n', nome)].ge(2))
    return res


def mimic():
    ids = pd.read_csv(SAIDA / 'mimic_ampliado_todas_estadias.csv', usecols=['stay_id']).stay_id
    s = pd.read_csv(localizar_tabela('data/mimic-iv', 'icustays'), usecols=['hadm_id', 'stay_id', 'first_careunit', 'intime'])
    s = s[s.stay_id.isin(ids)].copy(); s['intime'] = pd.to_datetime(s.intime)
    a = pd.read_csv(localizar_tabela('data/mimic-iv', 'admissions'), usecols=['hadm_id', 'admittime'])
    s = s.merge(a, on='hadm_id'); s['admittime'] = pd.to_datetime(s.admittime)
    t = pd.read_csv(localizar_tabela('data/mimic-iv', 'transfers'), usecols=['hadm_id', 'careunit', 'intime'])
    t = t[t.hadm_id.isin(s.hadm_id) & t.careunit.notna()].copy(); t['intime'] = pd.to_datetime(t.intime)
    m = t.merge(s[['hadm_id', 'stay_id', 'intime']].rename(columns={'intime': 'uti'}), on='hadm_id')
    prev = m[m.intime.lt(m.uti)].sort_values('intime').groupby('stay_id').careunit.last()
    out = pd.DataFrame(index=pd.Index(s.stay_id, name='stay_id'))
    st = s.set_index('stay_id')
    indicadores(out, st.first_careunit.map(MAPA_UNIDADE), 'unidade', UNIDADES_UTI)
    indicadores(out, pd.Series(origem_mimic(prev.reindex(out.index)), index=out.index), 'origem', ORIGENS)
    out['horas_pre_uti'] = ((st.intime - st.admittime).dt.total_seconds() / 3600).clip(lower=0)
    return out.join(tendencias('mimic', set(ids)))


def eicu():
    ids = pd.read_csv(SAIDA / 'eicu_ampliado_todas_estadias.csv', usecols=['stay_id']).stay_id
    p = pd.read_csv(localizar_tabela('data/eicu', 'patient'),
                    usecols=['patientunitstayid', 'unittype', 'unitadmitsource', 'hospitaladmitoffset'])
    p = p.rename(columns={'patientunitstayid': 'stay_id'}).set_index('stay_id').loc[ids]
    out = pd.DataFrame(index=p.index)
    indicadores(out, p.unittype.map(MAPA_UNIDADE), 'unidade', UNIDADES_UTI)
    # Floor, step-down, observação etc. = enfermaria; origem ausente fica com todos os indicadores 0.
    origem = p.unitadmitsource.map(MAPA_ORIGEM_EICU).fillna('enfermaria').where(p.unitadmitsource.notna())
    indicadores(out, origem, 'origem', ORIGENS)
    out['horas_pre_uti'] = (-pd.to_numeric(p.hospitaladmitoffset, errors='coerce') / 60).clip(lower=0)
    return out.join(tendencias('eicu', set(ids)))


def calcular(base):
    out = mimic() if base == 'mimic' else eicu()
    if out.index.duplicated().any(): raise ValueError('Estadia duplicada')
    out.reset_index().to_csv(SAIDA / f'{base}_contexto_24h.csv', index=False)
    resumo = {c: (round(float(out[c].mean()), 4) if not c.endswith(('_24h', 'pre_uti')) else
                  {'mediana': round(float(out[c].median()), 3), 'ausente': round(float(out[c].isna().mean()), 3)})
              for c in out}
    (SAIDA / f'{base}_contexto_24h.json').write_text(json.dumps(resumo, indent=2), encoding='utf-8')
    print(base, json.dumps(resumo), flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('base', choices=['mimic', 'eicu'])
    calcular(a.parse_args().base)
