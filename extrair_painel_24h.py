"""Painel comum MIMIC-IV/eICU, sem depender dos caches de versões anteriores."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from caminhos_dados import localizar_tabela

# Unidades comuns: bpm, rpm, %, mmHg, mg/dL, mEq/L e segundos.
# BUN é nitrogênio ureico, não ureia total.
CHART = {220045: 'fc', 220210: 'fr', 220277: 'spo2',
         220052: 'pam', 220181: 'pam', 220050: 'pas', 220179: 'pas'}
LAB = {50912: 'creatinina', 51006: 'bun', 50983: 'sodio',
       50971: 'potassio', 50931: 'glicose', 51275: 'ptt'}
LAB_EICU = {'creatinine': 'creatinina', 'bun': 'bun', 'sodium': 'sodio',
            'potassium': 'potassio', 'glucose': 'glicose', 'ptt': 'ptt'}
FAIXAS = {'fc': (20, 280), 'fr': (4, 70), 'spo2': (50, 100),
          'pam': (20, 250), 'pas': (30, 300), 'creatinina': (.1, 30),
          'bun': (1, 300), 'sodio': (100, 180), 'potassio': (1.5, 9),
          'glicose': (20, 1500), 'ptt': (10, 200)}
UNIDADES = {'fc': {'bpm'}, 'fr': {'insp/min', 'breaths/min'},
            'spo2': {'%'}, 'pam': {'mmhg'}, 'pas': {'mmhg'},
            'creatinina': {'mg/dl'}, 'bun': {'mg/dl'},
            'glicose': {'mg/dl'}, 'sodio': {'meq/l', 'mmol/l'},
            'potassio': {'meq/l', 'mmol/l'}, 'ptt': {'sec', 'seconds', 's'}}
NUMERICAS = ['idade'] + [f'{x}_24h' for x in FAIXAS]
CATEGORICAS = ['genero']
TABELAS = {'mimic': ['patients', 'admissions', 'diagnoses_icd', 'icustays',
                     'chartevents', 'labevents'],
           'eicu': ['patient', 'diagnosis', 'vitalPeriodic', 'vitalAperiodic', 'lab']}


def ler(raiz, nome, colunas, blocos=False):
    return pd.read_csv(localizar_tabela(raiz, nome), usecols=colunas,
                       chunksize=500_000 if blocos else None, low_memory=False)


def codigo_ic(codigos):
    return codigos.astype('string').str.upper().str.contains(
        r'(?:^|[^A-Z0-9])(?:428(?:\.[0-9]+|[0-9]*)|I50(?:\.[A-Z0-9]+|[A-Z0-9]*))(?=$|[^A-Z0-9])',
        regex=True, na=False)


def raca_grupo(valores):
    def grupo(v):
        v = str(v).upper()
        if 'WHITE' in v or 'CAUCASIAN' in v: return 'branca'
        if 'BLACK' in v or 'AFRICAN' in v: return 'negra'
        if 'ASIAN' in v: return 'asiatica'
        if 'HISPANIC' in v or 'LATINO' in v: return 'hispanica'
        if 'NATIVE' in v or 'AMERICAN INDIAN' in v: return 'indigena'
        if v in {'NAN', '<NA>', ''} or 'UNKNOWN' in v or 'UNABLE' in v or 'DECLINED' in v:
            return 'desconhecida'
        return 'outras'
    return valores.map(grupo)


def coorte_mimic(raiz):
    p = ler(raiz, 'patients', ['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    a = ler(raiz, 'admissions', ['subject_id', 'hadm_id', 'hospital_expire_flag', 'race'])
    s = ler(raiz, 'icustays', ['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'])
    for c in ['intime', 'outtime']: s[c] = pd.to_datetime(s[c], errors='coerce', format='mixed')
    # Primeira UTI da internação, determinada ANTES do filtro de IC/24h.
    first = s.sort_values(['intime', 'stay_id']).drop_duplicates('hadm_id')['stay_id']
    s['primeira_uti_internacao'] = s.stay_id.isin(first)
    diag = ler(raiz, 'diagnoses_icd', ['hadm_id', 'icd_code', 'icd_version', 'seq_num'])
    code = diag.icd_code.astype('string').str.upper().str.replace('.', '', regex=False)
    ic = ((diag.icd_version.eq(9) & code.str.startswith('428')) |
          (diag.icd_version.eq(10) & code.str.startswith('I50'))).fillna(False)
    s = s[s.hadm_id.isin(diag.loc[ic, 'hadm_id'])].copy()
    s['ic_principal'] = s.hadm_id.isin(diag.loc[ic & diag.seq_num.eq(1), 'hadm_id'])
    s = s.merge(p, on='subject_id', validate='many_to_one').merge(
        a, on=['subject_id', 'hadm_id'], validate='many_to_one')
    s['idade'] = s.anchor_age + s.intime.dt.year - s.anchor_year
    s['idade'] = s.idade.clip(upper=90)  # mesma categoria top-coded do eICU
    s['los_dias'] = (s.outtime - s.intime).dt.total_seconds() / 86400
    s['obito_hospitalar'] = pd.to_numeric(s.hospital_expire_flag, errors='coerce')
    s['genero'] = s.gender.where(s.gender.isin(['M', 'F']), 'desconhecido')
    s['raca'] = raca_grupo(s.race)
    s['patient_id'] = 'mimic:' + s.subject_id.astype(str)
    s['hospital_id'] = 'BIDMC'
    return s


def coorte_eicu(raiz):
    s = ler(raiz, 'patient', ['patientunitstayid', 'patienthealthsystemstayid',
        'uniquepid', 'age', 'gender', 'ethnicity', 'hospitalid',
        'hospitaladmitoffset', 'unitdischargeoffset', 'hospitaldischargestatus'])
    if s.uniquepid.isna().any():
        raise ValueError('eICU sem uniquepid: não é possível garantir split por paciente.')
    first = s.sort_values(['hospitaladmitoffset', 'patientunitstayid'],
                         ascending=[False, True]).drop_duplicates('patienthealthsystemstayid')
    s['primeira_uti_internacao'] = s.patientunitstayid.isin(first.patientunitstayid)
    ids, primary = set(), set()
    for d in ler(raiz, 'diagnosis', ['patientunitstayid', 'icd9code', 'diagnosispriority'], True):
        ic = codigo_ic(d.icd9code)
        ids.update(d.loc[ic, 'patientunitstayid'])
        primary.update(d.loc[ic & d.diagnosispriority.str.lower().eq('primary'), 'patientunitstayid'])
    s = s[s.patientunitstayid.isin(ids)].copy()
    s['ic_principal'] = s.patientunitstayid.isin(primary)
    s['idade'] = pd.to_numeric(s.age.astype('string').str.replace(r'>\s*89', '90', regex=True), errors='coerce')
    s['los_dias'] = pd.to_numeric(s.unitdischargeoffset, errors='coerce') / 1440
    s['obito_hospitalar'] = s.hospitaldischargestatus.str.lower().map({'expired': 1, 'alive': 0})
    s['genero'] = s.gender.str.lower().map({'male': 'M', 'female': 'F'}).fillna('desconhecido')
    s['raca'] = raca_grupo(s.ethnicity)
    s['patient_id'] = 'eicu:' + s.uniquepid.astype(str)
    return s.rename(columns={'patientunitstayid': 'stay_id', 'hospitalid': 'hospital_id'})


def limpar_eventos(d, auditoria, unidades=True):
    d = d.copy()
    d['valor'] = pd.to_numeric(d.valor, errors='coerce')
    valido = pd.Series(False, index=d.index)
    for nome, (lo, hi) in FAIXAS.items():
        m = d.variavel.eq(nome)
        ok = d.valor.between(lo, hi)
        if unidades:
            u = d.unidade.astype('string').str.lower().str.replace(' ', '', regex=False)
            ok &= u.isin(UNIDADES[nome])
            rejeitadas = d.loc[m & ~u.isin(UNIDADES[nome]), 'unidade'].fillna('(ausente)').value_counts()
            for unidade, n in rejeitadas.items():
                chave = f'{nome}|{unidade}'
                auditoria[chave] = auditoria.get(chave, 0) + int(n)
        valido |= m & ok
    return d.loc[valido, ['stay_id', 'variavel', 'valor']]


def eventos_mimic(raiz, coorte, auditoria):
    for tabela, mapa, chave in [('chartevents', CHART, 'stay_id'), ('labevents', LAB, 'hadm_id')]:
        cols = [chave, 'itemid', 'charttime', 'storetime', 'valuenum', 'valueuom']
        for b in ler(raiz, tabela, cols, True):
            b = b[b.itemid.isin(mapa) & b[chave].isin(coorte[chave])].copy()
            if b.empty: continue
            b = b.merge(coorte[[chave] + ([] if chave == 'stay_id' else ['stay_id']) + ['intime']], on=chave)
            t = pd.to_datetime(b.charttime, errors='coerce', format='mixed')
            registro = pd.to_datetime(b.storetime, errors='coerce', format='mixed')
            limite = b.intime + pd.Timedelta(hours=24)
            # Resultado coletado E registrado antes do instante de predição.
            b = b[(t >= b.intime) & (t < limite) & registro.notna() & (registro < limite)].copy()
            b['variavel'] = b.itemid.map(mapa)
            b = b.rename(columns={'valuenum': 'valor', 'valueuom': 'unidade'})
            yield limpar_eventos(b, auditoria)


def eventos_eicu(raiz, coorte, auditoria):
    ids = set(coorte.stay_id)
    mapas = [('vitalPeriodic', {'heartrate': 'fc', 'respiration': 'fr', 'sao2': 'spo2',
                               'systemicmean': 'pam', 'systemicsystolic': 'pas'}),
             ('vitalAperiodic', {'noninvasivemean': 'pam', 'noninvasivesystolic': 'pas'})]
    for tabela, mapa in mapas:
        for b in ler(raiz, tabela, ['patientunitstayid', 'observationoffset'] + list(mapa), True):
            b = b[b.patientunitstayid.isin(ids) & b.observationoffset.ge(0) & b.observationoffset.lt(1440)]
            b = b.rename(columns={'patientunitstayid': 'stay_id', **mapa})
            b = b.melt(id_vars='stay_id', value_vars=list(mapa.values()), var_name='variavel', value_name='valor')
            yield limpar_eventos(b, auditoria, unidades=False)
    for b in ler(raiz, 'lab', ['patientunitstayid', 'labname', 'labresult', 'labresultoffset',
                             'labresultrevisedoffset', 'labmeasurenamesystem', 'labmeasurenameinterface'], True):
        b = b[b.patientunitstayid.isin(ids) & b.labresultoffset.ge(0) & b.labresultoffset.lt(1440)
              & b.labresultrevisedoffset.notna() & b.labresultrevisedoffset.lt(1440)].copy()
        b['variavel'] = b.labname.astype('string').str.lower().str.strip().map(LAB_EICU)
        b['unidade'] = b.labmeasurenamesystem.fillna(b.labmeasurenameinterface)
        b = b.rename(columns={'patientunitstayid': 'stay_id', 'labresult': 'valor'})
        yield limpar_eventos(b[b.variavel.notna()], auditoria)


def extrair(base, raiz, saida):
    for nome in TABELAS[base]: localizar_tabela(raiz, nome)
    c = coorte_mimic(raiz) if base == 'mimic' else coorte_eicu(raiz)
    fluxo = {'ic_antes_filtros': len(c), 'adultos': int(c.idade.ge(18).sum())}
    c = c[c.idade.ge(18) & c.los_dias.gt(1) & np.isfinite(c.los_dias)].copy()
    if c.empty: raise ValueError('Nenhuma estadia elegível (adultos com IC e LOS >24h).')
    if c.stay_id.duplicated().any(): raise ValueError('stay_id duplicado na coorte.')
    fluxo['presentes_24h'] = len(c)
    auditoria = {}
    eventos = eventos_mimic(raiz, c, auditoria) if base == 'mimic' else eventos_eicu(raiz, c, auditoria)
    # Somas/contagens por bloco: memória limitada e média exata de TODOS os registros.
    acumulados = []
    for b in eventos:
        if not b.empty:
            acumulados.append(b.groupby(['stay_id', 'variavel']).valor.agg(['sum', 'count']))
        if len(acumulados) >= 20:
            acumulados = [pd.concat(acumulados).groupby(level=[0, 1]).sum()]
    if not acumulados: raise ValueError('Nenhum sinal/exame válido; confira unidades e janela temporal.')
    ag = pd.concat(acumulados).groupby(level=[0, 1]).sum()
    painel = (ag['sum'] / ag['count']).unstack('variavel').add_suffix('_24h')
    c = c.merge(painel, left_on='stay_id', right_index=True, how='left', validate='one_to_one')
    for v in NUMERICAS:
        if v not in c: c[v] = np.nan
    cols = ['patient_id', 'stay_id', 'hospital_id', 'los_dias', 'obito_hospitalar',
            'primeira_uti_internacao', 'ic_principal', 'raca'] + NUMERICAS + CATEGORICAS
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    c[cols].to_csv(saida, index=False)
    meta = {'base': base, 'versao_esperada': '3.1' if base == 'mimic' else '2.0',
            'versao_verificada_automaticamente': False, 'fluxo': fluxo,
            'janela': '[0,24h), registros disponíveis antes de 24h',
            'resumo': 'média de registros para sinais e exames nas duas bases',
            'unidades_descartadas': auditoria,
            'ausencia': c[NUMERICAS + CATEGORICAS].isna().mean().to_dict(),
            'fontes': {n: str(localizar_tabela(raiz, n).resolve()) for n in TABELAS[base]}}
    saida.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(c)} estadias gravadas em {saida}; confira o JSON de unidades/ausências.')
    return c[cols]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('base', choices=TABELAS)
    p.add_argument('--raiz', type=Path)
    p.add_argument('--saida', type=Path)
    a = p.parse_args()
    raiz = a.raiz or Path('data') / ('mimic-iv' if a.base == 'mimic' else 'eicu')
    extrair(a.base, raiz, a.saida or Path('data/processados') / f'{a.base}_24h.csv')
