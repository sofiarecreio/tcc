"""Arritmia maligna e ausculta no formato de Yang, em 24h, salvas à parte dos painéis.

MIMIC: ritmo cardíaco registrado (item 220048) e ausculta dos quatro lobos.
eICU: não há ritmo estruturado; usa diagnósticos até 24h mais história de FA/TV/FV
registrada até 24h. Ausculta: exame físico mais avaliação de enfermagem
(`nurseAssessment`), ambos esparsos no eICU.
Valvulopatia (retrospectiva, como no artigo): MIMIC = CID valvar entre as 3 primeiras
posições da internação; eICU = diagnóstico em qualquer momento ou história até 24h.
Saída: data/processados/{base}_preditores_yang.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from caminhos_dados import localizar_tabela
from extrair_fontes_ampliadas import CACHE, partes, carregar
from criterios_clinicos import janela, arritmia_maligna, ausculta_yang

SAIDA = Path('data/processados')
NURSE = CACHE / 'eicu' / 'nurseassessment_ausculta.pkl'


def flag(ids, positivos):
    return pd.Series(pd.Index(ids).isin(positivos).astype(int), index=pd.Index(ids, name='stay_id'))


def valvar_mimic(ids, max_seq=3):
    """CID valvar relevante para a internação: posições 1 a `max_seq` (todas: 37% vs 10,6%)."""
    s = pd.read_csv(localizar_tabela('data/mimic-iv', 'icustays'), usecols=['stay_id', 'hadm_id'])
    d = pd.read_csv(localizar_tabela('data/mimic-iv', 'diagnoses_icd'), usecols=['hadm_id', 'seq_num', 'icd_code', 'icd_version'])
    code = d.icd_code.astype(str).str.upper().str.replace('.', '', regex=False)
    valve = ((d.icd_version.eq(9) & code.str.match(r'^(394|395|396|397|424[0-3])')) |
             (d.icd_version.eq(10) & code.str.match(r'^(I0[5-8]|I3[4-9])')))
    hadm = set(d.loc[valve & d.seq_num.le(max_seq), 'hadm_id'])
    return flag(ids, s.loc[s.hadm_id.isin(hadm), 'stay_id'])


def mimic(ids):
    xs = []
    for b in partes('mimic', 'chartevents'):
        xs.append(janela(b[b.stay_id.isin(ids) & b.itemid.isin([220048, 223986, 223987, 223988, 223989])]))
    c = pd.concat(xs, ignore_index=True)
    rh = c[c.itemid.eq(220048)]
    out = pd.DataFrame({'arritmia_maligna_24h': flag(ids, rh.loc[arritmia_maligna(rh.value), 'stay_id'])})
    lungs = c[c.itemid.ne(220048)][['stay_id', 'value']].rename(columns={'value': 'texto'})
    out['valvulopatia_yang'] = valvar_mimic(ids)
    return out.join(ausculta_yang(lungs, ids))


def nurse_assessment(ids):
    if not NURSE.exists():
        xs = []
        for b in pd.read_csv(localizar_tabela('data/eicu', 'nurseAssessment'), chunksize=1_000_000, low_memory=False):
            a = b.cellattributepath.fillna('').str.lower()
            xs.append(b[a.str.contains('breath sounds', regex=False)])
        pd.concat(xs, ignore_index=True).to_pickle(NURSE)
    n = pd.read_pickle(NURSE).rename(columns={'patientunitstayid': 'stay_id', 'nurseassessoffset': 'offset',
                                               'nurseassessentryoffset': 'disponivel'})
    n = janela(n[n.stay_id.isin(ids) & n.cellattributepath.str.lower().str.contains('breath sounds', regex=False)])
    return n.assign(texto=n.cellattributevalue.astype(str))[['stay_id', 'texto']]


def eicu(ids):
    dg = janela(carregar('eicu', 'diagnosis'))
    h = janela(carregar('eicu', 'pastHistory'))
    ht = h.pasthistorypath.fillna('') + ' ' + h.pasthistoryvalue.fillna('')
    pos = set(dg.loc[arritmia_maligna(dg.diagnosisstring), 'stay_id']) | set(h.loc[arritmia_maligna(ht), 'stay_id'])
    out = pd.DataFrame({'arritmia_maligna_24h': flag(ids, pos)})
    painel = pd.read_csv(SAIDA / 'eicu_ampliado_todas_estadias.csv', usecols=['stay_id', 'valvulopatia_retro', 'valvulopatia_previa'])
    out['valvulopatia_yang'] = flag(ids, painel.loc[painel.valvulopatia_retro.eq(1) | painel.valvulopatia_previa.eq(1), 'stay_id'])
    exam = janela(carregar('eicu', 'physicalExam'))
    ex = exam.physicalexampath.fillna('').str.lower()
    lungs = exam[ex.str.contains(r'auscult|breath sounds|rhonch|rales|crackl|wheez|stridor|bronchial', regex=True)].copy()
    lungs['texto'] = (lungs.physicalexampath.fillna('').astype(str) + ' ' + lungs.physicalexamvalue.fillna('').astype(str)
                      + ' ' + lungs.physicalexamtext.fillna('').astype(str))
    lungs = pd.concat([lungs[['stay_id', 'texto']], nurse_assessment(ids)], ignore_index=True)
    return out.join(ausculta_yang(lungs, ids))


def calcular(base):
    ids = pd.read_csv(SAIDA / f'{base}_ampliado_todas_estadias.csv', usecols=['stay_id']).stay_id
    out = (mimic if base == 'mimic' else eicu)(set(ids)).reindex(ids).fillna(0).astype(int)
    out.reset_index().to_csv(SAIDA / f'{base}_preditores_yang.csv', index=False)
    resumo = {c: round(float(out[c].mean()), 4) for c in out}
    (SAIDA / f'{base}_preditores_yang.json').write_text(json.dumps(resumo, indent=2), encoding='utf-8')
    print(base, json.dumps(resumo), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('base', choices=['mimic', 'eicu'])
    calcular(p.parse_args().base)
