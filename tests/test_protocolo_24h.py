import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from caminhos_dados import localizar_tabela
from extrair_painel_24h import (codigo_ic, eventos_eicu, eventos_mimic, coorte_eicu, extrair,
                               limpar_eventos, NUMERICAS)
from avaliar_protocolo_24h import dividir, dca, metricas, treinar, validar_externo


class ProtocoloTest(unittest.TestCase):
    def test_caminhos_gzip_e_ambiguidade(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / '3.1/hosp'
            p.mkdir(parents=True)
            (p / 'patients.csv.gz').touch()
            self.assertEqual(localizar_tabela(t, 'patients'), p / 'patients.csv.gz')
            (p / 'patients.csv').touch()
            with self.assertRaises(ValueError): localizar_tabela(t, 'patients')
            with self.assertRaises(FileNotFoundError): localizar_tabela(t, 'inexistente')

    def test_codigos(self):
        v = pd.Series(['428.0', 'I50.9', '401.9, 428.1', '1428', 'I500', None, '4289'])
        self.assertEqual(codigo_ic(v).tolist(), [True, True, True, False, True, False, True])

    def test_unidades_nao_sao_silenciosamente_assumidas(self):
        d = pd.DataFrame({'stay_id': [1]*4, 'variavel': ['creatinina']*4,
                          'valor': [1, 100, 1, 1], 'unidade': ['mg/dL', 'umol/L', None, 'mg/dL']})
        audit = {}
        self.assertEqual(len(limpar_eventos(d, audit)), 2)
        self.assertEqual(sum(audit.values()), 2)

    def test_janela_mimic_e_disponibilidade(self):
        with tempfile.TemporaryDirectory() as t:
            c = pd.DataFrame({'stay_id': [1], 'hadm_id': [10], 'intime': pd.to_datetime(['2020-01-01'])})
            for table, key, item, unit in [('chartevents', 'stay_id', 220045, 'bpm'),
                                          ('labevents', 'hadm_id', 50912, 'mg/dL')]:
                pd.DataFrame({key: [1 if key == 'stay_id' else 10]*4, 'itemid': [item]*4,
                    'charttime': ['2020-01-01 00:00', '2020-01-02 00:00', '2019-12-31 23:59', '2020-01-01 01:00'],
                    'storetime': ['2020-01-01 00:01', '2020-01-02 00:01', '2020-01-01 00:00', '2020-01-02 01:00'],
                    'valuenum': [80 if item == 220045 else 1]*4, 'valueuom': [unit]*4,
                }).to_csv(Path(t) / f'{table}.csv.gz', index=False)
            b = pd.concat(list(eventos_mimic(t, c, {})))
            self.assertEqual(len(b), 2)

    def test_eicu_primeira_uti_e_obitos(self):
        with tempfile.TemporaryDirectory() as t:
            pd.DataFrame({'patientunitstayid': [1, 2], 'patienthealthsystemstayid': [10, 10],
                'uniquepid': ['p', 'p'], 'age': ['> 89', '> 89'], 'gender': ['Female']*2,
                'ethnicity': ['Caucasian']*2, 'hospitalid': [7]*2,
                'hospitaladmitoffset': [-5000, -10], 'unitdischargeoffset': [4000]*2,
                'hospitaldischargestatus': ['Expired']*2}).to_csv(Path(t)/'patient.csv', index=False)
            pd.DataFrame({'patientunitstayid': [1, 2], 'icd9code': ['428.0']*2,
                          'diagnosispriority': ['Primary']*2}).to_csv(Path(t)/'diagnosis.csv', index=False)
            c = coorte_eicu(t)
            self.assertEqual(c.loc[c.primeira_uti_internacao, 'stay_id'].tolist(), [2])
            self.assertTrue(c.obito_hospitalar.eq(1).all())
            self.assertTrue(c.idade.eq(90).all())

    def test_eicu_janela_lab_revisado(self):
        with tempfile.TemporaryDirectory() as t:
            for table, cols in [('vitalPeriodic', ['heartrate', 'respiration', 'sao2', 'systemicmean', 'systemicsystolic']),
                                ('vitalAperiodic', ['noninvasivemean', 'noninvasivesystolic'])]:
                d = {'patientunitstayid': [1,1,1], 'observationoffset': [-1, 0, 1440]}
                d.update({c: [80]*3 for c in cols})
                pd.DataFrame(d).to_csv(Path(t)/f'{table}.csv', index=False)
            pd.DataFrame({'patientunitstayid': [1]*3, 'labname': ['creatinine']*3,
                'labresult': [1., 2., 3.], 'labresultoffset': [0, 10, 1440],
                'labresultrevisedoffset': [10, 1500, 1440],
                'labmeasurenamesystem': ['mg/dL']*3, 'labmeasurenameinterface': ['mg/dL']*3,
            }).to_csv(Path(t)/'lab.csv', index=False)
            b = pd.concat(list(eventos_eicu(t, pd.DataFrame({'stay_id':[1]}), {})))
            self.assertEqual(b.loc[b.variavel.eq('creatinina'), 'valor'].tolist(), [1.])

    def test_metricas_e_dca(self):
        y, p = np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])
        m = metricas(y, p, .5)
        self.assertEqual(m['vpp'], 1)
        self.assertEqual(m['vpn'], 1)
        curva = dca(y, p)
        self.assertAlmostEqual(curva.iloc[49].modelo, .5)
        self.assertTrue(np.isnan(metricas([0,0], [.1,.2], .5)['auc']))

    def test_extracao_mimic_completa_e_media(self):
        with tempfile.TemporaryDirectory() as t:
            raiz = Path(t)
            pd.DataFrame({'subject_id':[1], 'gender':['F'], 'anchor_age':[70],
                          'anchor_year':[2020]}).to_csv(raiz/'patients.csv', index=False)
            pd.DataFrame({'subject_id':[1], 'hadm_id':[10], 'hospital_expire_flag':[1],
                          'race':['WHITE']}).to_csv(raiz/'admissions.csv', index=False)
            pd.DataFrame({'subject_id':[1,1], 'hadm_id':[10,10], 'stay_id':[1,2],
                'intime':['2020-01-01', '2020-01-03'],
                'outtime':['2020-01-01 12:00', '2020-01-08']}).to_csv(raiz/'icustays.csv', index=False)
            pd.DataFrame({'hadm_id':[10], 'icd_code':['I509'], 'icd_version':[10],
                          'seq_num':[1]}).to_csv(raiz/'diagnoses_icd.csv', index=False)
            pd.DataFrame({'stay_id':[2,2], 'itemid':[220045,220045],
                'charttime':['2020-01-03 01:00', '2020-01-03 02:00'],
                'storetime':['2020-01-03 01:05', '2020-01-03 02:05'],
                'valuenum':[60,100], 'valueuom':['bpm','bpm']}).to_csv(raiz/'chartevents.csv', index=False)
            pd.DataFrame({'hadm_id':[10], 'itemid':[50912], 'charttime':['2020-01-03 01:00'],
                'storetime':['2020-01-03 01:05'], 'valuenum':[1.2], 'valueuom':['mg/dL']
            }).to_csv(raiz/'labevents.csv', index=False)
            saida = raiz/'out/painel.csv'
            c = extrair('mimic', raiz, saida)
            self.assertEqual(c.stay_id.tolist(), [2])
            self.assertEqual(c.fc_24h.iloc[0], 80)
            self.assertEqual(c.obito_hospitalar.iloc[0], 1)
            self.assertFalse(c.primeira_uti_internacao.iloc[0])
            self.assertTrue(saida.with_suffix('.json').exists())

    def test_integracao_treino_externo_sem_fit(self):
        rng = np.random.default_rng(42)
        n = 240
        d = pd.DataFrame({c: rng.normal(70, 3, n) for c in NUMERICAS})
        d['idade'] = 60
        d['patient_id'] = ['mimic:' + str(i//2) for i in range(n)]
        d['stay_id'] = np.arange(n)
        d['hospital_id'] = 'BIDMC'
        d['los_dias'] = np.tile([2., 5., 10., 12.], n//4)
        d['obito_hospitalar'] = np.tile([0, 0, 1, 0], n//4)
        d['genero'] = np.tile(['F', 'M'], n//2)
        d['raca'] = 'desconhecida'
        tr, te = dividir(d)
        self.assertFalse(set(tr.patient_id) & set(te.patient_id))
        with tempfile.TemporaryDirectory() as t:
            with patch('avaliar_protocolo_24h.modelos', return_value=(
                    {'logistica': LogisticRegression(max_iter=500)}, {'ridge': Ridge()})):
                pacote = treinar(d, t, usar_xgb=False, bootstrap=5, usar_shap=False, perfil='comum')
            ext = d.iloc[:60].copy()
            ext['patient_id'] = 'eicu:' + ext.patient_id
            ext['genero'] = 'categoria_nova'
            for v in pacote['classificadores'].values():
                v['modelo'].fit = lambda *a, **kw: self.fail('fit indevido no externo')
            for m in pacote['regressores'].values():
                m.fit = lambda *a, **kw: self.fail('fit indevido no externo')
            validar_externo(ext, pacote, Path(t)/'externo', bootstrap=5)
            self.assertTrue((Path(t)/'externo/comparacao_externa.csv').exists())
            self.assertTrue((Path(t)/'interno/logistica_3.55d/sensibilidade_obitos.csv').exists())

    def test_notebooks_sintaxe(self):
        for path in Path('.').glob('*.ipynb'):
            nb = json.loads(path.read_text(encoding='utf-8'))
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    code = ''.join(cell['source'])
                    if not any(line.startswith(('%', '!')) for line in code.splitlines()):
                        compile(code, f'{path}:cell{i}', 'exec')


if __name__ == '__main__': unittest.main()
