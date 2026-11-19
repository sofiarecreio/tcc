import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from painel_completo_24h import (recortar_input, resumir_input, resumir_output,
    resumir_chart, selecionar_feve, ADMISSION, VERSAO)
from variaveis_protocolo import colunas, COMPLETO_BIN, COMPLETO_NUM
from avaliar_protocolo_24h import modelos, treinar, validar_externo


class PainelCompletoTest(unittest.TestCase):
    def coorte(self):
        return pd.DataFrame({'stay_id':[1], 'subject_id':[10], 'intime':pd.to_datetime(['2020-01-01'])})

    def test_contrato_lista(self):
        self.assertEqual(len(colunas('completo')),54)
        self.assertEqual(len(COMPLETO_BIN),17)
        self.assertNotIn('pas_24h',colunas('completo'))
        self.assertNotIn('peso_kg',colunas('completo'))
        self.assertEqual(len(set(ADMISSION.values())),8)

    def test_gcs_mesmo_instante_temperatura(self):
        d=pd.DataFrame({'stay_id':[1]*8,
            'itemid':[220739,223900,223901,220739,223900,223901,223761,223762],
            'charttime':pd.to_datetime(['2020-01-01 01:00']*3+['2020-01-01 02:00']*3+['2020-01-01 03:00']*2),
            'valuenum':[1,5,6,4,1,6,98.6,39]})
        r=resumir_chart(d,[1])
        self.assertEqual(r.gcs_min_24h.iloc[0],11)  # não 1+1+6=8
        self.assertAlmostEqual(r.temperatura_mediana_24h.iloc[0],38)

    def test_input_corte_taxa_e_rewritten(self):
        d=pd.DataFrame({'stay_id':[1]*4,'itemid':[225158]*4,
            'starttime':['2020-01-01 23:00','2020-01-02 00:00','2020-01-01 01:00','2020-01-01 01:00'],
            'endtime':['2020-01-02 02:00','2020-01-02 01:00','2020-01-01 02:00','2020-01-01 02:00'],
            'storetime':['2020-01-01 23:30','2020-01-02 00:00','2020-01-01 02:01','2020-01-02 01:00'],
            'amount':[9000]*4,'rate':[100]*4,'rateuom':['mL/hour']*4,'amountuom':['mL']*4,
            'patientweight':[80]*4,'statusdescription':['FinishedRunning','FinishedRunning','Rewritten','FinishedRunning']})
        r=recortar_input(d,self.coorte())
        self.assertEqual(len(r),1)
        self.assertEqual(r.quantidade.iloc[0],100)  # ignora amount final de 9000

    def test_remedios_bolus_e_volume(self):
        d=pd.DataFrame({'stay_id':[1]*5,'itemid':[221906,229639,228340,225158,220949],
            'amount':[1,2,20,500,100],'rate':[80,0,2,0,0],
            'rateuom':['mcg/min','','mg/hour','',''],'amountuom':['mg','mg','mg','mL','mL'],
            'quantidade':[1,2,20,500,100],'patientweight':[80]*5,
            'ordercategorydescription':['Continuous','Drug Push','Continuous','Bolus','Bolus'],
            'ordercategoryname':['01-Drips','02-Drug','01-Drips','02-Fluids (Crystalloids)','14-Oral/Gastric Intake']})
        r=resumir_input(d,[1,2])
        self.assertEqual(r.loc[1,'noradrenalina_taxa_max_24h'],1)
        self.assertEqual(r.loc[1,'furosemida_dose_total_24h'],20)
        self.assertEqual(r.loc[1,'volume_infundido_24h'],500)
        self.assertEqual(r.loc[1,'bolus_cristaloide_24h'],1)
        self.assertEqual(r.loc[1,'diuretico_alca_24h'],1)
        self.assertEqual(r.loc[2,'n_vasopressores_24h'],0)

    def test_diurese_irrigacao_ausente_nao_zero(self):
        d=pd.DataFrame({'stay_id':[1]*3,'itemid':[226559,227488,227489],
                        'value':[400,100,150],'valueuom':['mL']*3})
        r=resumir_output(d,[1,2],pd.Series({1:50,2:70}))
        self.assertEqual(r.loc[1,'diurese_total_24h'],450)
        self.assertEqual(r.loc[1,'oliguria_24h'],1)
        self.assertTrue(np.isnan(r.loc[2,'oliguria_24h']))
        self.assertEqual(r.loc[2,'diurese_medida'],0)

    def test_feve_janela(self):
        d=pd.DataFrame({'subject_id':[10]*3,'measurement_id':[1,2,3],
            'measurement':['lvef']*3,'result':[.55,80,20],
            'measurement_datetime':['2019-12-31 23:00','2020-01-02 00:00','2019-12-28 00:00']})
        r=selecionar_feve(d,self.coorte())
        self.assertAlmostEqual(r.loc[1],55)

    def test_rede_neural_presente(self):
        cls,reg=modelos(False)
        self.assertIn('rede_neural',cls)
        self.assertIn('rede_neural',reg)
        self.assertFalse(cls['rede_neural'].early_stopping)

    def test_treino_usa_exatamente_painel_completo(self):
        rng=np.random.default_rng(42)
        n=240
        d=pd.DataFrame({c:rng.normal(size=n) for c in COMPLETO_NUM})
        for c in COMPLETO_BIN: d[c]=rng.integers(0,2,n)
        d['idade']=60
        d['genero']='F'
        d['tipo_admissao']='Urgente'
        d['patient_id']=['mimic:'+str(i//2) for i in range(n)]
        d['stay_id']=range(n)
        d['hospital_id']='BIDMC'
        d['raca']='desconhecida'
        d['obito_hospitalar']=0
        d['los_dias']=np.tile([2.,5.,10.,12.],n//4)
        d['versao_painel']=VERSAO
        with TemporaryDirectory() as t, patch('avaliar_protocolo_24h.modelos',return_value=(
            {'logistica':LogisticRegression(max_iter=500)}, {'ridge':Ridge()})):
            p=treinar(d,Path(t),usar_xgb=False,usar_shap=False,bootstrap=3)
            self.assertEqual(p['features'],colunas('completo'))
            with self.assertRaisesRegex(ValueError,'equivalência'):
                validar_externo(d,p,Path(t)/'externo')


if __name__=='__main__': unittest.main()
