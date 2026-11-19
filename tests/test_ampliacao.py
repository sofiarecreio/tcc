import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from criterios_clinicos import janela, aki_creatinina, sofa, pafi, taxa_mcg_kg_min, ausculta, arritmias
from experimentos_24h import preparo, TODAS, NOVAS, IO, YANG

class CriteriosTest(unittest.TestCase):
    def test_arritmia_supraventricular_nao_e_ventricular(self):
        a,v=arritmias(pd.Series(['SVT (Supra Ventricular Tachycardia)','supraventricular tachycardia',
            'VT (Ventricular Tachycardia)','AF (Atrial Fibrillation)','3rd AV (Complete Heart Block)']))
        self.assertEqual(a.tolist(),[False,False,True,True,True])
        self.assertEqual(v.tolist(),[False,False,True,False,False])

    def test_corte_exclui_registro_tardio(self):
        d=pd.DataFrame({'offset':[0,1439,1440,10,-1],'disponivel':[1,1440,1440,20,0]})
        self.assertEqual(janela(d).index.tolist(),[0,3])

    def test_aki_nao_usa_creatinina_futura_ou_mesmo_instante(self):
        d=pd.DataFrame({'stay_id':[1,1,1,2,3,3],'offset':[-60,120,1500,20,10,10],
            'disponivel':[-30,130,1501,21,11,12],'valor':[1.,1.4,4.,4.,1.,4.]})
        r=aki_creatinina(d,[1,2,3])
        self.assertEqual(r.loc[1],1)
        self.assertTrue(np.isnan(r.loc[2]))
        self.assertTrue(np.isnan(r.loc[3]))

    def test_aki_creatinina_alta_sem_subida_nao_e_estagio3(self):
        d=pd.DataFrame({'stay_id':[1,1,2,2],'offset':[-30,100,-30,100],
            'disponivel':[-20,110,-20,110],'valor':[4.2,4.2,1.,3.]})
        r=aki_creatinina(d,[1,2])
        self.assertEqual(r.loc[1],0);self.assertEqual(r.loc[2],3)

    def test_pafi_nao_combina_extremos_de_horarios_distintos(self):
        a=pd.DataFrame({'stay_id':[1],'offset':[60.],'valor':[80.]})
        f=pd.DataFrame({'stay_id':[1,1],'offset':[30.,120.],'valor':[40.,100.]})
        s=pd.DataFrame({'stay_id':[1],'offset':[30.],'valor':[1]})
        r=pafi(a,f,s,[1,2])
        self.assertEqual(r.loc[1,'pafi_min_24h'],200)
        self.assertTrue(np.isnan(r.loc[2,'pafi_min_24h']))

    def test_ausculta_negacao_nao_vira_achado(self):
        d=pd.DataFrame({'stay_id':[1,2,3],'texto':['Pulmonary/Wheezes/no wheezing; Rhonchi/No rhonchi',
            'diffuse wheezing; bibasilar rales','no decreased breath sounds']})
        r=ausculta(d,[1,2,3,4])
        self.assertEqual(r.loc[1,'sibilos_24h'],0)
        self.assertEqual(r.loc[1,'roncos_24h'],0)
        self.assertEqual(r.loc[2,'sibilos_24h'],1)
        self.assertEqual(r.loc[3,'murmurio_diminuido_24h'],0)
        self.assertTrue(np.isnan(r.loc[4,'ausculta_alterada_24h']))

    def test_sofa_zero_vasopressor_nao_recebe_tres_pontos(self):
        d=pd.DataFrame({'plaquetas_min_24h':[160], 'bilirrubina_max_24h':[1.], 'gcs_min_24h':[15],
            'creatinina_max_24h':[1.], 'diurese_total_24h':[1000], 'noradrenalina_taxa_max_24h':[0.],
            'adrenalina_taxa_max_24h':[0.], 'dopamina_taxa_max_24h':[0.], 'dobutamina_24h':[0],
            'pam_min_24h':[80], 'pafi_min_24h':[450.], 'pafi_suporte_min_24h':[np.nan]})
        r=sofa(d)
        self.assertEqual(r.sofa_24h.iloc[0],0)
        d['noradrenalina_taxa_max_24h']=.2
        self.assertEqual(sofa(d).sofa_cardiovascular.iloc[0],4)
        d['noradrenalina_24h']=1
        d['noradrenalina_taxa_max_24h']=np.nan
        self.assertTrue(np.isnan(sofa(d).sofa_cardiovascular.iloc[0]))

    def test_eicu_volume_nao_soma_contagens_ou_totais(self):
        from painel_ampliado import fluidos_eicu
        d=pd.DataFrame({'stay_id':[1]*5,'offset':[10,20,20,30,40],'disponivel':[11,21,22,31,41],
            'intakeoutputid':[1,2,3,4,5],
            'cellpath':['x|intake (ml)|saline','x|output (ml)|urine','x|output (ml)|urine',
                        'x|output (ml)|count','x|output (ml)|total'],
            'celllabel':['Saline','Urine','Urine','Urine Count','Output Total'],
            'cellvaluenumeric':[1000,100,120,2,500]})
        with patch('painel_ampliado.carregar',return_value=d):
            r=fluidos_eicu([1,2],pd.Series({1:100,2:100}),{})
        self.assertEqual(r.loc[1,'volume_infundido_24h'],1000)
        self.assertEqual(r.loc[1,'diurese_total_24h'],120)
        self.assertEqual(r.loc[1,'balanco_hidrico_24h'],880)
        self.assertTrue(np.isnan(r.loc[2,'diurese_total_24h']))

    def test_dose_nao_confunde_ml_hr_com_mcg(self):
        r=taxa_mcg_kg_min(pd.Series([80,80,60]),pd.Series(['mcg/min','mL/hr','mg/hr']),pd.Series([80,80,100]))
        self.assertEqual(r.iloc[0],1);self.assertTrue(np.isnan(r.iloc[1]));self.assertAlmostEqual(r.iloc[2],10)

    def test_ablação_remove_sofa_derivado_dos_fluidos(self):
        self.assertIn('sofa_24h',IO)
        self.assertEqual(len(TODAS),len(set(TODAS)))
        self.assertTrue(set(NOVAS)<=set(TODAS))
        self.assertNotIn('arritmia_yang_retro',TODAS)
        self.assertNotIn('diuretico_alca_48h',TODAS)
        self.assertIn('arritmia_yang_retro',YANG)

    def test_cobertura_e_imputacao_aprendidas_no_treino(self):
        a=pd.DataFrame({'idade':[20,40,60,80], 'x':[1,np.nan,np.nan,np.nan], 'genero':['F','M','F','M']})
        b=pd.DataFrame({'idade':[np.nan], 'x':[999], 'genero':['nova']})
        p=preparo(['idade','x','genero'],max_missing=.3)
        p.fit(a)
        names=p.get_feature_names_out()
        self.assertNotIn('num__x',names)
        self.assertEqual(p.transform(b).shape[1],len(names))

if __name__=='__main__':unittest.main()
