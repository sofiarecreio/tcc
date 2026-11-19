import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from experimento_populacao import filtrar, FILTROS, ic_ate_posicao


class FiltrosTest(unittest.TestCase):
    def setUp(self):
        self.d = pd.DataFrame({'stay_id': [1, 2, 3, 4, 5],
                               'ic_principal': [True, True, False, True, True],
                               'primeira_uti_internacao': [True, False, True, True, True],
                               'obito_hospitalar': [0, 0, 0, 1, np.nan]})

    def test_cada_filtro_isolado(self):
        self.assertEqual(filtrar(self.d, ['ic_principal']).stay_id.tolist(), [1, 2, 4, 5])
        self.assertEqual(filtrar(self.d, ['primeira_uti']).stay_id.tolist(), [1, 3, 4, 5])
        self.assertEqual(filtrar(self.d, ['sobreviventes']).stay_id.tolist(), [1, 2, 3])

    def test_tres_filtros_e_obito_desconhecido_excluido(self):
        self.assertEqual(filtrar(self.d, FILTROS['tres_filtros']).stay_id.tolist(), [1])
        self.assertEqual(len(filtrar(self.d, FILTROS['base'])), 5)


class IcPosicaoTest(unittest.TestCase):
    def test_ic_na_posicao_2_entra_e_posicao_3_nao(self):
        with tempfile.TemporaryDirectory() as raiz:
            pd.DataFrame({'stay_id': [1, 2, 3, 4], 'hadm_id': [10, 20, 30, 40]}).to_csv(Path(raiz) / 'icustays.csv', index=False)
            pd.DataFrame({'hadm_id': [10, 20, 20, 30, 30, 40], 'seq_num': [1, 1, 2, 1, 3, 1],
                          'icd_code': ['I509', 'I130', 'I5023', 'I214', '4280', 'J189'],
                          'icd_version': [10, 10, 10, 10, 9, 10]}).to_csv(Path(raiz) / 'diagnoses_icd.csv', index=False)
            m = pd.DataFrame({'stay_id': [1, 2, 3, 4], 'ic_principal': [True, False, False, False]})
            self.assertEqual(ic_ate_posicao(m, 2, raiz).ic_principal.tolist(), [True, True, False, False])


if __name__ == '__main__':
    unittest.main()
