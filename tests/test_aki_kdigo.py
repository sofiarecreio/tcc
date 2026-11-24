import unittest
import numpy as np
import pandas as pd
from criterios_clinicos import aki_diurese, aki_kdigo, aki_creatinina


def horaria(stay, ml, horas, inicio=0):
    t = np.arange(inicio, inicio + horas + 1) * 60.
    return pd.DataFrame({'stay_id': stay, 'offset': t, 'disponivel': t + 5, 'valor': ml})


class AkiDiureseTest(unittest.TestCase):
    def test_oliguria_12h_e_estagio_2(self):
        r = aki_diurese(horaria(1, 40., 12), pd.Series({1: 100.}), [1])
        self.assertEqual(r.loc[1], 2)

    def test_diurese_adequada_e_zero(self):
        r = aki_diurese(horaria(1, 60., 20), pd.Series({1: 100.}), [1])
        self.assertEqual(r.loc[1], 0)

    def test_oliguria_6h_apenas_e_estagio_1(self):
        u = pd.concat([horaria(1, 40., 6), horaria(1, 80., 12, inicio=7)])
        self.assertEqual(aki_diurese(u, pd.Series({1: 100.}), [1]).loc[1], 1)

    def test_anuria_12h_e_estagio_3(self):
        self.assertEqual(aki_diurese(horaria(1, 0., 12), pd.Series({1: 70.}), [1]).loc[1], 3)

    def test_sem_cobertura_ou_peso_e_desconhecido(self):
        u = pd.concat([horaria(1, 0., 5), horaria(2, 0., 12)])
        r = aki_diurese(u, pd.Series({1: 70., 2: np.nan}), [1, 2])
        self.assertTrue(r.isna().all())

    def test_registro_liberado_apos_corte_nao_entra(self):
        u = horaria(1, 0., 23)
        u['disponivel'] = np.where(u.offset.ge(10 * 60), 1500., u.disponivel)
        self.assertEqual(aki_diurese(u, pd.Series({1: 70.}), [1]).loc[1], 1)

    def test_janela_48h_ve_oliguria_tardia(self):
        u = pd.concat([horaria(1, 100., 23), horaria(1, 0., 20, inicio=24)])
        peso = pd.Series({1: 70.})
        self.assertEqual(aki_diurese(u, peso, [1], horas=24).loc[1], 0)
        self.assertEqual(aki_diurese(u, peso, [1], horas=48).loc[1], 3)


class AkiKdigoTest(unittest.TestCase):
    def test_maior_estagio_entre_criterios(self):
        cr = pd.Series([1., np.nan, np.nan, 0.], index=[1, 2, 3, 4])
        uo = pd.Series([2., 1., np.nan, np.nan], index=[1, 2, 3, 4])
        r = aki_kdigo(cr, uo)
        self.assertEqual(r.loc[[1, 2, 4]].tolist(), [2., 1., 0.])
        self.assertTrue(np.isnan(r.loc[3]))

    def test_creatinina_48h_preserva_padrao_24h(self):
        d = pd.DataFrame({'stay_id': [1, 1, 1], 'offset': [-60, 120, 2000],
                          'disponivel': [-30, 130, 2010], 'valor': [1., 1.1, 3.1]})
        self.assertEqual(aki_creatinina(d, [1]).loc[1], 0)
        self.assertEqual(aki_creatinina(d, [1], horas=48).loc[1], 3)


if __name__ == '__main__':
    unittest.main()
