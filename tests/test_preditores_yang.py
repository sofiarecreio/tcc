import unittest
import pandas as pd
from criterios_clinicos import arritmia_maligna, ausculta_yang


class PreditoresYangTest(unittest.TestCase):
    def test_tv_nao_sustentada_nao_conta(self):
        r = arritmia_maligna(pd.Series(['cardiovascular|arrhythmias|ventricular tachycardia|non-sustained',
                                        'cardiovascular|arrhythmias|ventricular tachycardia',
                                        'AF (Atrial Fibrillation)', 'ST (Sinus Tachycardia)', None]))
        self.assertEqual(r.tolist(), [False, True, True, False, False])

    def test_ausculta_ignora_estertores_e_sem_registro_e_zero(self):
        reg = pd.DataFrame({'stay_id': [1, 2, 3, 4], 'texto': ['Rhonchi', 'Crackles', 'Tubular', 'no stridor']})
        r = ausculta_yang(reg, [1, 2, 3, 4, 5])
        self.assertEqual(r.roncos_yang_24h.tolist(), [1, 0, 0, 0, 0])
        self.assertEqual(r.bronquico_yang_24h.tolist(), [0, 0, 1, 0, 0])
        self.assertEqual(r.estridor_yang_24h.tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(r.ausculta_yang_registrada_24h.tolist(), [1, 1, 1, 1, 0])


if __name__ == '__main__':
    unittest.main()
