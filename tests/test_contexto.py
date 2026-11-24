import unittest
import pandas as pd
from contexto_24h import origem_mimic, MAPA_UNIDADE, MAPA_ORIGEM_EICU, indicadores, UNIDADES_UTI


class ContextoTest(unittest.TestCase):
    def test_origem_mimic(self):
        r = origem_mimic(pd.Series(['Emergency Department', 'PACU', 'Medicine/Cardiology',
                                    'Medical Intensive Care Unit (MICU)', None]))
        self.assertEqual(list(r), ['pronto_socorro', 'centro_cirurgico', 'enfermaria', 'outra_uti', 'direta_externa'])

    def test_unidades_harmonizadas_entre_bases(self):
        self.assertEqual(MAPA_UNIDADE['Coronary Care Unit (CCU)'], MAPA_UNIDADE['Cardiac ICU'])
        self.assertEqual(MAPA_UNIDADE['Cardiac Vascular Intensive Care Unit (CVICU)'], MAPA_UNIDADE['CTICU'])
        self.assertEqual(MAPA_UNIDADE['Medical/Surgical Intensive Care Unit (MICU/SICU)'], MAPA_UNIDADE['Med-Surg ICU'])
        self.assertEqual(MAPA_ORIGEM_EICU['Operating Room'], 'centro_cirurgico')

    def test_indicadores_sem_categoria_ficam_zero(self):
        out = pd.DataFrame(index=[1, 2])
        indicadores(out, pd.Series(['clinica', None], index=[1, 2]), 'unidade', UNIDADES_UTI)
        self.assertEqual(out.loc[1].sum(), 1)
        self.assertEqual(out.loc[2].sum(), 0)


if __name__ == '__main__':
    unittest.main()
