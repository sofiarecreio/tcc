import unittest
from unittest.mock import patch
import pandas as pd
from painel_ampliado import fluidos_eicu


class TratamentosTest(unittest.TestCase):
    def test_janelas_e_disponibilidade(self):
        rows = []
        for i, (t, entry, label) in enumerate([(60, 60, 'Lasix'), (1800, 1801, 'Lasix'),
                                              (100, 3000, 'Lasix'), (100, 100, 'oral Lasix')]):
            rows.append(dict(stay_id=i+1, intakeoutputid=i, offset=t, disponivel=entry,
                             cellpath='x|intake (ml)|drug', celllabel=label, cellvaluenumeric=10))
        with patch('painel_ampliado.carregar', return_value=pd.DataFrame(rows)):
            a=fluidos_eicu([1,2,3,4],pd.Series(70,index=[1,2,3,4]),{},24)
            b=fluidos_eicu([1,2,3,4],pd.Series(70,index=[1,2,3,4]),{},48)
        self.assertEqual(a.furosemida_io.tolist(),[1,0,0,0])
        self.assertEqual(b.furosemida_io.tolist(),[1,1,0,0])

