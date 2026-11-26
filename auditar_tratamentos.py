"""Cobertura eICU: distingue ordens de medicamentos de evidencia de administracao."""
from pathlib import Path
import json
import pandas as pd
from extrair_fontes_ampliadas import carregar
from criterios_clinicos import janela
from painel_ampliado import fluidos_eicu


def rodar():
    p=Path('data/processados')
    d=pd.read_csv(p/'eicu_ampliado_todas_estadias.csv',low_memory=False).set_index('stay_id')
    io=fluidos_eicu(d.index,d.peso_kg,{},48)
    corrected=pd.concat([d.diuretico_alca_48h,
        io[['furosemida_io','bumetanida_io','torsemida_io']].max(axis=1)],axis=1).max(axis=1)
    orders=janela(carregar('eicu','medication'),48)
    loop=orders.drugname.fillna('').str.contains('furosemide|lasix|bumetanide|bumex|torsemide',case=False,regex=True)
    if 'drugordercancelled' in orders:
        loop &= ~orders.drugordercancelled.fillna('').str.lower().eq('yes')
    ordered=d.index.isin(orders.loc[loop,'stay_id'])
    result={'n':len(d),'diuretico_24h':int(d.diuretico_alca_24h.sum()),
        'diuretico_48h_anterior':int(d.diuretico_alca_48h.sum()),
        'diuretico_48h_corrigido':int(corrected.sum()),
        'positivos_recuperados':int(corrected.gt(d.diuretico_alca_48h).sum()),
        'contradicoes_antes':int(d.diuretico_alca_24h.gt(d.diuretico_alca_48h).sum()),
        'contradicoes_depois':int(d.diuretico_alca_24h.gt(corrected).sum()),
        'ordem_diuretico_48h':int(ordered.sum()),
        'ordem_sem_evidencia_administracao':int((ordered & corrected.eq(0)).sum()),
        'limite':'Ordens nao comprovam administracao; ausencia de evidencia nao comprova nao uso.'}
    assert result['contradicoes_depois']==0
    out=Path('artefatos_protocolo_24h/auditoria_tratamentos'); out.mkdir(exist_ok=True)
    (out/'eicu.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': rodar()
