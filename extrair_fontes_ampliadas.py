"""Cache local de eventos candidatos; nunca publica registros clínicos.

O cache inclui até 48h apenas para o experimento retrospectivo identificado.
Os painéis prospectivos aplicam novamente o corte de disponibilidade de 24h.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from caminhos_dados import localizar_tabela
from extrair_painel_24h import coorte_mimic, coorte_eicu, CHART, LAB
from painel_completo_24h import EXTRA_CHART, LABS

CACHE=Path('data/processados/fontes_ampliadas')
VERSION=1
CHART_IDS=set(CHART)|EXTRA_CHART|{220048,223986,223987,223988,223989,223848,223849,224688,224690,226732}
LAB_IDS=set(LAB)|set(LABS)|{52033,50816,50821,52546,52024}

def extrair(base):
    raiz=Path('data')/('mimic-iv' if base=='mimic' else 'eicu')
    c=(coorte_mimic(raiz) if base=='mimic' else coorte_eicu(raiz))
    c=c[c.idade.ge(18)&c.los_dias.gt(0)&np.isfinite(c.los_dias)].copy()
    dest=CACHE/base
    dest.mkdir(parents=True,exist_ok=True)
    c.to_pickle(dest/'coorte.pkl')
    if base=='mimic':
        specs={
          'chartevents':('stay_id','charttime',['stay_id','itemid','charttime','storetime','value','valuenum','valueuom']),
          'labevents':('subject_id','charttime',['subject_id','hadm_id','itemid','charttime','storetime','value','valuenum','valueuom']),
          'inputevents':('stay_id','starttime',['stay_id','starttime','endtime','storetime','itemid','amount','amountuom','rate','rateuom','ordercategoryname','ordercategorydescription','patientweight','statusdescription']),
          'outputevents':('stay_id','charttime',['stay_id','charttime','storetime','itemid','value','valueuom']),
          'procedureevents':('stay_id','starttime',['stay_id','starttime','endtime','storetime','itemid','statusdescription'])}
    else:
        specs={n:('patientunitstayid',t,None) for n,t in [
            ('lab','labresultoffset'),('nurseCharting','nursingchartoffset'),
            ('respiratoryCharting','respchartoffset'),('infusionDrug','infusionoffset'),
            ('intakeOutput','intakeoutputoffset'),('medication','drugstartoffset'),
            ('diagnosis','diagnosisoffset'),('pastHistory','pasthistoryoffset'),
            ('physicalExam','physicalexamoffset'),('treatment','treatmentoffset'),
            ('vitalPeriodic','observationoffset'),('vitalAperiodic','observationoffset')]}
    for name,(key,time,cols) in specs.items():
        source=localizar_tabela(raiz,name)
        folder=dest/name
        folder.mkdir(exist_ok=True)
        stamp={'version':VERSION,'path':str(source),'size':source.stat().st_size,'mtime':source.stat().st_mtime_ns}
        done=folder/'concluido.json'
        if done.exists() and json.loads(done.read_text())['source']==stamp:
            print(f'{base}/{name}: cache conferido',flush=True); continue
        count=0
        nfiles=0
        for i,b in enumerate(pd.read_csv(source,usecols=cols,chunksize=500_000,low_memory=False)):
            b=b[b[key].isin(c['stay_id' if base=='eicu' else key])].copy()
            if base=='mimic':
                if name=='chartevents': b=b[b.itemid.isin(CHART_IDS)]
                if name=='labevents': b=b[b.itemid.isin(LAB_IDS)]
                if name=='procedureevents': b=b[b.itemid.eq(225430)]
                joincols=[key,'intime','outtime']+(['stay_id'] if key!='stay_id' else [])
                b=b.merge(c[joincols],on=key)
                for t in [time,'storetime']+(['endtime'] if 'endtime' in b else []):
                    b[t]=pd.to_datetime(b[t],errors='coerce',format='mixed')
                b['offset']=(b[time]-b.intime).dt.total_seconds()/60
                b['disponivel']=(b.storetime-b.intime).dt.total_seconds()/60
                if 'endtime' in b: b['fim']=(b.endtime-b.intime).dt.total_seconds()/60
                lower=-10080 if name=='labevents' else (-1440 if name=='inputevents' else 0)
                b=b[b.offset.ge(lower)&b.offset.lt(2880)&b.disponivel.lt(2880)&b[time].lt(b.outtime)]
                b=b.drop(columns=['intime','outtime'])
            else:
                b=b.rename(columns={'patientunitstayid':'stay_id'})
                b['offset']=pd.to_numeric(b[time],errors='coerce')
                entry={'lab':'labresultrevisedoffset','nurseCharting':'nursingchartentryoffset',
                       'respiratoryCharting':'respchartentryoffset','intakeOutput':'intakeoutputentryoffset',
                       'pastHistory':'pasthistoryenteredoffset','medication':'drugorderoffset'}.get(name,time)
                b['disponivel']=pd.to_numeric(b[entry],errors='coerce')
                lower=-10080 if name=='lab' else (-1440 if name=='medication' else 0)
                end=c.set_index('stay_id').los_dias*1440
                b=b[b.offset.ge(lower)&b.offset.lt(2880)&b.disponivel.lt(2880)&b.offset.lt(b.stay_id.map(end))]
            if not b.empty:
                b.to_pickle(folder/f'parte_{nfiles:05}.pkl')
                nfiles+=1
                count+=len(b)
            if (i+1)%20==0: print(f'{base}/{name}: {(i+1)*500_000:,} lidas; {count:,} selecionadas',flush=True)
        done.write_text(json.dumps({'source':stamp,'n':count,'partes':nfiles},indent=2))
        print(f'{base}/{name}: CONCLUÍDO ({count:,} registros)',flush=True)

def partes(base,name):
    folder=CACHE/base/name
    meta=json.loads((folder/'concluido.json').read_text())
    for i in range(meta['partes']): yield pd.read_pickle(folder/f'parte_{i:05}.pkl')

def carregar(base,name):
    xs=list(partes(base,name))
    return pd.concat(xs,ignore_index=True) if xs else pd.DataFrame()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',choices=['mimic','eicu'])
    extrair(p.parse_args().base)
