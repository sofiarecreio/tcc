"""Painéis 24h harmonizados e aproximação retrospectiva de Yang, sem misturar alvos."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from extrair_fontes_ampliadas import CACHE, partes, carregar
from caminhos_dados import localizar_tabela
from extrair_painel_24h import CHART, LAB, limpar_eventos
from painel_completo_24h import (LABS, ADMISSION, resumir_chart, recortar_input,
    resumir_input, resumir_output, VASO, FURO, ALCA, INOTROPICOS)
from criterios_clinicos import janela, aki_creatinina, sofa, pafi, ausculta, taxa_mcg_kg_min, arritmias

SAIDA=Path('data/processados')
VERSAO='ampliado-24h-v1'
BASE_LAB={'creatinine':'creatinina','bun':'bun','sodium':'sodio','potassium':'potassio',
          'glucose':'glicose','ptt':'ptt'}
EXTRA_LAB={'lactate':'lactato_max_24h','bicarbonate':'bicarbonato_min_24h',
    'wbc x 1000':'leucocitos_max_24h','hgb':'hemoglobina_min_24h',
    'platelets x 1000':'plaquetas_min_24h','pt - inr':'inr_max_24h',
    'total bilirubin':'bilirrubina_max_24h','troponin - t':'troponina_t_max_24h',
    'pH'.lower():'ph_min_24h','paco2':'pco2_max_24h',
    'nt-probnp':'nt_probnp_max_24h','nt-pro-bnp':'nt_probnp_max_24h'}
LAB_RANGES={v[0]:(v[2],v[3],v[4]) for v in LABS.values()}
DRUGS={'noradrenalina':r'norepinephrine|levophed', 'adrenalina':r'(?<!nor)epinephrine|(?<!nor)adrenalin',
       'dopamina':r'dopamine|intropin', 'dobutamina':r'dobutamine|dobutrex',
       'vasopressina':r'vasopressin|pitressin', 'fenilefrina':r'phenylephrine|neosynephrine',
       'milrinona':r'milrinone|primacor','furosemida':r'furosemide|lasix',
       'bumetanida':r'bumetanide|bumex','torsemida':r'torsemide|demadex'}

def flag(out,name,ids,observados=None):
    out[name]=out.index.isin(ids).astype(float)
    if observados is not None: out.loc[~out.index.isin(observados),name]=np.nan

def laboratorios(base,ids,audit):
    print(base+': agregando laboratórios e AKI...',flush=True)
    xs=[]
    for b in partes(base,'labevents' if base=='mimic' else 'lab'):
        if base=='mimic':
            b['nome']=b.itemid.map({**LAB,**{k:v[0] for k,v in LABS.items()},52546:'creatinina',52024:'creatinina',50821:'pao2',50816:'fio2',52033:'specimen'})
            b=b.rename(columns={'valuenum':'valor','valueuom':'unidade','value':'texto'})
        else:
            b['nome']=b.labname.str.lower().str.strip().map({**BASE_LAB,**EXTRA_LAB,'pao2':'pao2','fio2':'fio2'})
            b['unidade']=b.labmeasurenamesystem.fillna(b.labmeasurenameinterface)
            b=b.rename(columns={'labresult':'valor','labresulttext':'texto'})
        b=b[b.nome.notna()&b.stay_id.isin(ids)].copy()
        b['valor']=pd.to_numeric(b.valor,errors='coerce')
        xs.append(b[['stay_id','offset','disponivel','nome','valor','unidade','texto']])
    raw=pd.concat(xs,ignore_index=True)
    raw=janela(raw,anteriores=True)
    out=pd.DataFrame(index=pd.Index(ids,name='stay_id'))
    basic=raw[raw.nome.isin(set(LAB.values()))].rename(columns={'nome':'variavel'})
    from extrair_painel_24h import FAIXAS, UNIDADES
    for nome in set(LAB.values()):
        q=basic[basic.variavel.eq(nome)].copy()
        u=q.unidade.fillna('').str.lower().str.replace(' ','',regex=False)
        ok=q.valor.between(*FAIXAS[nome])&u.isin(UNIDADES[nome])
        audit['unidades_'+nome]=u[~u.isin(UNIDADES[nome])].value_counts().to_dict()
        q=q[ok]
        current=q[q.offset.ge(0)]
        out[nome+'_24h']=current.groupby('stay_id').valor.mean()
        out[nome+'_max_24h']=current.groupby('stay_id').valor.max()
        if nome=='creatinina':
            cr=q[['stay_id','offset','disponivel','valor']]
    for nome,(lo,hi,units) in LAB_RANGES.items():
        q=raw[raw.nome.eq(nome)&raw.offset.ge(0)].copy()
        u=q.unidade.fillna('').str.lower().str.replace(' ','',regex=False)
        # eICU usa K/mcL para contagens e sec/seconds para tempos.
        accepted=set(units)
        if 'k/ul' in accepted: accepted|={'k/mcl','k/mm3','x1000','10^3/ul','thou/mcl'}
        q=q[q.valor.between(lo,hi)&u.isin(accepted)]
        out[nome]=q.groupby('stay_id').valor.agg('min' if '_min_' in nome else 'max')
    out['ureia_24h']=out.bun_24h*2.14
    out['ureia_max_24h']=out.bun_max_24h*2.14
    out['aki_creatinina_estagio_24h']=aki_creatinina(cr,ids)
    out['aki_creatinina_24h']=out.aki_creatinina_estagio_24h.gt(0).astype(float).where(out.aki_creatinina_estagio_24h.notna())
    for flagname,value in [('lactato_medido','lactato_max_24h'),('nt_probnp_medido','nt_probnp_max_24h'),
        ('troponina_medida','troponina_t_max_24h'),('gasometria_medida','ph_min_24h')]:
        out[flagname]=out[value].notna().astype(int)
    bg=raw[raw.nome.isin(['pao2','fio2','specimen'])&raw.offset.ge(0)].copy()
    if base=='mimic':
        art=bg[bg.nome.eq('specimen')&bg.texto.fillna('').str.upper().str.contains(r'^ART',regex=True)][['stay_id','offset']].drop_duplicates()
        pa=bg[bg.nome.eq('pao2')].merge(art,on=['stay_id','offset'])
    else:
        # O nome paO2 identifica pressão parcial arterial; não usar pO2 venoso.
        pa=bg[bg.nome.eq('pao2')]
    fi=bg[bg.nome.eq('fio2')]
    return out,pa,fi

def diagnosticos_mimic(c,out):
    root=Path('data/mimic-iv')
    diag=pd.read_csv(localizar_tabela(root,'diagnoses_icd'),usecols=['subject_id','hadm_id','icd_code','icd_version'])
    code=diag.icd_code.fillna('').str.upper().str.replace('.','',regex=False)
    valve=(diag.icd_version.eq(9)&code.str.match(r'^(394|395|396|397|424[0-3])'))|(diag.icd_version.eq(10)&code.str.match(r'^(I0[5-8]|I3[4-9])'))
    arr=(diag.icd_version.eq(9)&code.str.match(r'^(4271|42731|4274|4260)'))|(diag.icd_version.eq(10)&code.str.match(r'^(I48|I472|I490|I442)'))
    flag(out,'valvulopatia_retro',c.loc[c.hadm_id.isin(diag.loc[valve,'hadm_id']),'stay_id'])
    flag(out,'arritmia_yang_retro',c.loc[c.hadm_id.isin(diag.loc[arr,'hadm_id']),'stay_id'])
    adm=pd.read_csv(localizar_tabela(root,'admissions'),usecols=['hadm_id','dischtime','admission_type'])
    adm['dischtime']=pd.to_datetime(adm.dischtime,format='mixed',errors='coerce')
    prior=diag[valve].merge(adm[['hadm_id','dischtime']],on='hadm_id')
    pairs=c[['subject_id','stay_id','intime']].merge(prior[['subject_id','dischtime']],on='subject_id')
    flag(out,'valvulopatia_previa',pairs.loc[pairs.dischtime.lt(pairs.intime),'stay_id'])
    out['tipo_admissao']=c.set_index('stay_id').hadm_id.map(adm.set_index('hadm_id').admission_type).map(ADMISSION)

def mimic(c,audit):
    print('mimic: agregando sinais, ritmos e ausculta...',flush=True)
    ids=c.stay_id
    xs=[]
    for b in partes('mimic','chartevents'):
        b=janela(b[b.stay_id.isin(ids)])
        xs.append(b[['stay_id','offset','disponivel','charttime','itemid','value','valuenum','valueuom']])
    chart=pd.concat(xs,ignore_index=True)
    chart['valuenum']=pd.to_numeric(chart.valuenum,errors='coerce')
    out=resumir_chart(chart,ids)
    base=chart[chart.itemid.isin(CHART)].copy()
    base['variavel']=base.itemid.map(CHART)
    base=limpar_eventos(base.rename(columns={'valuenum':'valor','valueuom':'unidade'}),audit)
    out=out.join(base.groupby(['stay_id','variavel']).valor.mean().unstack().add_suffix('_24h'))
    out['pam_min_24h']=base[base.variavel.eq('pam')].groupby('stay_id').valor.min()
    lungs=chart[chart.itemid.isin([223986,223987,223988,223989])][['stay_id','value']].rename(columns={'value':'texto'})
    out=out.join(ausculta(lungs,ids))
    rh=chart[chart.itemid.eq(220048)]
    audit['rotulos_ritmo']=rh.value.value_counts().to_dict()
    text=rh.value.fillna('').str.lower()
    broad,ventricular=arritmias(text)
    flag(out,'arritmia_yang_24h',rh.loc[broad,'stay_id'],rh.stay_id)
    flag(out,'arritmia_ventricular_24h',rh.loc[ventricular,'stay_id'],rh.stay_id)
    fi=chart[chart.itemid.eq(223835)][['stay_id','offset','valuenum']].rename(columns={'valuenum':'valor'})
    # Suporte invasivo: modo ventilatório, não somente PEEP de CPAP não invasivo.
    vent=chart[chart.itemid.isin([223849,224688,226732])].copy()
    vent=vent[pd.to_numeric(vent.value,errors='coerce').isna()]
    audit['rotulos_suporte']=vent.value.value_counts().head(40).to_dict()
    vent['valor']=vent.value.fillna('').str.contains(r'CMV|SIMV|APRV|PRVC|Assist|Volume|Pressure Control|Endotracheal|Tracheostomy',case=False,regex=True).astype(int)
    sup=vent.groupby(['stay_id','offset'],as_index=False).valor.max()
    del chart
    lab,pa,fl=laboratorios('mimic',ids,audit)
    out=out.join(lab).join(pafi(pa,pd.concat([fi,fl[['stay_id','offset','valor']]],ignore_index=True),sup,ids))
    raw=carregar('mimic','inputevents')
    print('mimic: infusões e saídas...',flush=True)
    b=recortar_input(raw,c)
    inp=resumir_input(b,ids)
    for item,name in [(221289,'adrenalina'),(221662,'dopamina'),(221653,'dobutamina')]:
        q=b[b.itemid.eq(item)].copy()
        q['taxa']=taxa_mcg_kg_min(q.rate,q.rateuom,q.patientweight)
        inp[name+'_taxa_max_24h']=q.groupby('stay_id').taxa.max()
        flag(inp,name+'_24h',q.loc[q.rate.gt(0)|q.amount.gt(0),'stay_id'])
        inp.loc[inp[name+'_24h'].eq(0),name+'_taxa_max_24h']=0
    out=out.join(inp)
    iv=b.amountuom.eq('mL')&~b.ordercategoryname.fillna('').str.contains('Oral|Pre Admission|Irrig|Dialys',case=False,regex=True)&~b.itemid.eq(226401)
    out.loc[~out.index.isin(b.loc[iv,'stay_id']),'volume_infundido_24h']=np.nan
    peso=out.peso_kg.fillna(out.peso_input_kg)
    rawout=janela(carregar('mimic','outputevents'))
    out=out.join(resumir_output(rawout,ids,peso))
    out['peso_kg']=peso
    out['balanco_hidrico_24h']=out.volume_infundido_24h-out.saida_total_24h
    out['balanco_ml_kg_24h']=out.balanco_hidrico_24h/peso
    p=janela(carregar('mimic','procedureevents'))
    p=p[~p.statusdescription.fillna('').str.lower().eq('rewritten')]
    flag(out,'cateterismo_cardiaco_24h',p.stay_id)
    p48=janela(carregar('mimic','procedureevents'),horas=48)
    p48=p48[~p48.statusdescription.fillna('').str.lower().eq('rewritten')]
    flag(out,'cateterismo_cardiaco_48h',p48.stay_id)
    raw=raw[raw.offset.lt(2880)&raw.disponivel.lt(2880)&raw.fim.gt(0)&~raw.statusdescription.fillna('').str.lower().eq('rewritten')]
    flag(out,'diuretico_alca_48h',raw.loc[raw.itemid.isin(ALCA)&(raw.amount.gt(0)|raw.rate.gt(0)),'stay_id'])
    diagnosticos_mimic(c,out)
    antigo=pd.read_csv(SAIDA/'mimic_completo_24h.csv').set_index('stay_id')
    out['feve']=antigo.feve
    return out

def drogas_eicu(ids,peso,audit):
    raw=carregar('eicu','infusionDrug'); b=janela(raw)
    out=pd.DataFrame(index=pd.Index(ids,name='stay_id'))
    names=b.drugname.fillna('').str.lower()
    rate=pd.to_numeric(b.drugrate,errors='coerce')
    ir=pd.to_numeric(b.infusionrate,errors='coerce')
    active=rate.gt(0)|ir.gt(0)
    # Sem registro de infusão, ausência de evidência é 0; cobertura é relatada.
    for name,regex in DRUGS.items():
        mask=names.str.contains(regex,regex=True)&active
        flag(out,name+'_24h',b.loc[mask,'stay_id'])
        if name in ['noradrenalina','adrenalina','dopamina','dobutamina']:
            q=b[mask].copy()
            unit=q.drugname.str.extract(r'\(([^()]*)\)')[0]
            weight=pd.to_numeric(q.patientweight,errors='coerce').where(lambda x:x.between(25,350)).fillna(q.stay_id.map(peso))
            q['taxa']=taxa_mcg_kg_min(q.drugrate,unit,weight)
            hi=5 if name in ['noradrenalina','adrenalina'] else 50
            q['taxa']=q.taxa.where(q.taxa.le(hi))
            audit['taxas_nao_convertidas_'+name]=int(q.taxa.isna().sum())
            out[name+'_taxa_max_24h']=q.groupby('stay_id').taxa.max()
            out.loc[out[name+'_24h'].eq(0),name+'_taxa_max_24h']=0
    vaso=[n+'_24h' for n in VASO.values()]
    out['n_vasopressores_24h']=out[vaso].sum(axis=1)
    out['vasopressor_24h']=out.n_vasopressores_24h.gt(0).astype(int)
    out['inotropico_24h']=out[['dobutamina_24h','milrinona_24h']].max(axis=1)
    out['diuretico_alca_24h']=out[['furosemida_24h','bumetanida_24h','torsemida_24h']].max(axis=1)
    out['furosemida_infusao_continua_24h']=out.furosemida_24h
    # Não integrar snapshots sem término: dose total não é recuperável com segurança.
    out['furosemida_dose_total_24h']=np.nan
    flag(out,'diuretico_alca_48h',raw.loc[raw.drugname.fillna('').str.contains('furosemide|lasix|bumetanide|bumex|torsemide',case=False,regex=True)&
        (pd.to_numeric(raw.drugrate,errors='coerce').gt(0)|pd.to_numeric(raw.infusionrate,errors='coerce').gt(0)),'stay_id'])
    return out

def fluidos_eicu(ids,peso,audit,horas=24):
    b=janela(carregar('eicu','intakeOutput'),horas)
    # Somente eventos individuais; nunca somar intaketotal/outputtotal/nettotal cumulativos.
    b=b.drop_duplicates('intakeoutputid').sort_values('disponivel').drop_duplicates(['stay_id','offset','cellpath','celllabel'],keep='last').copy()
    path=b.cellpath.fillna('').str.lower(); label=b.celllabel.fillna('').str.lower()
    v=pd.to_numeric(b.cellvaluenumeric,errors='coerce')
    unit=path.str.contains(r'\(ml\)',regex=True)|label.str.contains(r'\(ml\)',regex=True)
    summary=label.str.contains(r'total|cumulative|balance|net|shift|24 hour|8 hr|count|occurrence|unmeasured',regex=True)
    exclude=label.str.contains(r'oral|po intake|tube feed|gastric.*intake|flush|irrig|dialys|ultrafil',regex=True)
    b['valor']=v.where(v.between(0,30000))
    inp=path.str.contains(r'\|intake \(ml\)\|',regex=True)&unit&~summary&~exclude
    output=path.str.contains(r'\|output \(ml\)\|',regex=True)&unit&~summary
    urine=output&label.str.contains(r'urine|urinary|foley|void|nephrost',regex=True)&~label.str.contains('irrig',regex=False)
    out=pd.DataFrame(index=pd.Index(ids,name='stay_id'))
    for name,mask in [('volume_infundido_24h',inp),('saida_total_24h',output),('diurese_total_24h',urine)]:
        out[name]=b[mask].groupby('stay_id').valor.sum(min_count=1)
    for name,regex in DRUGS.items():
        flag(out,name+'_io',b.loc[inp&label.str.contains(regex,regex=True)&b.valor.gt(0),'stay_id'])
    out['diurese_medida']=out.diurese_total_24h.notna().astype(int)
    out['diurese_ml_kg_h_24h']=out.diurese_total_24h/(peso*horas)
    out['oliguria_24h']=out.diurese_ml_kg_h_24h.lt(.5).astype(float).where(out.diurese_ml_kg_h_24h.notna())
    out['balanco_hidrico_24h']=out.volume_infundido_24h-out.saida_total_24h
    out['balanco_ml_kg_24h']=out.balanco_hidrico_24h/peso
    bolus=inp&label.str.contains('bolus')&label.str.contains('saline|sodium chloride|ringer|plasma.?lyte',regex=True)&b.valor.gt(0)
    flag(out,'bolus_cristaloide_24h',b.loc[bolus,'stay_id'],b.stay_id)
    audit['io_linhas_resumo_excluidas']=int(summary.sum())
    audit['io_registros_sem_unidade_ml']=int((~unit).sum())
    audit['io_oral_irrigacao_outros_excluidos']=int(exclude.sum())
    audit['io_rotulos_urina']=b.loc[urine,'celllabel'].value_counts().head(25).to_dict()
    return out

def eicu(c,audit):
    print('eicu: sinais vitais...',flush=True)
    ids=c.stay_id; out=pd.DataFrame(index=pd.Index(ids,name='stay_id'))
    patient=pd.read_csv(localizar_tabela('data/eicu','patient'),usecols=['patientunitstayid','admissionweight']).set_index('patientunitstayid')
    out['peso_kg']=pd.to_numeric(patient.admissionweight,errors='coerce').where(lambda x:x.between(25,350))
    weight=janela(carregar('eicu','infusionDrug'))
    weight['peso']=pd.to_numeric(weight.patientweight,errors='coerce')
    out['peso_kg']=out.peso_kg.fillna(weight[weight.peso.between(25,350)].groupby('stay_id').peso.median())
    sums=[]; minima=[]
    for table,mapping in [('vitalPeriodic',{'heartrate':'fc','respiration':'fr','sao2':'spo2','systemicmean':'pam','systemicsystolic':'pas'}),
                          ('vitalAperiodic',{'noninvasivemean':'pam','noninvasivesystolic':'pas'})]:
        for b in partes('eicu',table):
            b=janela(b).rename(columns=mapping)
            b=b.melt(id_vars='stay_id',value_vars=list(mapping.values()),var_name='variavel',value_name='valor')
            b=limpar_eventos(b,audit,unidades=False)
            sums.append(b.groupby(['stay_id','variavel']).valor.agg(['sum','count']))
            minima.append(b[b.variavel.eq('pam')].groupby('stay_id').valor.min())
    ag=pd.concat(sums).groupby(level=[0,1]).sum()
    print('eicu: enfermagem e respiração...',flush=True)
    out=out.join((ag['sum']/ag['count']).unstack().add_suffix('_24h'))
    out['pam_min_24h']=pd.concat(minima).groupby(level=0).min()
    temps=[]; glasgow=[]; rhythms=[]
    for b in partes('eicu','nurseCharting'):
        b=janela(b)
        name=b.nursingchartcelltypevalname.fillna('')
        val=pd.to_numeric(b.nursingchartvalue,errors='coerce')
        mask=name.isin(['Temperature (C)','Temperature (F)'])
        t=b.loc[mask,['stay_id']].copy()
        t['valor']=val[mask].where(~name[mask].eq('Temperature (F)'),(val[mask]-32)/1.8)
        temps.append(t[t.valor.between(30,43)])
        q=b.loc[name.eq('GCS Total'),['stay_id']].copy();q['valor']=val[name.eq('GCS Total')]
        glasgow.append(q[q.valor.between(3,15)])
        mask=(name+' '+b.nursingchartcelltypevallabel.fillna('')).str.contains('rhythm',case=False)
        rhythms.append(b.loc[mask,['stay_id','nursingchartvalue']])
    out['temperatura_mediana_24h']=pd.concat(temps).groupby('stay_id').valor.median()
    out['gcs_min_24h']=pd.concat(glasgow).groupby('stay_id').valor.min()
    resp=janela(carregar('eicu','respiratoryCharting'))
    resp['valor']=pd.to_numeric(resp.respchartvalue,errors='coerce')
    rn=resp.respchartvaluelabel.fillna('').str.lower()
    fi=resp[rn.eq('fio2')][['stay_id','offset','valor']].copy()
    fi['valor']=fi.valor.where(fi.valor.gt(1),fi.valor*100)
    fi=fi[fi.valor.between(21,100)]
    out['fio2_max_24h']=fi.groupby('stay_id').valor.max()
    peep=resp[rn.eq('peep')|rn.eq('peep/cpap')]
    out['peep_max_24h']=peep[peep.valor.between(0,30)].groupby('stay_id').valor.max()
    flag(out,'registro_ventilador_24h',resp.loc[(rn.str.contains(r'peep|tidal volume|observed \(vt\)',regex=True))&resp.valor.notna(),'stay_id'])
    # Evidência de volume corrente/set ventilador no instante; CPAP isolado não basta.
    sup=resp[rn.str.contains(r'tidal volume|observed \(vt\)',regex=True)&resp.valor.gt(0)][['stay_id','offset']].assign(valor=1)
    exam=janela(carregar('eicu','physicalExam'))
    print('eicu: exame físico, diagnósticos e procedimentos...',flush=True)
    ex=exam.physicalexampath.fillna('').str.lower()
    audit['caminhos_ausculta']=exam.loc[ex.str.contains('respiratory|pulmonary|lung',regex=True),'physicalexampath'].value_counts().head(60).to_dict()
    lungs=exam[ex.str.contains(r'auscult|breath sounds|rhonch|rales|crackl|wheez|stridor',regex=True)].copy()
    lungs['texto']=lungs.physicalexampath.fillna('').astype(str)+' '+lungs.physicalexamvalue.fillna('').astype(str)+' '+lungs.physicalexamtext.fillna('').astype(str)
    out=out.join(ausculta(lungs[['stay_id','texto']],ids))
    diag=janela(carregar('eicu','diagnosis'))
    text=diag.diagnosisstring.fillna('').str.lower()
    arr,ventricular=arritmias(text)
    flag(out,'arritmia_yang_24h',diag.loc[arr,'stay_id'],diag.stay_id)
    flag(out,'arritmia_ventricular_24h',diag.loc[ventricular,'stay_id'],diag.stay_id)
    hist=janela(carregar('eicu','pastHistory'))
    ht=hist.pasthistorypath.fillna('').astype(str)+' '+hist.pasthistoryvalue.fillna('').astype(str)+' '+hist.pasthistoryvaluetext.fillna('').astype(str)
    flag(out,'valvulopatia_previa',hist.loc[ht.str.contains(r'valvular|valve.*(?:stenosis|regurgitation|disease)|(?:aortic|mitral).*(?:stenosis|regurgitation)',case=False,regex=True),'stay_id'])
    tr=carregar('eicu','treatment'); tt=tr.treatmentstring.fillna('').str.lower()
    cath=tt.str.contains(r'cardiac catheter|cardiac angiograph|coronary angiograph|left heart cath|right heart cath',regex=True)&~tt.str.contains('planned|consider|possible',regex=True)
    flag(out,'cateterismo_cardiaco_48h',tr.loc[cath,'stay_id'])
    flag(out,'cateterismo_cardiaco_24h',tr.loc[cath&tr.offset.lt(1440)&tr.disponivel.lt(1440),'stay_id'])
    # Diagnósticos sem corte somente no benchmark retrospectivo claramente separado.
    all_diag=pd.read_csv(localizar_tabela('data/eicu','diagnosis'),usecols=['patientunitstayid','diagnosisstring'])
    text=all_diag.diagnosisstring.fillna('').str.lower()
    flag(out,'arritmia_yang_retro',all_diag.loc[arritmias(text)[0],'patientunitstayid'])
    flag(out,'valvulopatia_retro',all_diag.loc[text.str.contains(r'valvular|valve.*(?:stenosis|regurgitation|disease)|(?:aortic|mitral).*(?:stenosis|regurgitation)',regex=True),'patientunitstayid'])
    lab,pa,fl=laboratorios('eicu',ids,audit)
    out=out.join(lab).join(pafi(pa,pd.concat([fi,fl[['stay_id','offset','valor']]],ignore_index=True),sup,ids))
    out=out.join(drogas_eicu(ids,out.peso_kg,audit)).join(fluidos_eicu(ids,out.peso_kg,audit))
    print('eicu: consolidando tratamentos e balanço...',flush=True)
    for name in DRUGS:
        out[name+'_24h']=out[[name+'_24h',name+'_io']].max(axis=1)
        rate=name+'_taxa_max_24h'
        if rate in out: out.loc[out[name+'_24h'].eq(1)&out[rate].eq(0),rate]=np.nan
    out['n_vasopressores_24h']=out[[n+'_24h' for n in VASO.values()]].sum(axis=1)
    out['vasopressor_24h']=out.n_vasopressores_24h.gt(0).astype(int)
    out['inotropico_24h']=out[['dobutamina_24h','milrinona_24h']].max(axis=1)
    out['diuretico_alca_24h']=out[['furosemida_24h','bumetanida_24h','torsemida_24h']].max(axis=1)
    io48=fluidos_eicu(ids,out.peso_kg,{},horas=48)
    out['diuretico_alca_48h']=pd.concat([out.diuretico_alca_48h,
        io48[['furosemida_io','bumetanida_io','torsemida_io']].max(axis=1)],axis=1).max(axis=1)
    if (out.diuretico_alca_24h > out.diuretico_alca_48h).any():
        raise ValueError('Diuretico presente em 24h e ausente em 48h')
    out['feve']=np.nan;out['tipo_admissao']=np.nan
    return out

def extrair(base):
    c=pd.read_pickle(CACHE/base/'coorte.pkl')
    audit={}
    out=(mimic if base=='mimic' else eicu)(c,audit)
    out=out.join(sofa(out)).copy()
    meta=['stay_id','patient_id','hospital_id','los_dias','obito_hospitalar','raca','idade','genero','primeira_uti_internacao','ic_principal']
    d=c[meta].merge(out.reset_index(),on='stay_id',validate='one_to_one')
    d['versao_painel_ampliado']=VERSAO
    if d.stay_id.duplicated().any(): raise ValueError('Duplicação de estadias')
    if np.isinf(d.select_dtypes('number')).any().any(): raise ValueError('Valor infinito')
    d.to_csv(SAIDA/f'{base}_ampliado_todas_estadias.csv',index=False)
    d[d.los_dias.gt(1)].to_csv(SAIDA/f'{base}_ampliado_24h.csv',index=False)
    cov=pd.DataFrame({'variavel':out.columns,'fracao_ausente':out.isna().mean().values,'n_distintos':out.nunique().values})
    cov.to_csv(SAIDA/f'cobertura_{base}_ampliado.csv',index=False)
    (SAIDA/f'auditoria_{base}_ampliado.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,default=int),encoding='utf-8')
    print(f'{base}: {len(d):,} estadias; {d.los_dias.gt(1).sum():,} elegíveis em 24h; {len(out.columns)} medidas',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',choices=['mimic','eicu']);extrair(p.parse_args().base)
