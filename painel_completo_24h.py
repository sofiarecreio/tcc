"""Extrai o painel MIMIC solicitado, incluindo Echo, inputevents e outputevents."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from caminhos_dados import localizar_tabela
from extrair_painel_24h import coorte_mimic, limpar_eventos, CHART, LAB
from variaveis_protocolo import colunas

VERSAO = 'completo-24h-v2'
ADMISSION = {
    'EW EMER.': 'Emergência pelo pronto-socorro', 'URGENT': 'Urgente',
    'OBSERVATION ADMIT': 'Internação vinda de observação',
    'DIRECT EMER.': 'Emergência sem passar pelo pronto-socorro',
    'ELECTIVE': 'Eletiva/Programada', 'SURGICAL SAME DAY ADMISSION': 'Eletiva/Programada',
    'EU OBSERVATION': 'Observação de emergência', 'DIRECT OBSERVATION': 'Observação direta',
    'AMBULATORY OBSERVATION': 'Observação ambulatorial',
}
# itemid: (coluna, agregação, mínimo, máximo, unidades)
LABS = {
    50813: ('lactato_max_24h', 'max', .2, 30, {'mmol/l'}),
    50882: ('bicarbonato_min_24h', 'min', 5, 50, {'meq/l', 'mmol/l'}),
    51301: ('leucocitos_max_24h', 'max', .1, 200, {'k/ul'}),
    51222: ('hemoglobina_min_24h', 'min', 3, 22, {'g/dl'}),
    51265: ('plaquetas_min_24h', 'min', 5, 1500, {'k/ul'}),
    51237: ('inr_max_24h', 'max', .5, 15, {'', 'ratio'}),
    50885: ('bilirrubina_max_24h', 'max', .1, 60, {'mg/dl'}),
    50963: ('nt_probnp_max_24h', 'max', 5, 70000, {'pg/ml'}),
    51003: ('troponina_t_max_24h', 'max', .001, 50, {'ng/ml'}),
    50820: ('ph_min_24h', 'min', 6.8, 7.8, {'', 'units'}),
    50818: ('pco2_max_24h', 'max', 10, 200, {'mmhg'}),
}
EXTRA_CHART = {223761, 223762, 220739, 223900, 223901, 223835, 220339,
               224684, 224685, 226512, 226531, 224639}
VASO = {221906: 'noradrenalina', 222315: 'vasopressina', 221289: 'adrenalina',
        221749: 'fenilefrina', 221662: 'dopamina'}
FURO = {221794, 228340}
ALCA = FURO | {229639}
INOTROPICOS = {221653, 221986}
CRISTALOIDES = {220953, 220954, 220955, 220956, 225158}
URINA = {226559, 226560, 226561, 226584, 226563, 226564, 226565,
         226567, 226557, 226558, 227488, 227489}


def blocos(raiz, tabela, cols):
    return pd.read_csv(localizar_tabela(raiz, tabela), usecols=cols,
                       chunksize=1_000_000, low_memory=False)


def filtrar_tempos(b, coorte, chave='stay_id', tempo='charttime'):
    b = b[b[chave].isin(coorte[chave])].copy()
    if b.empty: return b.assign(intime=pd.NaT)
    cols = [chave, 'intime'] + ([] if chave == 'stay_id' else ['stay_id'])
    b = b.merge(coorte[cols], on=chave)
    for c in [tempo, 'storetime']:
        b[c] = pd.to_datetime(b[c], errors='coerce', format='mixed')
    limite = b.intime + pd.Timedelta(hours=24)
    return b[b[tempo].ge(b.intime) & b[tempo].lt(limite) & b.storetime.lt(limite)].copy()


def resumir_chart(b, ids):
    out = pd.DataFrame(index=pd.Index(ids, name='stay_id'))
    temp = b[b.itemid.isin([223761, 223762])].copy()
    temp['celsius'] = np.where(temp.itemid.eq(223761), (temp.valuenum - 32) / 1.8, temp.valuenum)
    out['temperatura_mediana_24h'] = temp[temp.celsius.between(30, 43)].groupby('stay_id').celsius.median()
    g = b[b.itemid.isin([220739, 223900, 223901])].copy()
    g['valuenum'] = g.valuenum.where(g.valuenum.ge(1) & g.valuenum.le(g.itemid.map({220739:4,223900:5,223901:6})))
    # Soma somente componentes no MESMO horário; não cria um GCS fictício.
    g = g.pivot_table(index=['stay_id','charttime'], columns='itemid', values='valuenum', aggfunc='min')
    g = g.reindex(columns=[220739,223900,223901]).sum(axis=1, min_count=3)
    out['gcs_min_24h'] = g.groupby(level=0).min()
    for item, nome, lo, hi in [(223835,'fio2_max_24h',21,100),(220339,'peep_max_24h',0,30)]:
        q = b[b.itemid.eq(item)].copy()
        if item == 223835: q['valuenum'] = q.valuenum.where(q.valuenum.gt(1), q.valuenum*100)
        out[nome] = q[q.valuenum.between(lo,hi)].groupby('stay_id').valuenum.max()
    out['registro_ventilador_24h'] = out.index.isin(b.loc[
        b.itemid.isin([220339,224684,224685]) & b.valuenum.notna(), 'stay_id']).astype(int)
    w = b[b.itemid.isin([226512,226531,224639])].copy()
    w['peso'] = np.where(w.itemid.eq(226531), w.valuenum*.45359237, w.valuenum)
    out['peso_kg'] = w[w.peso.between(25,350)].groupby('stay_id').peso.median()
    return out


def extrair_sinais_labs(raiz, coorte, audit):
    ids = coorte.stay_id
    out = pd.DataFrame(index=pd.Index(ids, name='stay_id'))
    for tabela, mapa, chave, extras in [('chartevents',CHART,'stay_id',EXTRA_CHART),
                                       ('labevents',LAB,'hadm_id',set(LABS))]:
        print(f'Lendo {tabela}...', flush=True)
        sums, partes, total = [], [], 0
        for b in blocos(raiz,tabela,[chave,'itemid','charttime','storetime','valuenum','valueuom']):
            total += len(b)
            b = b[b.itemid.isin(set(mapa)|extras)]
            b = filtrar_tempos(b,coorte,chave)
            if b.empty: continue
            b['valuenum'] = pd.to_numeric(b.valuenum, errors='coerce')
            base = b[b.itemid.isin(mapa)].copy()
            base['variavel'] = base.itemid.map(mapa)
            base = limpar_eventos(base.rename(columns={'valuenum':'valor','valueuom':'unidade'}),audit)
            sums.append(base.groupby(['stay_id','variavel']).valor.agg(['sum','count']))
            partes.append(b[b.itemid.isin(extras)][['stay_id','itemid','charttime','valuenum','valueuom']])
            if len(sums)>=20: sums=[pd.concat(sums).groupby(level=[0,1]).sum()]
            if total % 10_000_000 == 0: print(f'  {total:,} linhas percorridas',flush=True)
        if not sums: raise ValueError(f'Sem eventos válidos em {tabela}')
        ag=pd.concat(sums).groupby(level=[0,1]).sum()
        out=out.join((ag['sum']/ag['count']).unstack().add_suffix('_24h'))
        extra=pd.concat(partes,ignore_index=True)
        if tabela=='chartevents': out=out.join(resumir_chart(extra,ids))
        else:
            for item,(nome,agg,lo,hi,units) in LABS.items():
                q=extra[extra.itemid.eq(item)].copy()
                u=q.valueuom.fillna('').str.lower().str.replace(' ','',regex=False)
                for unit,n in u[~u.isin(units)].value_counts().items(): audit[f'{nome}|{unit}']=int(n)
                out[nome]=q[q.valuenum.between(lo,hi)&u.isin(units)].groupby('stay_id').valuenum.agg(agg)
    out['ureia_24h']=out.bun_24h*2.14
    for flag,valor in [('lactato_medido','lactato_max_24h'),('nt_probnp_medido','nt_probnp_max_24h'),
                       ('troponina_medida','troponina_t_max_24h'),('gasometria_medida','ph_min_24h')]:
        out[flag]=out[valor].notna().astype(int)
    return out


def recortar_input(b, coorte):
    b=b[b.stay_id.isin(coorte.stay_id)].copy()
    b=b.merge(coorte[['stay_id','intime']],on='stay_id',validate='many_to_one')
    for c in ['starttime','endtime','storetime']: b[c]=pd.to_datetime(b[c],errors='coerce',format='mixed')
    limite=b.intime+pd.Timedelta(hours=24)
    b=b[b.storetime.lt(limite)&~b.statusdescription.fillna('').str.lower().eq('rewritten')].copy()
    limite=b.intime+pd.Timedelta(hours=24)
    for c in ['amount','rate','patientweight']: b[c]=pd.to_numeric(b[c],errors='coerce')
    dur=(b.endtime-b.starttime).dt.total_seconds()
    sobre=(b.endtime.clip(upper=limite)-b.starttime.clip(lower=b.intime)).dt.total_seconds().clip(lower=0)
    pontual=b.starttime.ge(b.intime)&b.starttime.lt(limite)&dur.eq(0)
    b['segundos_janela']=sobre
    b['quantidade']=b.amount*np.where(dur.gt(0),sobre/dur.where(dur.gt(0)),pontual.astype(float))
    b=b[(sobre.gt(0)|pontual)&dur.ge(0)].copy()
    # Se cruza o corte, não consulta o total final. Integra a taxa conhecida.
    cruza=b.endtime.ge(b.intime+pd.Timedelta(hours=24))
    b.loc[cruza,'quantidade']=np.nan
    for unidade,fator in [('mL/hour',1/3600),('mg/hour',1/3600),('mg/min',1/60)]:
        m=cruza&b.rateuom.eq(unidade)
        b.loc[m,'quantidade']=b.loc[m,'rate']*b.loc[m,'segundos_janela']*fator
    return b


def resumir_input(b, ids):
    out=pd.DataFrame(index=pd.Index(ids,name='stay_id'))
    # Um registro com dose/taxa explicitamente zero não comprova administração.
    ativo=b[(b.amount.gt(0)|b.rate.gt(0))].copy()
    for item,nome in VASO.items(): out[nome+'_24h']=out.index.isin(ativo.loc[ativo.itemid.eq(item),'stay_id']).astype(int)
    out['n_vasopressores_24h']=out[[n+'_24h' for n in VASO.values()]].sum(axis=1)
    out['vasopressor_24h']=out.n_vasopressores_24h.gt(0).astype(int)
    for nome,itens in [('inotropico_24h',INOTROPICOS),('diuretico_alca_24h',ALCA),
                       ('furosemida_infusao_continua_24h',{228340})]:
        out[nome]=out.index.isin(ativo.loc[ativo.itemid.isin(itens),'stay_id']).astype(int)
    n=ativo[ativo.itemid.eq(221906)].copy()
    n['taxa']=n.rate.where(n.rateuom.eq('mcg/kg/min'))
    peso=n.patientweight.where(n.patientweight.between(25,350))
    n.loc[n.rateuom.eq('mcg/min'),'taxa']=n.rate/peso
    n.loc[n.rateuom.eq('mcg/kg/hour'),'taxa']=n.rate/60
    n.loc[n.rateuom.eq('mcg/hour'),'taxa']=n.rate/(60*peso)
    n.loc[n.rateuom.eq('mg/hour'),'taxa']=n.rate*1000/(60*peso)
    out['noradrenalina_taxa_max_24h']=n[n.taxa.between(0,5)].groupby('stay_id').taxa.max()
    out.loc[out.noradrenalina_24h.eq(0),'noradrenalina_taxa_max_24h']=0
    f=ativo[ativo.itemid.isin(FURO)].copy()
    f['mg']=f.quantidade.where(f.amountuom.eq('mg'))
    f.loc[f.amountuom.eq('mcg'),'mg']=f.quantidade/1000
    f.loc[f.amountuom.eq('g'),'mg']=f.quantidade*1000
    out['furosemida_dose_total_24h']=f.groupby('stay_id').mg.sum(min_count=1)
    out.loc[~out.index.isin(f.stay_id),'furosemida_dose_total_24h']=0
    # Não somar ingestão oral, pré-admissão, irrigação ou soluções de diálise ao IV.
    iv=b[b.amountuom.eq('mL')&~b.ordercategoryname.fillna('').str.contains('Oral|Pre Admission|Irrig|Dialys',case=False,regex=True)
         &~b.itemid.eq(226401)].copy()
    iv['quantidade']=iv.quantidade.where(iv.quantidade.ge(0))
    out['volume_infundido_24h']=iv.groupby('stay_id').quantidade.sum(min_count=1)
    out.loc[~out.index.isin(iv.stay_id),'volume_infundido_24h']=0
    # Se um componente IV foi registrado mas não pode ser quantificado, total desconhecido.
    out.loc[out.index.isin(iv.loc[iv.quantidade.isna(),'stay_id']),'volume_infundido_24h']=np.nan
    out.loc[out.index.isin(f.loc[f.mg.isna(),'stay_id']),'furosemida_dose_total_24h']=np.nan
    bolus=ativo.itemid.isin(CRISTALOIDES)&(
        ativo.ordercategorydescription.fillna('').str.contains('bolus',case=False)|
        ativo.ordercategoryname.fillna('').str.contains('bolus',case=False))
    out['bolus_cristaloide_24h']=out.index.isin(ativo.loc[bolus,'stay_id']).astype(int)
    out['peso_input_kg']=b[b.patientweight.between(25,350)].groupby('stay_id').patientweight.median()
    return out


def resumir_output(b, ids, peso):
    out=pd.DataFrame(index=pd.Index(ids,name='stay_id'))
    b=b[b.valueuom.eq('mL')&b.value.between(0,5000)].copy()
    b.loc[b.itemid.eq(227488),'value'] *= -1  # descontar irrigante, conforme mimic-code
    urina=b[b.itemid.isin(URINA)]
    out['diurese_total_24h']=urina.groupby('stay_id').value.sum(min_count=1)
    out.loc[out.diurese_total_24h.lt(0),'diurese_total_24h']=np.nan
    out['saida_total_24h']=b.groupby('stay_id').value.sum(min_count=1)
    out.loc[out.saida_total_24h.lt(0),'saida_total_24h']=np.nan
    out['diurese_medida']=out.diurese_total_24h.notna().astype(int)
    out['diurese_ml_kg_h_24h']=out.diurese_total_24h/(peso*24)
    out['oliguria_24h']=out.diurese_ml_kg_h_24h.lt(.5).astype(float).where(out.diurese_ml_kg_h_24h.notna())
    return out


def extrair_io(raiz,coorte,peso):
    cols=['stay_id','starttime','endtime','storetime','itemid','amount','amountuom','rate',
          'rateuom','ordercategoryname','ordercategorydescription','patientweight','statusdescription']
    print('Lendo inputevents...',flush=True)
    partes=[recortar_input(b,coorte) for b in blocos(raiz,'inputevents',cols)]
    entrada=resumir_input(pd.concat(partes,ignore_index=True),coorte.stay_id)
    peso=peso.fillna(entrada.peso_input_kg)
    print('Lendo outputevents...',flush=True)
    partes=[filtrar_tempos(b,coorte) for b in blocos(raiz,'outputevents',
            ['stay_id','charttime','storetime','itemid','value','valueuom'])]
    saida=resumir_output(pd.concat(partes,ignore_index=True),coorte.stay_id,peso)
    out=entrada.join(saida)
    out['peso_kg_io']=peso
    out['balanco_hidrico_24h']=out.volume_infundido_24h-out.saida_total_24h
    out['balanco_ml_kg_24h']=out.balanco_hidrico_24h/peso
    return out


def selecionar_feve(medidas,coorte):
    nomes=['biplane_lvef','lvef','lvef_3d','ef','ejection_fraction','lv_ef']
    m=medidas.copy()
    m['measurement']=m.measurement.str.lower().str.strip()
    m=m[m.measurement.isin(nomes)&m.subject_id.isin(coorte.subject_id)].copy()
    m['feve']=pd.to_numeric(m.result,errors='coerce')
    m['feve']=m.feve.where(m.feve.gt(1.5),m.feve*100)
    m['measurement_datetime']=pd.to_datetime(m.measurement_datetime,errors='coerce',format='mixed')
    m=m[m.feve.between(5,85)].copy()
    m['prioridade']=m.measurement.map({n:i for i,n in enumerate(nomes)})
    m=m.sort_values('prioridade').drop_duplicates(['subject_id','measurement_id'])
    m=coorte[['subject_id','stay_id','intime']].merge(m,on='subject_id')
    delta=m.measurement_datetime-m.intime
    m=m[delta.ge(pd.Timedelta(hours=-48))&delta.lt(pd.Timedelta(hours=24))].copy()
    m['distancia']=(m.measurement_datetime-m.intime).abs()
    return m.sort_values(['distancia','measurement_datetime']).drop_duplicates('stay_id').set_index('stay_id').feve


def extrair_completo(raiz='data/mimic-iv', saida='data/processados/mimic_completo_24h.csv'):
    raiz,saida=Path(raiz),Path(saida)
    for t in ['patients','admissions','diagnoses_icd','icustays','chartevents','labevents','inputevents','outputevents','d_items']:
        localizar_tabela(raiz,t)
    eco=localizar_tabela(raiz.parent/'mimic-iv-structured-measurement','structured-measurement')
    c=coorte_mimic(raiz)
    c=c[c.idade.ge(18)&c.los_dias.gt(1)&np.isfinite(c.los_dias)].copy()
    a=pd.read_csv(localizar_tabela(raiz,'admissions'),usecols=['hadm_id','admission_type'])
    c=c.merge(a,on='hadm_id',validate='many_to_one')
    nao_mapeados=sorted(set(c.admission_type.dropna())-set(ADMISSION))
    if nao_mapeados: raise ValueError(f'Tipos de admissão sem mapeamento: {nao_mapeados}')
    c['tipo_admissao']=c.admission_type.map(ADMISSION)
    c['genero']=c.genero.where(c.genero.isin(['F','M']))
    audit={}
    sinais=extrair_sinais_labs(raiz,c,audit)
    io=extrair_io(raiz,c,sinais.peso_kg)
    sinais=sinais.join(io)
    print('Lendo medidas de fração de ejeção...',flush=True)
    partes=[]
    for b in pd.read_csv(eco,chunksize=500_000,low_memory=False):
        partes.append(b[b.measurement.str.lower().isin(['biplane_lvef','lvef','lvef_3d','ef','ejection_fraction','lv_ef'])])
    sinais['feve']=selecionar_feve(pd.concat(partes,ignore_index=True),c)
    c=c.merge(sinais.reset_index(),on='stay_id',validate='one_to_one')
    c['versao_painel']=VERSAO
    meta_cols=['patient_id','stay_id','hospital_id','los_dias','obito_hospitalar','raca',
               'primeira_uti_internacao','ic_principal','versao_painel']
    saida.parent.mkdir(parents=True,exist_ok=True)
    c[meta_cols+colunas('completo')].to_csv(saida,index=False)
    pd.DataFrame({'variavel':colunas('completo'),
        'fracao_ausente':[float(c[v].isna().mean()) for v in colunas('completo')],
        'valores_distintos':[int(c[v].nunique()) for v in colunas('completo')]}).to_csv(saida.with_name('cobertura_completo.csv'),index=False)
    meta={'versao_painel':VERSAO,'n_estadias':len(c),'features':colunas('completo'),
          'unidades_rejeitadas':audit,'resumo_base':'médias; temperatura mediana; demais extremos conforme nome',
          'ureia':'BUN mg/dL x 2.14 = ureia mg/dL','feve':'mais próxima da entrada em [-48h,+24h); sem timestamp de liberação disponível',
          'gcs':'menor soma dos três componentes no mesmo charttime; intubado não é convertido em resposta normal',
          'oliguria':'média de diurese registrada por kg/24h, não diagnóstico KDIGO; sem urina/peso = ausente',
          'medicamentos':'administração documentada em inputevents; ausência de registro não comprova ausência de tratamento',
          'balanco':'volume IV registrado menos saídas registradas, corrigido irrigante; oral excluído',
          'fontes':{t:str(localizar_tabela(raiz,t)) for t in ['chartevents','labevents','inputevents','outputevents','d_items']},
          'fonte_feve':str(eco)}
    saida.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Painel completo salvo: {saida} ({len(c):,} estadias, {len(colunas("completo"))} variáveis)',flush=True)
    return c[meta_cols+colunas('completo')]


if __name__=='__main__': extrair_completo()
