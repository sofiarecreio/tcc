"""Critérios explícitos, testáveis e sem uso de valores posteriores ao corte."""
import numpy as np
import pandas as pd

def arritmias(texto):
    """Evita confundir taquicardia supraventricular com ventricular."""
    t=texto.fillna('').str.lower()
    vent=t.str.contains(r'\bv\s*fib|\bv\s*tach|(?<!supra)(?<!supra )\bventricular (?:fibrillation|fib|tachycardia|tach|flutter)',regex=True)
    ampla=vent|t.str.contains(r'\ba\s*fib|atrial fibrillation|atrial fib|3rd|third.degree|complete.*(?:heart|av).*block',regex=True)
    return ampla,vent

def janela(d, horas=24, anteriores=False):
    return d[d.offset.lt(horas*60)&d.disponivel.lt(horas*60)&
             (d.offset.ge(-10080) if anteriores else d.offset.ge(0))].copy()

def aki_creatinina(eventos, ids):
    """Maior estágio pelo critério de creatinina KDIGO; não inclui RRT/diurese.

    Basal = menor medida PRÉVIA nos últimos sete dias; subida absoluta em 48h.
    Sem medida anterior comparável, o estágio é desconhecido (não zero).
    """
    out=pd.Series(np.nan,index=pd.Index(ids,name='stay_id'),name='aki_creatinina_estagio_24h')
    d=janela(eventos,anteriores=True)
    d=d[d.valor.between(.1,30)].groupby(['stay_id','offset'],as_index=False).valor.mean()
    for stay,g in d.groupby('stay_id'):
        g=g.sort_values('offset'); t=g.offset.to_numpy(); x=g.valor.to_numpy(); scores=[]
        for i in np.flatnonzero(t>=0):
            before=(t<t[i])&(t>=t[i]-10080)
            if not before.any(): continue
            b7=x[before].min(); h48=before&(t>=t[i]-2880)
            increase=bool(h48.any() and x[i]-x[h48].min()>=.3-1e-9)
            ratio=x[i]/b7
            stage=3 if ratio>=3 or (x[i]>=4 and (increase or ratio>=1.5)) else (
                2 if ratio>=2 else (1 if increase or ratio>=1.5 else 0))
            scores.append(stage)
        if scores: out.loc[stay]=max(scores)
    return out

def taxa_mcg_kg_min(rate, unidade, peso):
    rate=pd.to_numeric(rate,errors='coerce')
    peso=pd.to_numeric(peso,errors='coerce').where(lambda x:x.between(25,350))
    u=unidade.fillna('').str.lower().str.replace(' ','',regex=False).str.replace('hour','hr',regex=False)
    out=pd.Series(np.nan,index=rate.index)
    for unit,factor in [('mcg/kg/min',1),('mcg/kg/hr',1/60),('mg/kg/min',1000)]:
        out.loc[u.eq(unit)]=rate*factor
    for unit,factor in [('mcg/min',1),('mcg/hr',1/60),('mg/min',1000),('mg/hr',1000/60)]:
        out.loc[u.eq(unit)]=rate*factor/peso
    return out.where(out.ge(0))

def sofa(d):
    """SOFA de primeiro dia adaptado a janela [0,24h), com cobertura explícita.

    Respiração: PaO2 arterial com FiO2 conhecida e suporte registrado no horário.
    Componente ausente contribui zero à soma, convenção do mimic-code; o número
    de componentes observados é preservado. Não se chama um SOFA completo observado.
    """
    out=pd.DataFrame(index=d.index)
    def escore(col,limits,asc=True):
        x=d[col]; v=sum((x.ge(t) if asc else x.lt(t)).astype(int) for t in limits)
        return v.astype(float).where(x.notna())
    out['sofa_coagulacao']=escore('plaquetas_min_24h',[150,100,50,20],False)
    out['sofa_hepatico']=escore('bilirrubina_max_24h',[1.2,2,6,12])
    out['sofa_neurologico']=escore('gcs_min_24h',[15,13,10,6],False)
    renal=escore('creatinina_max_24h',[1.2,2,3.5,5])
    urine=d.diurese_total_24h
    u=pd.Series(np.select([urine.lt(200),urine.lt(500)],[4,3],default=0),index=d.index).where(urine.notna())
    out['sofa_renal']=pd.concat([renal,u],axis=1).max(axis=1)
    norepi=d.noradrenalina_taxa_max_24h; epi=d.adrenalina_taxa_max_24h; dopa=d.dopamina_taxa_max_24h
    cv=pd.Series(np.select([
        dopa.gt(15)|epi.gt(.1)|norepi.gt(.1),
        dopa.gt(5)|epi.gt(0)|norepi.gt(0),
        dopa.gt(0)|d.dobutamina_24h.eq(1), d.pam_min_24h.lt(70)], [4,3,2,1],default=0),index=d.index,dtype=float)
    out['sofa_cardiovascular']=cv.where(d.pam_min_24h.notna()|dopa.gt(0)|epi.gt(0)|norepi.gt(0)|d.dobutamina_24h.eq(1))
    unknown=((d.get('noradrenalina_24h',norepi.gt(0)).eq(1)&norepi.isna())|
             (d.get('adrenalina_24h',epi.gt(0)).eq(1)&epi.isna())|
             (d.get('dopamina_24h',dopa.gt(0)).eq(1)&dopa.isna()))
    out.loc[unknown&cv.lt(4),'sofa_cardiovascular']=np.nan
    pf=d.pafi_min_24h; pfv=d.pafi_suporte_min_24h
    out['sofa_respiratorio']=pd.Series(np.select([pfv.lt(100),pfv.lt(200),pf.lt(300),pf.lt(400)],[4,3,2,1],default=0),index=d.index,dtype=float).where(pf.notna())
    out['sofa_componentes_observados']=out.notna().sum(axis=1)
    components=[c for c in out if c!='sofa_componentes_observados']
    out['sofa_24h']=out[components].fillna(0).sum(axis=1).where(out.sofa_componentes_observados.gt(0))
    return out

def pafi(pao2, fio2, suporte, ids):
    """Pareamento retrospectivo: último FiO2/suporte até 4h ANTES da gasometria.

    Nunca divide PaO2 mínimo por FiO2 máximo de outro horário.
    `pao2` deve conter exclusivamente gasometrias arteriais identificadas.
    """
    out=pd.DataFrame(index=pd.Index(ids,name='stay_id'),columns=['pafi_min_24h','pafi_suporte_min_24h'],dtype=float)
    if pao2.empty or fio2.empty: return out
    a=pao2[['stay_id','offset','valor']].sort_values('offset').rename(columns={'valor':'pao2'})
    b=fio2[['stay_id','offset','valor']].sort_values('offset').rename(columns={'valor':'fio2'})
    b['fio2']=b.fio2.where(b.fio2.le(1),b.fio2/100)
    b=b[b.fio2.between(.21,1)]
    b=b.drop_duplicates(['stay_id','offset'],keep='last')
    for frame in [a,b]: frame['offset']=frame.offset.astype(float)
    x=pd.merge_asof(a,b,on='offset',by='stay_id',direction='backward',tolerance=240.)
    x['ratio']=x.pao2/x.fio2
    x=x[x.pao2.between(20,800)&x.ratio.between(20,2000)]
    out['pafi_min_24h']=x.groupby('stay_id').ratio.min()
    if not suporte.empty:
        s=suporte[['stay_id','offset','valor']].sort_values('offset').rename(columns={'valor':'suporte'})
        s['offset']=s.offset.astype(float)
        x=pd.merge_asof(x.sort_values('offset'),s,on='offset',by='stay_id',direction='backward',tolerance=240.)
        out['pafi_suporte_min_24h']=x[x.suporte.eq(1)].groupby('stay_id').ratio.min()
    return out

def ausculta(registros,ids):
    out=pd.DataFrame(index=pd.Index(ids,name='stay_id'))
    text=registros.texto.fillna('').str.lower()
    measured=set(registros.loc[text.ne(''),'stay_id'])
    maps={'roncos_24h':r'rhonch|ronc','sibilos_24h':r'wheez|sibil',
          'estertores_24h':r'crackl|rales|crepit','estridor_24h':r'stridor',
          'murmurio_diminuido_24h':r'diminish|decreas|absent'}
    for name,regex in maps.items():
        negated=text.str.contains(r'(?:no|without|absent)\s+(?:'+regex+r')',regex=True)
        if name=='murmurio_diminuido_24h':
            negated=text.str.contains(r'no\s+(?:decreased|diminished|absent)',regex=True)
        positives=set(registros.loc[text.str.contains(regex,regex=True)&~negated,'stay_id'])
        out[name]=out.index.isin(positives).astype(float)
        out.loc[~out.index.isin(measured),name]=np.nan
    out['ausculta_alterada_24h']=out[list(maps)].max(axis=1)
    out['ausculta_registrada_24h']=out.index.isin(measured).astype(int)
    return out
