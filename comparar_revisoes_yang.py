import pandas as pd
from pathlib import Path
from experimentos_24h import diferenca_pareada
R=Path('artefatos_protocolo_24h/experimentos_v3'); ref='yang_aproximado_ic_seq1a2_aki_kdigo48'
vs=['','_arritmia','_ausculta','_valvulopatia','_arritmia_ausculta','_arritmia_ausculta_valvulopatia']
rows=[]
for v in vs:
    f=R/(ref+v)
    if not (f/'comparacao.csv').exists(): continue
    c=pd.read_csv(f/'comparacao.csv')
    s=pd.read_csv(f/'mimic/xgboost/shap.csv').variavel.str.replace('num__','')
    for _,r in c.iterrows():
        row=dict(revisao=v.strip('_') or 'nenhuma (KDIGO 48h)',modelo=r.modelo,base=r.base,auc=r.auc,ic95_inf=r.auc_ic95_inf,ic95_sup=r.auc_ic95_sup,shap_top3=', '.join(s.head(3)))
        if v:
            a=pd.read_csv(f/r.base/r.modelo/'previsoes.csv'); b=pd.read_csv(R/ref/r.base/r.modelo/'previsoes.csv')
            dd=diferenca_pareada(a,b,500); row.update(delta_vs_kdigo48=dd['delta_completo_menos_variante'],delta_ic95_inf=dd['ic95_inf'],delta_ic95_sup=dd['ic95_sup'])
        rows.append(row)
out=pd.DataFrame(rows); out.to_csv(R/'yang_revisoes_comparacao.csv',index=False)
pd.set_option('display.width',250); print(out.round(3).to_string(index=False))
