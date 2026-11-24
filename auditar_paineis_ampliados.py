"""Verificações de contrato, ausência, distribuição e cobertura hospitalar."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experimentos_24h import TODAS, ROOT, harmonizar, carregar_painel

def auditar():
    ROOT.mkdir(parents=True,exist_ok=True)
    frames={b:carregar_painel(b) for b in ['mimic','eicu']}
    rows=[]; hospitais=[]; summary={}
    for base,d in frames.items():
        assert d.stay_id.notna().all() and not d.stay_id.duplicated().any()
        assert d.patient_id.notna().all() and d.idade.ge(18).all() and d.los_dias.gt(1).all()
        assert not np.isinf(d.select_dtypes('number')).any().any()
        assert d.aki_creatinina_estagio_24h.dropna().isin([0,1,2,3]).all()
        assert d.sofa_24h.dropna().between(0,24).all()
        assert d.sofa_componentes_observados.between(0,6).all()
        assert d.diurese_total_24h.dropna().ge(0).all()
        assert np.allclose(d.balanco_hidrico_24h,d.volume_infundido_24h-d.saida_total_24h,equal_nan=True)
        for flag,value in [('lactato_medido','lactato_max_24h'),('nt_probnp_medido','nt_probnp_max_24h'),
                           ('troponina_medida','troponina_t_max_24h'),('gasometria_medida','ph_min_24h')]:
            assert d[flag].eq(d[value].notna().astype(int)).all()
        for col in TODAS:
            x=d[col]
            record={'base':base,'variavel':col,'n':len(x),'ausentes':int(x.isna().sum()),
                    'fracao_ausente':float(x.isna().mean()),'n_distintos':int(x.nunique())}
            if pd.api.types.is_numeric_dtype(x):
                record.update({f'p{q:g}':float(x.quantile(q/100)) for q in [1,50,99]})
            rows.append(record)
        for h,g in d.groupby('hospital_id'):
            record={'base':base,'hospital_id':h,'n':len(g)}
            for col in ['aki_creatinina_estagio_24h','sofa_24h','ausculta_alterada_24h','diurese_total_24h','volume_infundido_24h','troponina_t_max_24h']:
                record['cobertura_'+col]=float(g[col].notna().mean())
            record['registro_vasopressor']=float(g.vasopressor_24h.mean())
            hospitais.append(record)
        summary[base]={'estadias':len(d),'pacientes':d.patient_id.nunique(),'hospitais':d.hospital_id.nunique(),
            'prevalencia_355':d.los_dias.gt(3.55).mean(),'prevalencia_7':d.los_dias.gt(7).mean()}
    assert not set(frames['mimic'].patient_id)&set(frames['eicu'].patient_id)
    common=harmonizar(frames['mimic'],frames['eicu'])
    for col in common:
        assert frames['mimic'][col].notna().any() and frames['eicu'][col].notna().any()
    summary['n_preditores_completo']=len(TODAS);summary['n_preditores_harmonizado']=len(common)
    pd.DataFrame(rows).to_csv(ROOT/'qualidade_variaveis.csv',index=False)
    pd.DataFrame(hospitais).to_csv(ROOT/'cobertura_por_hospital.csv',index=False)
    (ROOT/'auditoria_resumo.json').write_text(json.dumps(summary,indent=2,default=float),encoding='utf-8')
    print(json.dumps(summary,indent=2,default=float),flush=True)
    return summary

if __name__=='__main__':auditar()
