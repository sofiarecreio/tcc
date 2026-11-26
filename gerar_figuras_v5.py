"""Exporta somente figuras agregadas da v5, em PNG/PDF e ZIP para publicacao."""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import joblib
from threadpoolctl import threadpool_limits
from experimentos_24h import carregar_painel, dividir_preservando
from avaliar_protocolo_24h import explicar

ROOT=Path('artefatos_protocolo_24h/experimentos_v5')
OUT=Path('figuras_artigo_v5')
CUTS=[3.55,7.]
SERIES=[('completo','mimic','MIMIC completo','#2166ac'),
        ('harmonizado','mimic','MIMIC harmonizado','#1b7837'),
        ('harmonizado','eicu','eICU harmonizado','#b35806')]

def save(fig,name):
    fig.tight_layout()
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    metrics=pd.read_csv(ROOT/'classificacao.csv')
    # Verifica a AUC com as previsoes, sem exportar registros individuais.
    for r in metrics.itertuples():
        p=pd.read_csv(ROOT/r.perfil/f'{r.corte:g}d'/r.base/'previsoes.csv')
        assert abs(roc_auc_score(p.alvo,p.probabilidade)-r.auc)<1e-10
    for kind,title in [('roc','Curva ROC'),('calibracao','Calibracao'),('dca','Curva de decisao')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,cut in zip(axes,CUTS):
            for profile,base,label,color in SERIES:
                folder=ROOT/profile/f'{cut:g}d'/base
                if kind=='roc':
                    d=pd.read_csv(folder/'previsoes.csv'); x,y,_=roc_curve(d.alvo,d.probabilidade)
                    auc=roc_auc_score(d.alvo,d.probabilidade)
                    ax.plot(x,y,label=f'{label}: AUC {auc:.3f}',color=color)
                elif kind=='calibracao':
                    d=pd.read_csv(folder/'calibracao.csv')
                    ax.plot(d.probabilidade_media,d.frequencia_observada,'o-',label=label,color=color)
                else:
                    d=pd.read_csv(folder/'dca.csv'); ax.plot(d.limiar,d.modelo,label=label,color=color)
                    # A estrategia todos depende da prevalencia em cada base.
                    if profile=='harmonizado':
                        ax.plot(d.limiar,d.tratar_todos,'--',color=color,alpha=.55,label=f'Todos: {base.upper()}')
            if kind in ['roc','calibracao']:
                ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlim=(0,1),ylim=(0,1))
            if kind=='roc': ax.set(xlabel='1 - especificidade',ylabel='Sensibilidade')
            elif kind=='calibracao': ax.set(xlabel='Probabilidade prevista',ylabel='Frequencia observada')
            else:
                ax.axhline(0,color='black',ls=':',label='Nenhum')
                ax.set(xlim=(0,.8),ylim=(-.1,.5),xlabel='Limiar de decisao',ylabel='Beneficio liquido')
            ax.set_title(f'{title} - LOS > {cut:g} dias'); ax.legend(fontsize=8); ax.grid(alpha=.15)
        save(fig,{'roc':'01_roc','calibracao':'02_calibracao','dca':'03_dca'}[kind])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,cut in zip(axes,CUTS):
        for i,(profile,base,label,color) in enumerate(SERIES):
            r=metrics[(metrics.perfil==profile)&(metrics.base==base)&(metrics.corte==cut)].iloc[0]
            ax.errorbar(r.auc,i,xerr=[[r.auc-r.auc_ic95_inf],[r.auc_ic95_sup-r.auc]],fmt='o',capsize=4,color=color)
        ax.set(yticks=range(3),yticklabels=[s[2] for s in SERIES],xlim=(.5,1),xlabel='AUC (IC95% por paciente)',title=f'LOS > {cut:g} dias')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    save(fig,'04_auc_ic95')
    _,test=dividir_preservando(carregar_painel('mimic'))
    for profile in ['completo','harmonizado']:
        fig,axes=plt.subplots(1,2,figsize=(16,7))
        for ax,cut in zip(axes,CUTS):
            folder=ROOT/profile/f'{cut:g}d'
            fit=joblib.load(folder/'modelo_selecionado.joblib')
            assert not set(test.patient_id)&fit['pacientes_treino']
            explicar(fit['modelo'],test[fit['features']],folder/'mimic')
            s=pd.read_csv(folder/'mimic/shap.csv').head(15).iloc[::-1]
            labels=s.variavel.str.replace('num__','',regex=False).str.replace('cat__','',regex=False)
            ax.barh(labels,s.shap_absoluto_medio,color='#2166ac')
            ax.set(xlabel='Media de |SHAP| (saida bruta)',title=f'{profile.capitalize()} - LOS > {cut:g} dias')
        save(fig,'05_shap_'+profile)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,profile in zip(axes,['completo','harmonizado']):
        vals=[json.loads((ROOT/profile/f'{c:g}d/selecao.json').read_text())['candidatos'] for c in CUTS]
        names=list(vals[0]); x=np.arange(len(names))
        for i,c in enumerate(CUTS): ax.bar(x+(i-.5)*.35,[vals[i][n] for n in names],width=.35,label=f'>{c:g} dias')
        ax.set(xticks=x,xticklabels=['XGBoost','Random Forest','MLP'],ylim=(0,1),ylabel='AUC media na CV de selecao',title=profile.capitalize())
        ax.legend()
    save(fig,'06_comparacao_cv')
    text='''# Figuras do modelo v5

PNG: 300 dpi. PDF: formato vetorial para diagramacao. Resultados exploratorios.

- 01_roc: discriminacao no teste MIMIC e validacao externa eICU.
- 02_calibracao: probabilidades previstas versus frequencias observadas.
- 03_dca: beneficio liquido estimado; nao demonstra beneficio clinico prospectivo. Eixo de limiar limitado a 0-0,8 e beneficio a -0,1-0,5 para legibilidade.
- 04_auc_ic95: AUC e IC95% por bootstrap agrupado por paciente, dos resultados salvos.
- 05_shap_completo/harmonizado: 15 variaveis de maior SHAP absoluto medio; 500 estadias de teste MIMIC, amostradas com semente 42. Saida bruta do XGBoost. Nao indica causalidade ou direcao do efeito.
- 06_comparacao_cv: melhores configuracoes por familia na CV do treino; nao confundir com desempenho independente no teste.

Completo: 88 variaveis; harmonizado: 83. Sao modelos distintos treinados no MIMIC. Para avaliar transporte entre bases, compare harmonizado MIMIC e harmonizado eICU.
O XGBoost foi escolhido pela CV em todos os cenarios. Testes interno e externo foram consultados no desenvolvimento: resultados exploratorios.
As figuras foram geradas exclusivamente da v5, sem misturar v3/v4. Nao incluem regressao nem ablacoes, que nao foram refeitas na v5.
Fonte local: artefatos_protocolo_24h/experimentos_v5. Reproducao: python gerar_figuras_v5.py (requer os dados e modelos locais autorizados).
O pacote contem somente figuras agregadas e este texto. Nao contem identificadores, tabelas individuais, modelos serializados ou bases clinicas.
'''
    (OUT/'LEIA-ME.txt').write_text(text,encoding='utf8')
    files=sorted([p for p in OUT.iterdir() if p.suffix in ['.png','.pdf']]+[OUT/'LEIA-ME.txt'])
    with zipfile.ZipFile('figuras_artigo_v5.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p,arcname=f'figuras_artigo_v5/{p.name}')
    print(f'ZIP criado: {len(files)} arquivos',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=3): main()
