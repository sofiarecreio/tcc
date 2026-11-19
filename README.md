# TCC — previsão de permanência na UTI em insuficiência cardíaca

Abra **protocolo_24h_mimic_eicu.ipynb** com o kernel de `.venv`.
O protocolo ampliado usa **70 variáveis no MIMIC** e **65 harmonizadas com o eICU**.
Inclui as 54 variáveis anteriores, `inputevents`/`outputevents`, SBP/PAS, AKI por
creatinina, SOFA adaptado, arritmias, ausculta, valvulopatia prévia e cateterismo.
Compara XGBoost e rede neural MLP, ajustados exclusivamente no treino MIMIC.

As definições, fontes e limitações estão em [docs/protocolo_ampliado.md](docs/protocolo_ampliado.md).
O benchmark aproximado de Yang fica separado do protocolo de 24h.

## Resultados da ampliação

Execução de 19/11/2026. AUC de teste dos classificadores ajustados:

| Alvo | MIMIC, 70 campos: XGBoost | MIMIC: MLP | eICU, 65 campos: XGBoost | eICU: MLP |
|---|---:|---:|---:|---:|
| LOS >3,55 dias | 0,741 | 0,748 | 0,666 | 0,626 |
| LOS >7 dias | 0,758 | 0,754 | 0,687 | 0,654 |

O XGBoost externo anterior, com 13 campos, tinha AUC 0,620/0,637. As diferenças
internas entre MLP e XGBoost são incertas nos ICs pareados; no eICU, o XGBoost
teve AUC maior nos dois cortes. Na ablação sob o mesmo ajuste, as 16 novas variáveis
acrescentaram cerca de 0,005 de AUC para >3,55 dias e 0,001 para >7 dias.
Retirar entradas/saídas e SOFA reduziu a AUC em aproximadamente 0,015 nos dois alvos.
Essas comparações são exploratórias e não foram corrigidas por multiplicidade.

Indicadores de ausência foram a variante escolhida pela CV do treino nos dois
cortes; no teste, tiveram AUC 0,744/0,757. Não substituem silenciosamente o painel
principal. O benchmark Yang aproximado obteve XGBoost 0,720 interno/0,660 externo,
com 1.492/3.744 estadias nas coortes: não reproduz os números publicados.
ICs, calibração, DCA, SHAP, regressão e avisos de ajuste estão no notebook.

## Onde armazenar os dados

Extraia somente o ZIP externo. Os `.csv.gz` internos podem continuar comprimidos.
O programa aceita pastas de versão e interrompe se encontrar duas cópias da mesma tabela.

```text
data/
  mimic-iv/mimiciv-3.1/
    hosp/   # patients, admissions, diagnoses_icd, labevents
    icu/    # icustays, chartevents, inputevents, outputevents, procedureevents, d_items
  mimic-iv-structured-measurement/1.0.1/structured-measurement.csv.gz
  eicu/eicu-collaborative-research-database-2.0/
    # patient, diagnosis, lab, vitalPeriodic, vitalAperiodic, nurseCharting,
    # respiratoryCharting, infusionDrug, intakeOutput, medication,
    # pastHistory, physicalExam, treatment
  processados/   # painéis e cache local de eventos selecionados
```

Os dados, modelos e previsões individuais estão excluídos do Git pelo `.gitignore`.
Downloads: [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/) e
[eICU-CRD 2.0](https://physionet.org/content/eicu-crd/2.0/), com acesso credenciado.
A FEVE usa medidas estruturadas de Echo. ECG não é entrada do fluxo atual.

## Executar o fluxo ampliado

Os dados já extraídos podem ser reutilizados. Para reproduzir tudo, no terminal da pasta TCC:

```powershell
.\.venv\Scripts\python.exe verificar_dados.py
.\.venv\Scripts\python.exe extrair_fontes_ampliadas.py mimic
.\.venv\Scripts\python.exe extrair_fontes_ampliadas.py eicu
.\.venv\Scripts\python.exe painel_ampliado.py mimic
.\.venv\Scripts\python.exe painel_ampliado.py eicu
.\.venv\Scripts\python.exe auditar_paineis_ampliados.py
.\.venv\Scripts\python.exe experimentos_24h.py --etapa tudo
.\.venv\Scripts\python.exe experimento_populacao.py --rebusca
```

A instalação inicial usa `python -m venv .venv` e
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
O painel ampliado reaproveita FEVE de `data/processados/mimic_completo_24h.csv`;
se esse arquivo não existir, execute `painel_completo_24h.py` antes.
A partição por paciente é preservada de `artefatos_protocolo_24h/completo/divisao.csv`;
para criar o histórico em uma instalação nova, execute
`avaliar_protocolo_24h.py treinar --perfil completo` antes dos experimentos ampliados.

A extração bruta armazena partes selecionadas com tamanho/data de origem. Se mudar
as regras do cache, incremente sua versão. Depois de alterar a extração, gere novamente
os painéis. As buscas verificam assinatura dos dados e configuração antes de reutilizar modelos.

`--etapa` também aceita `principal`, `externo`, `ablacoes` e `yang`.
A ablação depende da busca principal já concluída. O notebook apresenta os resultados
salvos; refazer os treinos é uma opção explícita e pode demorar.

## As cinco frentes

| Frente | Implementação |
|---|---|
| Ampliar eICU | 65 preditores compatíveis, com relatório de ausências/exclusões |
| Ajustar modelos | Busca XGBoost/MLP com CV em três partes por paciente, somente no treino |
| Revisar extração | Corte temporal e disponibilidade, unidades, duplicações, escores e auditoria por hospital |
| Testar contribuição | Seis ablações: completo, sem novas, sem eventos de entrada/saída e SOFA, cobertura >70%, indicadores de ausência, máximos dos exames |
| Comparar com Yang | Experimento aproximado separado: IC principal, sobreviventes, primeira UTI da internação, dez conceitos e tratamentos até 48h |

Classificação para LOS >3,55 e >7 dias, regressão de LOS em dias, AUC/IC95%, AUC-PR,
VPP/VPN, Brier, calibração, DCA, SHAP, subgrupos e sensibilidade por óbito.
ICs de AUC e diferenças entre modelos usam bootstrap por paciente.
O conjunto eICU é avaliado com modelos congelados, sem novo treinamento.

## Arquivos usados

| Arquivo | Função |
|---|---|
| `protocolo_24h_mimic_eicu.ipynb` | Notebook principal de resultados e comparações |
| `extrair_fontes_ampliadas.py` | Cache temporal de eventos MIMIC/eICU |
| `painel_ampliado.py` | Constrói os novos painéis |
| `criterios_clinicos.py` | AKI, SOFA, ausculta, arritmias, unidades e tempo |
| `auditar_paineis_ampliados.py` | Contratos, cobertura e distribuições |
| `experimentos_24h.py` | Buscas, avaliações, ablações e benchmark |
| `experimento_populacao.py` | Efeito isolado dos filtros de população de Yang (IC principal, primeira UTI, sobreviventes) |
| `painel_completo_24h.py` | Extrator anterior de 54 campos e funções reutilizadas |
| `extrair_painel_24h.py` | Coortes, sinais/exames básicos e painel histórico de 13 campos |
| `avaliar_protocolo_24h.py` | Modelos de referência e funções reutilizadas de avaliação |
| `variaveis_protocolo.py` | Listas históricas de 54 e 13 campos |
| `caminhos_dados.py`, `verificar_dados.py`, `baixar_mimic.py` | Localização, conferência e download |

`input_output_24h.ipynb` e `rede_neural_contra_xgboost.ipynb` são experimentos anteriores,
não etapas necessárias do protocolo ampliado. Arquivos em `tmp/` são auxiliares locais.

Resultados novos: `artefatos_protocolo_24h/experimentos_v3/`.
Resultados históricos preservados: `artefatos_protocolo_24h/completo/` (54 campos) e
`artefatos_protocolo_24h/comum/` (13 campos). O painel completo de 70 campos não é
validado externamente como tal: o modelo externo usa o subconjunto harmonizado de 65.

## Limites que devem constar no TCC

- Previsão principal em 24h, adultos ainda na UTI, mantendo óbitos. LOS inclui saída por morte;
  não equivale a tempo até alta viva. Não há análise de riscos concorrentes.
- AKI é somente pelo critério de creatinina. SOFA é adaptado, com contagem de componentes
  observados. Ausência de registro de tratamento não prova ausência clínica.
- FEVE não tem horário de liberação do laudo. Valvulopatia histórica, ritmos e tratamentos
  têm mecanismos de registro diferentes entre as bases; harmonização não elimina essa diferença.
- Ausculta e alguns exames são esparsos no eICU. Nenhuma coluna inteira ausente é inventada
  por imputação. BNP não substitui NT-proBNP e troponina I não substitui T.
- O teste interno já foi consultado no desenvolvimento: as novas comparações são exploratórias.
  SHAP não prova causalidade, DCA não prova benefício clínico prospectivo e AUC semelhante
  entre subgrupos não comprova equidade.
- O benchmark Yang usa informações retrospectivas e até 48h. Não deve ser confundido com
  previsão prospectiva em 24h ou reprodução exata do artigo.
- AUC 0,922 do artigo corresponde à Random Forest no treino. O XGBoost publicado teve
  AUC 0,861 interna e 0,815 externa, em população e desenho diferentes.
- Os filtros de população de Yang, isolados com os mesmos preditores, janela de 24h e
  hiperparâmetros, não aproximaram a AUC do artigo (`experimentos_v3/populacao/`). O
  subgrupo de IC principal tem poucos pacientes (319 estadias no teste interno), então os intervalos são largos.

## Testes e fontes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes sintéticos verificam fronteiras temporais, unidades, escores, negações,
separação por paciente e ausência de ajuste externo. Não são validação clínica.

[Yang et al. (2025)](https://doi.org/10.1093/eurjcn/zvaf075), PDF fornecido em
`D:/Download/artigo2.pdf`; propostas da apresentação em `D:/Download/artigo02.pptx.pdf`.
As fontes dos mapeamentos estão na documentação do protocolo ampliado.
