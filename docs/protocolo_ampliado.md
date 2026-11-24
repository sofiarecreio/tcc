# Ampliação do protocolo de 24 horas

O protocolo acrescenta 16 campos às 54 variáveis anteriores. Os sete conceitos
solicitados foram desdobrados em medidas e indicadores: pressão sistólica, AKI por
creatinina, SOFA adaptado, arritmia, ausculta pulmonar, valvulopatia prévia e
registro de cateterismo cardíaco. `experimentos_24h.TODAS` é a lista executável.

**Protocolo adotado (v4):** às 70 variáveis (v3) somam-se 18 de contexto conhecidas
até 24h, totalizando 88 no MIMIC e 83 no painel harmonizado. Resultados em
`artefatos_protocolo_24h/experimentos_v4/`; o v3 é preservado para comparação.

## População e tempo

O painel principal contém adultos com IC identificada retrospectivamente e LOS
de UTI >24h. Mantém óbitos e múltiplas estadias. O instante de previsão é 24h;
eventos e registros disponibilizados a partir desse corte são excluídos.
Identificadores, LOS, óbito, hospital, raça e diagnósticos de alta não entram em X.
A identificação retrospectiva da própria coorte é uma limitação para uso prospectivo.

Preserva-se a divisão por paciente do experimento anterior. Como o teste interno
já foi consultado durante o desenvolvimento, os novos resultados são exploratórios.
Não há seleção de modelo, treinamento, imputação ajustada, escolha de limiar ou
recalibração usando desfechos do eICU.

## Definições das novas variáveis

| Conceito | MIMIC-IV | eICU | Definição e limites |
|---|---|---|---|
| SBP/PAS | Pressões sistólicas em `chartevents` | Pressões sistólicas invasivas e não invasivas | Média dos valores válidos até 24h, em mmHg |
| Arritmia | Ritmo cardíaco, item 220048 | Diagnósticos registrados até 24h | Definição ampla inspirada em Yang: inclui FA, arritmias ventriculares e bloqueio AV completo. Não é sinônimo clínico de toda arritmia maligna |
| Arritmia ventricular | Mesmo registro de ritmo | Diagnósticos precoces | Indicador separado de TV/FV/flutter ventricular; sem duração confiável para provar TV sustentada |
| AKI | Creatinina de laboratório | Creatinina de laboratório | Estágio 0–3 somente pelo critério de creatinina; compara com medidas anteriores em 48h/7 dias. Sem comparação anterior, fica ausente |
| SOFA | Sinais, exames, Glasgow, urina, infusões e ventilação | Mesmas famílias de informação | Escore adaptado de primeiro dia; seis componentes e quantidade observada preservados no painel |
| Breath sounds | Ausculta, itens 223986–223989 | Caminhos de ausculta em `physicalExam` | Roncos, sibilos, estertores, estridor, murmúrio diminuído, qualquer alteração e presença de exame. Negação explícita não vira achado positivo |
| Valvulopatia prévia | Diagnóstico em internação hospitalar encerrada antes da UTI atual | Histórico registrado até 24h em `pastHistory` | Ausência de evidência não prova ausência da doença. História hospitalar e história relatada têm coberturas diferentes |
| Cateterismo cardíaco | `procedureevents`, item 225430 | `treatment`: cateterismo/angiografia cardíaca | Registro até 24h; remove procedimentos reescritos no MIMIC e menções de planejamento no eICU. A tabela de tratamento não é confirmação independente da realização |

## Contexto de 24h (adotado no v4)

Gerado por `contexto_24h.py` em `data/processados/{base}_contexto_24h.csv` e
juntado ao painel por `experimentos_24h.carregar_painel`.

| Grupo | MIMIC-IV | eICU | Definição e limites |
|---|---|---|---|
| Tipo de UTI | `icustays.first_careunit` | `patient.unittype` | Seis indicadores: coronariana (CCU, Cardiac ICU, CCU-CTICU), cardiotorácica (CVICU, CTICU, CSICU), clínica, cirúrgica (SICU, TSICU), mista (MICU/SICU, Med-Surg ICU) e neuro. CCU-CTICU é unidade combinada e foi agrupada como coronariana |
| Origem antes da UTI | Última unidade em `transfers` antes da entrada | `patient.unitadmitsource` | Pronto-socorro, centro cirúrgico/recuperação, enfermaria/step-down, outra UTI e direta/externa. Sem transferência prévia no MIMIC = direta/externa |
| Tempo pré-UTI | Entrada na UTI − `admittime` | −`hospitaladmitoffset` | Horas, mínimo zero |
| Tendências | Laboratório até 24h | Laboratório até 24h | Último valor e variação (último − primeiro) de creatinina, ureia (BUN) e lactato; variação só com duas medidas ou mais |

Antes de adotar, `experimento_contexto.py` comparou as variantes sobre o v3 com
hiperparâmetros fixos e diferenças pareadas (`experimentos_v3/contexto/`). O tipo de UTI
respondeu pela maior parte do ganho; uma nova busca escolheu os mesmos hiperparâmetros.
No eICU, o APACHE IVa (permanência prevista) teve AUC semelhante ao modelo v3 e serve
apenas como referência, pois foi desenvolvido em UTIs semelhantes às do eICU.

O AKI **não implementa KDIGO completo**: não incorpora terapia renal substitutiva
nem janelas móveis de diurese. Creatinina elevada sem subida documentada não é
automaticamente AKI. Doença renal crônica e lesão já presente sem basal anterior
podem não ser reconhecidas por essa definição.

O SOFA usa PaO2 arterial, pareada com a última FiO2 disponível até quatro horas
antes da coleta. Não divide extremos de horários diferentes. Suporte ventilatório
é aproximado por modo/via no MIMIC e por registro de volume corrente no eICU;
este último não distingue perfeitamente suporte invasivo de não invasivo.
Componentes ausentes contribuem zero à soma, acompanhados da contagem observada;
isso não significa que foram medidos e normais. Se todos faltarem, SOFA fica ausente.
Doses de vasopressor não convertíveis não são tratadas como dose zero.

## Eventos de entrada e saída

`inputevents` e `outputevents` permanecem no modelo MIMIC. Infusões são recortadas
ao intervalo; quando cruzam 24h, usa-se a taxa conhecida, sem consultar o total final.
Noradrenalina é expressa em mcg/kg/min, furosemida em mg e volumes em mL.

No eICU, `infusionDrug` fornece registros de infusão; `intakeOutput` fornece volumes
individuais. Não se somam os campos cumulativos `intaketotal`/`outputtotal`, nem se
usa volume do frasco como volume administrado. Contagens de micções e ocorrências
não são mL. Entradas orais, dieta, flush, irrigação e diálise são excluídas do IV.
Indicadores de drogas combinam registros ativos de infusão e volumes IV positivos.
Ordens de `medication` não são convertidas em administrações confirmadas.

Sem volume/urina registrados, o total fica ausente. Sem peso ou diurese, o indicador
de oligúria também fica ausente. Diurese/kg/h divide o volume registrado por peso
e 24h e **não constitui diagnóstico KDIGO**. Registro parcial pode subestimar volumes.
Balanço corresponde a IV menos saídas registradas, não ao balanço fisiológico completo.

## Harmonização e qualidade

`harmonizacao.csv` enumera a inclusão/exclusão de cada campo e sua ausência em cada
base. FEVE, as oito categorias MIMIC de admissão e dose total de furosemida não têm
equivalência temporal/quantitativa suficiente no eICU. NT-proBNP não foi encontrado;
BNP não é substituído por NT-proBNP. Troponina T foi encontrada e não é trocada por I.
Uma coluna inteiramente ausente não é artificialmente criada por imputação externa.

`auditar_paineis_ampliados.py` verifica chaves únicas, domínio de escores, indicadores,
balanço, valores infinitos, cobertura e distribuição. Gera também cobertura por
hospital. Indicadores zero de medicamentos, procedimentos ou antecedentes significam
“sem registro encontrado”; não se pode inferir cobertura clínica completa.

FEVE preserva a definição anterior: medida mais próxima entre −48h e +24h da UTI.
A fonte não informa liberação do laudo, portanto sua disponibilidade em tempo real
não está comprovada. O painel harmonizado não utiliza FEVE.

## Cinco frentes executáveis

1. **Painel externo ampliado:** treinar no MIMIC com as variáveis compatíveis e
   aplicar o mesmo modelo congelado no eICU.
2. **Ajuste de XGBoost e MLP:** busca aleatória, semente 42, CV em três partes por
   paciente. Oito candidatos XGBoost por alvo de classificação; quatro MLP; quatro
   candidatos por regressor. Preparo é ajustado dentro de cada parte do treino.
   O limiar de Youden usa previsões fora da amostra no treino após a busca; essa AUC
   não é uma estimativa não enviesada de todo o processo de seleção.
3. **Revisão da extração:** janela e disponibilidade, unidades, dados ausentes,
   agregações e auditoria por hospital. Médias são usadas nos exames básicos;
   máximos/mínimos seguem os nomes das demais variáveis.
4. **Ablações XGBoost:** completo; sem as 16 novas; sem entradas/saídas e SOFA que
   delas depende; excluir >30% de ausência no treino; indicadores de ausência;
   máximos dos exames básicos no lugar de médias; sem o contexto de 24h (= v3). Usa os hiperparâmetros do completo
   para isolar a mudança de variáveis. Comparações e IC pareados por paciente são
   exploratórios; a escolha de variante usa somente CV do treino.
5. **Yang aproximado separado:** IC principal, primeira UTI da internação,
   sobreviventes hospitalares, incluindo estadias curtas; dez conceitos importantes,
   diagnósticos retrospectivos e tratamentos até 48h. Não representa um modelo
   prospectivo de 24h nem reprodução exata dos 41 candidatos/nove algoritmos/artigo.

“IC principal” foi operacionalizada como `diagnoses_icd.seq_num = 1` no MIMIC e
`diagnosispriority = primary` no eICU. Isso produziu 1.492 estadias MIMIC e 3.744
eICU, diferentes das 5.744/4.056 publicadas. A prioridade hospitalar de faturamento
e a prioridade diagnóstica da UTI não são necessariamente equivalentes; a definição
operacional do artigo não foi reproduzida por completo. Não se ajustou o filtro
para obter artificialmente o tamanho amostral ou a AUC publicados.

Análises de reprodução posteriores (exploratórias, em `experimentos_v3/`): IC nas
posições 1–2 do CID aproxima o tamanho, a prevalência (25%) e a LOS do artigo
(`--ic-max-seq 2`); AKI KDIGO com diurese em 48h (`aki_kdigo.py`) reproduz a
distribuição de estágios; arritmia, ausculta e valvulopatia no formato da Tabela 1
(`preditores_yang.py`) aproximam as prevalências. Mesmo assim o benchmark ficou em
AUC ≈0,72 interna e ≈0,68 externa. Variáveis em 48h usam dados posteriores ao
instante de previsão e não entram no protocolo de 24h.

A MLP compara redes de uma ou duas camadas, regularização e taxa de aprendizado;
máximo de 200 épocas, sem divisão interna por linhas para early stopping.
Os avisos de convergência ficam junto a cada busca. Regressão prevê LOS em dias
com transformação log1p. Há AUC/IC95%, AUC-PR, Brier, calibração, DCA, subgrupos,
sensibilidade por óbito, SHAP do XGBoost e comparação pareada entre MLP/XGBoost.

As execuções usam limite de três threads numéricas. Nesta instalação, repetir o
mesmo XGBoost com limite OpenMP diferente alterou as previsões; as ablações foram
padronizadas e refeitas com o mesmo limite do principal. A semente isolada não
garante identidade entre versões, hardware ou configurações de paralelismo.

## Fontes de definição

- [Yang et al., 2025](https://academic.oup.com/eurjcn/article/24/6/973/8121188), DOI 10.1093/eurjcn/zvaf075; PDF fornecido `D:/Download/artigo2.pdf`.
- [MIMIC: SOFA do primeiro dia](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/firstday/first_day_sofa.sql).
- [MIMIC: creatinina para KDIGO](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/organfailure/kdigo_creatinine.sql).
- [eICU: infusionDrug](https://eicu.mit.edu/eicutables/infusiondrug/), [intakeOutput](https://eicu.mit.edu/eicutables/intakeoutput/), [medication](https://eicu.mit.edu/eicutables/medication/), [diagnosis](https://eicu.mit.edu/eicutables/diagnosis/).

O 0,922 do artigo é AUC da Random Forest no treino. O XGBoost publicado teve
0,861 na validação interna e 0,815 na externa; não são metas garantidas para esta
população, janela, versão de dados e conjunto de medidas. SHAP descreve o modelo
ajustado e não prova causalidade nem que acrescentar a variável elevará a AUC.
