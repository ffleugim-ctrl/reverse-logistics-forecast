# Reverse Logistics Forecast

Projeto de previsão de demanda (série temporal) aplicado ao contexto de
logística reversa (devolução ao vendedor) de um centro de distribuição.

## Contexto do problema

Diferente da sazonalidade "óbvia" de picos de venda (Black Friday, Natal,
Dia das Mães), o volume de **devoluções** em um DV (Devolução ao Vendedor)
segue essas mesmas datas comemorativas, mas com um **atraso** — o pico de
devolução do Natal, por exemplo, acontece em janeiro, não em dezembro,
por conta do prazo de entrega/troca (ETD) e do tempo de decisão do
comprador.

Este projeto simula essa dinâmica com um modelo aditivo:

```
demanda(dia) = base + tendência(dia) + efeito_semana(dia) + efeito_sazonal_anual(dia) + ruído(dia)
```

Onde `efeito_sazonal_anual` é a soma de 4 curvas gaussianas defasadas,
uma para cada evento sazonal (Dia das Mães, Dia das Crianças, Natal e
Copa do Mundo 2026), cada uma com seu próprio atraso e intensidade,
calibradas a partir de vivência real de operação — não de suposições
genéricas de mercado.

## Estrutura do projeto

| Arquivo | Etapa | Descrição |
|---|---|---|
| `gerar_dados_meli.py` | 1 | Gera a série temporal sintética (18 meses) com o modelo aditivo |
| `grafico.py` | 1 | Visualização geral da série gerada |
| `dados_devolucao_dv.csv` | 1 | Dataset gerado |
| `eda_decomposicao.py` | 2 | Decomposição estatística (statsmodels) + ACF/PACF |
| `modelo_prophet.py` | 3 | Modelagem preditiva com Prophet + avaliação |

## Resultados

### Série gerada (Etapa 1)

![Visão geral da demanda simulada](demanda_dv_visao_geral.png)

Picos de Natal→janeiro claramente visíveis e crescendo de 2025 para
2026 (efeito de tendência), além do padrão semanal de segunda-feira
como pico e domingo como vale.

### Decomposição e validação estatística (Etapa 2)

![Decomposição semanal](decomposicao_semanal.png)

A decomposição automática (sem qualquer conhecimento prévio da fórmula
usada para gerar os dados) confirmou o padrão semanal calibrado
manualmente — validação estatística independente do modelo de negócio.

![ACF e PACF](acf_pacf.png)

**Decomposição estatística (Etapa 2):** a decomposição semanal via
`seasonal_decompose` confirmou, de forma independente, o padrão semanal
calibrado manualmente a partir da vivência de operação (diferença menor
que 5% em todos os dias da semana). A decomposição anual, por outro
lado, não pôde ser calculada com confiança — o método exige pelo menos
24 meses de histórico (2 ciclos completos), e o projeto conta com 18.
Essa é uma limitação real de projetos de forecasting com histórico
curto, não só uma limitação deste projeto específico.

**Modelagem preditiva (Etapa 3):** o modelo Prophet, treinado com os
primeiros 516 dias e testado nos últimos 30, obteve:

- **MAE:** ~2.212 peças/dia
- **MAPE:** ~13,0%

![Previsão Prophet](previsao_prophet.png)

![Componentes identificados pelo Prophet](componentes_prophet.png)

O Prophet identificou sozinho, sem qualquer informação prévia sobre a
fórmula usada na geração dos dados, os mesmos 4 picos sazonais
definidos manualmente (Dia das Mães, Copa, Dia das Crianças e
Natal/janeiro) — reforçando a validade do modelo de negócio construído
na Etapa 1.

![Previsto vs. Real nos últimos 30 dias](previsto_vs_real.png)

O período de teste coincidiu com o pico de devolução da Copa do Mundo
2026 — um evento não recorrente (ocorre a cada 4 anos), o que
naturalmente eleva o erro em comparação a sazonalidades anuais
recorrentes como Natal e Dia das Mães, que o modelo já tinha visto
repetir uma vez no histórico de treino.

## Principais aprendizados

- Conhecimento de domínio operacional (por que a devolução tem pico
  defasado, por que segunda-feira concentra volume, por que a Copa gera
  um padrão de devolução diferente do Natal) foi determinante para
  calibrar um modelo sintético realista — muito mais do que aplicar
  sazonalidade genérica de mercado.
- Ferramentas de decomposição estatística têm pré-requisitos de volume
  de dado que nem sempre são óbvios (ex: 2 ciclos completos para
  sazonalidade anual) — importante mapear essa limitação antes de
  prometer uma análise que o dado disponível não sustenta.
- Validar um modelo com dados que ele nunca viu (split treino/teste) é
  o que diferencia "o modelo decorou o passado" de "o modelo generaliza
  para o futuro".

## Próximos passos (não implementados nesta versão)

- Comparar o Prophet com um modelo SARIMA, aproveitando os parâmetros
  de autocorrelação já identificados no ACF/PACF da Etapa 2.
- Estender o histórico simulado para 24+ meses e reavaliar a
  decomposição anual.
- Incorporar variáveis externas (ex: campanhas promocionais, greves)
  como regressores adicionais no Prophet.

## Autor

Miguel — projeto de portfólio em Data Analytics, aplicado ao contexto de
operações logísticas (DV — Devolução ao Vendedor, Mercado Livre,
Cajamar/SP).
