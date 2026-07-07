"""
modelo_prophet.py

Modelagem preditiva da demanda de devolução (DV) usando Prophet.

Prophet foi criado pelo Facebook/Meta especificamente para séries de
negócio com sazonalidade forte -- exatamente o nosso cenário. Ele
detecta tendência e sazonalidade automaticamente via séries de Fourier,
exigindo pouca configuração manual.

Fluxo:
1. Carregar os dados e preparar no formato que o Prophet exige (colunas
   'ds' para data e 'y' para o valor a prever).
2. Separar um período de teste (últimos 30 dias) para validar o modelo
   -- isso simula prever "o futuro" com dados que o modelo nunca viu.
3. Treinar o modelo só com o período de treino.
4. Gerar previsão para o período de teste + mais alguns dias à frente.
5. Comparar previsto vs. real no período de teste (MAE e MAPE).
6. Visualizar.
"""

import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. CARREGAR E PREPARAR OS DADOS
# ---------------------------------------------------------------------------

df = pd.read_csv("dados_devolucao_dv.csv", parse_dates=["data"])

# Prophet exige exatamente essas duas colunas, com esses nomes.
df_prophet = df[["data", "demanda"]].rename(columns={"data": "ds", "demanda": "y"})

# ---------------------------------------------------------------------------
# 2. SEPARAR TREINO E TESTE
# ---------------------------------------------------------------------------
# Separamos os últimos 30 dias como "teste" -- o modelo nunca vê esses
# valores durante o treino. Depois comparamos a previsão dele com o que
# realmente aconteceu nesses dias, pra medir se o modelo é bom de verdade
# (e não só "decorou" os dados de treino).

DIAS_TESTE = 30

treino = df_prophet.iloc[:-DIAS_TESTE].copy()
teste = df_prophet.iloc[-DIAS_TESTE:].copy()

print(f"Treino: {len(treino)} dias ({treino['ds'].min().date()} até {treino['ds'].max().date()})")
print(f"Teste:  {len(teste)} dias ({teste['ds'].min().date()} até {teste['ds'].max().date()})")

# ---------------------------------------------------------------------------
# 3. TREINAR O MODELO
# ---------------------------------------------------------------------------
# yearly_seasonality=True: deixa o Prophet tentar capturar o padrão anual
#   (as 4 gaussianas que a gente montou na mão) -- mesmo com 18 meses de
#   dado, o Prophet consegue estimar isso melhor que o seasonal_decompose,
#   porque ele usa Fourier, não médias móveis (não exige 2 ciclos completos).
# weekly_seasonality=True: captura o padrão de segunda-pico/domingo-vale.
# changepoint_prior_scale: controla o quão "flexível" a tendência pode
#   ser. Valor padrão (0.05) costuma funcionar bem; deixamos explícito
#   aqui pra você saber que existe esse parâmetro pra ajustar depois.

modelo = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
)

modelo.fit(treino)

# ---------------------------------------------------------------------------
# 4. GERAR PREVISÃO
# ---------------------------------------------------------------------------
# make_future_dataframe cria um "esqueleto" de datas futuras. Pedimos
# DIAS_TESTE (pra comparar com o real) + 14 dias extras além do fim dos
# dados (a previsão "de verdade", pro futuro nunca visto por ninguém).

HORIZONTE_EXTRA = 14
futuro = modelo.make_future_dataframe(periods=DIAS_TESTE + HORIZONTE_EXTRA)
previsao = modelo.predict(futuro)

# ---------------------------------------------------------------------------
# 5. AVALIAR NO PERÍODO DE TESTE
# ---------------------------------------------------------------------------

previsao_teste = previsao.set_index("ds").loc[teste["ds"]]
real_teste = teste.set_index("ds")["y"]

mae = np.mean(np.abs(previsao_teste["yhat"] - real_teste))
mape = np.mean(np.abs((previsao_teste["yhat"] - real_teste) / real_teste)) * 100

print(f"\nAvaliação no período de teste ({DIAS_TESTE} dias):")
print(f"MAE  (erro médio absoluto):        {mae:.0f} peças/dia")
print(f"MAPE (erro percentual médio):      {mape:.2f}%")

# ---------------------------------------------------------------------------
# 6. VISUALIZAÇÃO
# ---------------------------------------------------------------------------

fig1 = modelo.plot(previsao)
plt.title("Previsão Prophet -- demanda de devolução (DV)")
plt.xlabel("Data")
plt.ylabel("Peças/dia")
plt.tight_layout()
plt.savefig("previsao_prophet.png", dpi=130)
plt.close(fig1)
print("\nSalvo: previsao_prophet.png")

# Gráfico dos componentes -- mostra separadamente o que o Prophet
# "aprendeu" de tendência, sazonalidade semanal e sazonalidade anual.
# Compare este gráfico com o efeito_semana e as 4 gaussianas que
# definimos manualmente -- deve bater visualmente.
fig2 = modelo.plot_components(previsao)
fig2.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig("componentes_prophet.png", dpi=130)
plt.close(fig2)
print("Salvo: componentes_prophet.png")

# Zoom: previsto vs. real, só no período de teste -- a validação mais
# importante do modelo.
fig3, ax = plt.subplots(figsize=(12, 5))
ax.plot(real_teste.index, real_teste.values, marker="o", label="Real", color="black")
ax.plot(previsao_teste.index, previsao_teste["yhat"], marker="o", label="Previsto (Prophet)", color="#2c7fb8")
ax.fill_between(
    previsao_teste.index,
    previsao_teste["yhat_lower"],
    previsao_teste["yhat_upper"],
    alpha=0.2,
    color="#2c7fb8",
    label="Intervalo de confiança",
)
ax.set_title(f"Previsto vs. Real -- últimos {DIAS_TESTE} dias (MAPE: {mape:.1f}%)")
ax.set_ylabel("Peças/dia")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("previsto_vs_real.png", dpi=130)
plt.close(fig3)
print("Salvo: previsto_vs_real.png")