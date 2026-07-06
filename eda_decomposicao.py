"""
eda_decomposicao.py

Análise exploratória e decomposição da série de demanda de devolução
(DV) gerada em gerar_dados_meli.py.

Objetivo: usar ferramentas prontas (statsmodels) pra separar
automaticamente tendência, sazonalidade e resíduo, e comparar com o
modelo que construímos manualmente na etapa anterior.
"""

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------------------------------------------------------
# 1. CARREGAR OS DADOS
# ---------------------------------------------------------------------------

df = pd.read_csv("dados_devolucao_dv.csv", parse_dates=["data"])
df = df.set_index("data")

# seasonal_decompose exige frequência explícita no índice de datas
df = df.asfreq("D")

serie = df["demanda"]

print(f"Período coberto: {serie.index.min().date()} até {serie.index.max().date()}")
print(f"Total de dias: {len(serie)}")

# ---------------------------------------------------------------------------
# 2. DECOMPOSIÇÃO SEMANAL (period=7)
# ---------------------------------------------------------------------------
# Aqui a gente tem dado de sobra (546 dias / 7 = 78 ciclos completos),
# então essa decomposição deve funcionar bem e confirmar visualmente o
# efeito_semana que a gente definiu na mão.

decomp_semanal = seasonal_decompose(serie, model="additive", period=7)

fig1 = decomp_semanal.plot()
fig1.set_size_inches(12, 8)
fig1.suptitle("Decomposição semanal (period=7)", y=1.02)
plt.tight_layout()
plt.savefig("decomposicao_semanal.png", dpi=130)
plt.close(fig1)
print("\nSalvo: decomposicao_semanal.png")

# ---------------------------------------------------------------------------
# 3. DECOMPOSIÇÃO ANUAL (period=365) -- tentativa
# ---------------------------------------------------------------------------
# Com apenas 18 meses de dado, não temos os 2 ciclos completos (24 meses)
# que a seasonal_decompose geralmente pede pra estimar sazonalidade anual
# com confiança. Vamos tentar mesmo assim e ver o resultado -- é um
# aprendizado válido sobre as limitações do método com dado curto.

try:
    decomp_anual = seasonal_decompose(serie, model="additive", period=365)
    fig2 = decomp_anual.plot()
    fig2.set_size_inches(12, 8)
    fig2.suptitle("Decomposição anual (period=365) -- dado limitado a 18 meses", y=1.02)
    plt.tight_layout()
    plt.savefig("decomposicao_anual.png", dpi=130)
    plt.close(fig2)
    print("Salvo: decomposicao_anual.png (atenção: interpretar com cautela, dado < 24 meses)")
except Exception as e:
    print(f"\nDecomposição anual falhou ou é pouco confiável: {e}")

# ---------------------------------------------------------------------------
# 4. ACF / PACF -- identificar "memória" da série
# ---------------------------------------------------------------------------
# ACF (autocorrelação): mostra o quanto o valor de hoje se parece com o
# valor de N dias atrás. Picos em múltiplos de 7 confirmam o ciclo
# semanal. Um pico (mesmo que fraco) perto de 365 sugeriria eco anual,
# mas com 546 dias não dá pra enxergar um lag de 365 com confiança
# (fica muito perto da borda dos dados disponíveis).

fig3, axes = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(serie, lags=60, ax=axes[0])
axes[0].set_title("ACF -- Autocorrelação (até 60 dias de defasagem)")
plot_pacf(serie, lags=60, ax=axes[1])
axes[1].set_title("PACF -- Autocorrelação Parcial (até 60 dias de defasagem)")
plt.tight_layout()
plt.savefig("acf_pacf.png", dpi=130)
plt.close(fig3)
print("Salvo: acf_pacf.png")

# ---------------------------------------------------------------------------
# 5. RESUMO NUMÉRICO DA DECOMPOSIÇÃO SEMANAL
# ---------------------------------------------------------------------------

print("\nAmplitude da sazonalidade semanal estimada pela decomposição:")
print(decomp_semanal.seasonal.groupby(decomp_semanal.seasonal.index.day_name()).mean().round(1))