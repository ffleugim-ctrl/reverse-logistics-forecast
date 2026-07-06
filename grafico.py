"""
graficos.py

Visualiza a série de demanda sintética gerada em gerar_dados.py, com
foco em confirmar visualmente se os padrões esperados (tendência,
efeito semanal, picos sazonais) estão presentes.
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("dados_devolucao_dv.csv", parse_dates=["data"])

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# --- Gráfico 1: série completa, 18 meses ---
ax = axes[0]
ax.plot(df["data"], df["demanda"], color="#2c7fb8", linewidth=1)
ax.set_title("Demanda diária de devolução (DV) — série completa (18 meses)")
ax.set_ylabel("Peças/dia")
ax.grid(alpha=0.3)

# Marca visualmente os picos sazonais esperados
eventos = {
    "Natal→Jan": ["2025-01-14", "2026-01-14"],
    "Dia das Mães": ["2025-05-25"],
    "Copa": ["2026-06-25"],
    "Dia das Crianças": ["2025-10-27"],
}
cores = {"Natal→Jan": "red", "Dia das Mães": "green", "Copa": "orange", "Dia das Crianças": "purple"}
for evento, datas in eventos.items():
    for d in datas:
        ax.axvline(pd.Timestamp(d), color=cores[evento], linestyle="--", alpha=0.5)

# Legenda manual para as linhas verticais
from matplotlib.lines import Line2D
linhas_legenda = [Line2D([0], [0], color=c, linestyle="--", label=e) for e, c in cores.items()]
ax.legend(handles=linhas_legenda, loc="upper left")

# --- Gráfico 2: zoom de 6 semanas pra ver o padrão semanal ---
ax2 = axes[1]
trecho = df[(df["data"] >= "2025-07-01") & (df["data"] <= "2025-08-15")]
ax2.plot(trecho["data"], trecho["demanda"], marker="o", color="#2c7fb8")
ax2.set_title("Zoom: padrão semanal (jul-ago/2025, período sem sazonalidade forte)")
ax2.set_ylabel("Peças/dia")
ax2.grid(alpha=0.3)

# Marca as segundas-feiras pra evidenciar o pico semanal
segundas = trecho[trecho["data"].dt.day_name() == "Monday"]
ax2.scatter(segundas["data"], segundas["demanda"], color="red", zorder=5, label="Segundas-feiras (pico)")
ax2.legend()

plt.tight_layout()
plt.savefig("demanda_dv_visao_geral.png", dpi=130)
print("Gráfico salvo em demanda_dv_visao_geral.png")
