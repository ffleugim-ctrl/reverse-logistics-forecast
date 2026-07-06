"""
gerar_dados.py

Gera uma série temporal sintética representando o volume diário de
devoluções (retiros) recebidas no DV (Devolução ao Vendedor) do CD de
Cajamar, cobrindo 18 meses.

O modelo é ADITIVO, montado a partir de conversas com quem vive a
operação no chão de fábrica:

    demanda(dia) = base
                 + tendencia(dia)
                 + efeito_semana(dia)
                 + efeito_sazonal_anual(dia)   [soma de 4 gaussianas]
                 + ruido(dia)

Cada componente está isolado em sua própria função para que você
consiga estudar, testar e recalibrar cada peça de forma independente.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. CONFIGURAÇÃO GERAL
# ---------------------------------------------------------------------------

# Semente fixa: garante que toda vez que você rodar o script, o "aleatório"
# saia igual. Isso é essencial pra reprodutibilidade (outra pessoa rodando
# seu código chega no mesmo resultado que você).
np.random.seed(42)

DATA_INICIO = "2025-01-01"
DATA_FIM = "2026-06-30"

# Base diária: ancorada no dado real que você trouxe (T1 de 3/jul/2026
# teve 12.648 peças). Assumindo 2 turnos operando num dia comum, a base
# diária fica em torno disso. Ajuste esse número se tiver um valor melhor.
BASE_DIARIA = 12648

# ---------------------------------------------------------------------------
# 1. EIXO TEMPORAL
# ---------------------------------------------------------------------------


def gerar_eixo_temporal(data_inicio: str, data_fim: str) -> pd.DataFrame:
    """Cria um DataFrame com uma linha por dia no intervalo pedido."""
    datas = pd.date_range(start=data_inicio, end=data_fim, freq="D")
    df = pd.DataFrame({"data": datas})
    df["dia_do_ano"] = df["data"].dt.dayofyear
    df["dia_semana"] = df["data"].dt.day_name()
    # dia_sequencial: 0, 1, 2, 3... cresce sem repetir a cada ano novo.
    # É o eixo que vamos usar pra tendência de crescimento (dayofyear
    # sozinho "reseta" todo ano e não serve pra medir crescimento
    # de longo prazo).
    df["dia_sequencial"] = (df["data"] - df["data"].min()).dt.days
    return df


# ---------------------------------------------------------------------------
# 2. TENDÊNCIA (crescimento do volume ao longo do tempo)
# ---------------------------------------------------------------------------


def efeito_tendencia(dia_sequencial: np.ndarray, crescimento_diario: float = 3.5) -> np.ndarray:
    """
    Crescimento linear simples: a cada dia que passa, a operação processa
    um pouquinho mais que no dia anterior (mais mini e-commerces vendendo,
    mais gente devolvendo, qualidade ainda instável de quem tá começando).

    crescimento_diario: quanto a demanda sobe, em média, por dia.
    Um valor pequeno (poucas unidades/dia) já vira um crescimento grande
    ao longo de 18 meses -- ajuste esse número se achar que tá exagerado.
    """
    return dia_sequencial * crescimento_diario


# ---------------------------------------------------------------------------
# 3. EFEITO SEMANAL (segunda pico, domingo vale)
# ---------------------------------------------------------------------------

# Pesos que você definiu com base na vivência real: segunda concentra o
# ETD acumulado do fim de semana, domingo opera com equipe reduzida.
PESOS_SEMANA = {
    "Monday": 950,
    "Tuesday": 500,
    "Wednesday": 150,
    "Thursday": 0,
    "Friday": -150,
    "Saturday": -600,
    "Sunday": -850,
}


def efeito_semana(dias_semana: pd.Series) -> np.ndarray:
    """Aplica o peso correspondente a cada dia da semana."""
    return dias_semana.map(PESOS_SEMANA).values


# ---------------------------------------------------------------------------
# 4. SAZONALIDADE ANUAL (4 gaussianas defasadas)
# ---------------------------------------------------------------------------


def gaussiana(dia_do_ano: np.ndarray, dia_pico: float, amplitude: float, dispersao: float) -> np.ndarray:
    """
    Curva de sino clássica, sem normalização de área (queremos controlar
    a altura do pico manualmente via `amplitude`).

    Trata a "volta do calendário" com módulo: um pico perto do fim do
    ano (ex: dia 359, Natal) também precisa "vazar" pro início do ano
    seguinte -- por isso calculamos a distância circular mínima entre
    cada dia do ano e o dia_pico, ao invés da distância linear direta.
    """
    diferenca = np.abs(dia_do_ano - dia_pico)
    # distância circular: se a diferença linear for maior que meio ano,
    # o caminho "dando a volta" é mais curto -- pega o menor dos dois.
    diferenca_circular = np.minimum(diferenca, 365 - diferenca)
    return amplitude * np.exp(-(diferenca_circular ** 2) / (2 * dispersao ** 2))


# Parâmetros calibrados a partir da conversa (dia_pico = dia do ano em
# que o efeito de DEVOLUÇÃO bate mais forte, já considerando o atraso
# em relação à data de venda no MZ):

PARAMS_SAZONALIDADE = {
    # Dia das Mães: venda ~2ª semana de maio (dia ~130) + atraso ETD.
    "maes": {"dia_pico": 145, "amplitude": BASE_DIARIA * 0.4, "dispersao": 5},
    # Dia das Crianças: venda ~12/out (dia ~285) + atraso ETD.
    "criancas": {"dia_pico": 300, "amplitude": BASE_DIARIA * 0.4, "dispersao": 5},
    # Natal: venda 25/dez (dia ~359) + atraso maior (ETD longo, decisão
    # de devolução mais lenta) -> pico de devolução ~20 dias depois,
    # já caindo em janeiro. Amplitude dobra a base (confirmado por você).
    # dia 359 + 20 = 379 -> módulo 365 -> dia 14 do ano seguinte.
    "natal": {"dia_pico": 14, "amplitude": BASE_DIARIA * 1.0, "dispersao": 7.5},
    # Copa: venda concentrada em junho (dia ~165, meio da Copa 2026) +
    # atraso curto (motivo é erro de tamanho/avaria, percebido rápido).
    # Dispersão menor -> efeito mais concentrado no tempo.
    "copa": {"dia_pico": 175, "amplitude": BASE_DIARIA * 0.6, "dispersao": 3.5},
}


def efeito_sazonal_anual(dia_do_ano: np.ndarray) -> np.ndarray:
    """Soma as 4 gaussianas -- cada uma representa uma data comemorativa."""
    total = np.zeros_like(dia_do_ano, dtype=float)
    for nome, params in PARAMS_SAZONALIDADE.items():
        total += gaussiana(dia_do_ano, **params)
    return total


# ---------------------------------------------------------------------------
# 5. RUÍDO (pequena variação aleatória do dia a dia)
# ---------------------------------------------------------------------------


def efeito_ruido(n_dias: int, intensidade: float = 0.02) -> np.ndarray:
    """
    Você mencionou que o ruído praticamente não afeta a operação -- por
    isso a intensidade aqui é baixa (2% da base), só pra série não ficar
    matematicamente "perfeita demais" e sem graça de se analisar depois.
    """
    return np.random.normal(loc=0, scale=BASE_DIARIA * intensidade, size=n_dias)


# ---------------------------------------------------------------------------
# 6. MONTAGEM FINAL
# ---------------------------------------------------------------------------


def gerar_serie_completa() -> pd.DataFrame:
    df = gerar_eixo_temporal(DATA_INICIO, DATA_FIM)

    df["tendencia"] = efeito_tendencia(df["dia_sequencial"].values)
    df["efeito_semana"] = efeito_semana(df["dia_semana"])
    df["efeito_sazonal"] = efeito_sazonal_anual(df["dia_do_ano"].values)
    df["ruido"] = efeito_ruido(len(df))

    df["demanda"] = (
        BASE_DIARIA
        + df["tendencia"]
        + df["efeito_semana"]
        + df["efeito_sazonal"]
        + df["ruido"]
    )

    # Devolução não pode ser negativa nem fracionária -- arredonda e trava em 0.
    df["demanda"] = df["demanda"].round().clip(lower=0).astype(int)

    return df


if __name__ == "__main__":
    df = gerar_serie_completa()
    df.to_csv("dados_devolucao_dv.csv", index=False)
    print(f"Arquivo gerado com {len(df)} linhas.")
    print(df[["data", "dia_semana", "demanda"]].head(10))
    print("\nResumo estatístico da coluna 'demanda':")
    print(df["demanda"].describe())
