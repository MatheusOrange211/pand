"""
analisar_resultados.py — PAND v2.0
Gera todos os gráficos de análise de resultados usando o mapa_cenarios.json
como fonte verdade (corrige o bug de embaralhamento da v1.0).

Execute após a Etapa 3:
    python analisar_resultados.py

Gera:
  - figura_TAD_por_cenario.png
  - figura_matriz_confusao.png
  - figura_tempo_resposta.png
  - AVALIACAO_TAD_CORRIGIDA.csv
"""

import pandas as pd
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os


# --- Configurações visuais ---
CORES_CENARIO = {
    "A (Convergência Ruim)": "#5B4A8A",
    "B (Falso Positivo)": "#E07B54",
    "C (Falha Oculta)": "#2A8C7E",
    "D (Controle)": "#3BB273",
    "E (Ambiguidade Extrema)": "#A8C256",
}

GABARITO = {
    "A (Convergência Ruim)": "LIMPEZA_CORRETIVA",
    "B (Falso Positivo)": "FALSO_POSITIVO_IGNORADO",
    "C (Falha Oculta)": "MANUTENCAO_ELETRICA",
    "D (Controle)": "OPERACAO_NORMAL",
    "E (Ambiguidade Extrema)": "INSPECAO_AMBIGUIDADE",
}

CODIGOS_ORDEM = [
    "LIMPEZA_CORRETIVA",
    "FALSO_POSITIVO_IGNORADO",
    "MANUTENCAO_ELETRICA",
    "OPERACAO_NORMAL",
    "INSPECAO_AMBIGUIDADE",
]


def extrair_codigo(texto):
    match = re.search(r'\[CÓDIGO:\s*([^\]]+)\]', str(texto))
    return match.group(1).strip() if match else "ERRO_FORMATACAO"


def carregar_dados_corrigidos(csv_resultado: str, json_mapa: str) -> pd.DataFrame:
    """
    Carrega o CSV de resultados e CORRIGE o cenário de cada painel
    usando o mapa_cenarios.json como fonte verdade.
    """
    with open(json_mapa, "r") as f:
        mapa = json.load(f)

    df = pd.read_csv(csv_resultado, on_bad_lines="skip")

    # Determina o nome da coluna de resultado
    col_resultado = "Resultado_A3" if "Resultado_A3" in df.columns else "Relatorio_Completo"

    df["Decisao_PAND"] = df[col_resultado].apply(extrair_codigo)
    df["Cenario_Correto"] = df["Painel"].map(mapa)
    df["Esperado"] = df["Cenario_Correto"].map(GABARITO)
    df["Acerto"] = df["Decisao_PAND"] == df["Esperado"]

    # Coluna de tempo
    if "Tempo_s" not in df.columns:
        df["Tempo_s"] = np.nan

    return df


def gerar_figura_tad(df: pd.DataFrame, output_path: str):
    """Gráfico de barras da TAD por cenário."""
    fig, ax = plt.subplots(figsize=(11, 6))

    cenarios = sorted(df["Cenario_Correto"].dropna().unique())
    tads = []
    totais = []
    acertos_list = []
    cores = []

    for c in cenarios:
        grp = df[df["Cenario_Correto"] == c]
        ac = grp["Acerto"].sum()
        tot = len(grp)
        tads.append(ac / tot * 100)
        acertos_list.append(ac)
        totais.append(tot)
        cores.append(CORES_CENARIO.get(c, "#888888"))

    bars = ax.bar(cenarios, tads, color=cores, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, tad, ac, tot in zip(bars, tads, acertos_list, totais):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{tad:.1f}%\n({ac}/{tot})",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Linha de referência TAD global
    tad_global = df["Acerto"].mean() * 100
    ax.axhline(y=tad_global, color="navy", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(
        len(cenarios) - 0.4, tad_global + 1.5,
        f"TAD Global: {tad_global:.1f}%",
        color="navy", fontsize=9, fontweight="bold",
    )

    ax.set_ylim(0, 115)
    ax.set_xlabel("Cenário Experimental", fontsize=12)
    ax.set_ylabel("Taxa de Assertividade Decisória (%)", fontsize=12)
    ax.set_title("Eficácia Decisória por Cenário — PAND v2.0\n(Fonte: mapa_cenarios.json corrigido)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Salvo: {output_path}")


def gerar_figura_confusao(df: pd.DataFrame, output_path: str):
    """Heatmap da matriz de confusão."""
    esperados_presentes = [c for c in CODIGOS_ORDEM if c in df["Esperado"].values]
    decisoes_presentes = [c for c in CODIGOS_ORDEM if c in df["Decisao_PAND"].values]

    matriz = pd.crosstab(
        df["Esperado"],
        df["Decisao_PAND"],
        rownames=["Cenário Operacional (Gabarito)"],
        colnames=["Diagnóstico Gerado pelo Agente PAND"],
    ).reindex(index=esperados_presentes, columns=decisoes_presentes, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(
        matriz, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    # Destaca a diagonal (acertos)
    for i in range(min(len(esperados_presentes), len(decisoes_presentes))):
        col_idx = decisoes_presentes.index(esperados_presentes[i]) if esperados_presentes[i] in decisoes_presentes else -1
        if col_idx >= 0:
            ax.add_patch(plt.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor="limegreen", lw=2.5))

    ax.set_title("Matriz de Confusão — Decisão PAND vs Necessidade Real\n(PAND v2.0)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Salvo: {output_path}")


def gerar_figura_tempo(df: pd.DataFrame, output_path: str):
    """Boxplot de tempo de resposta por cenário."""
    df_valid = df.dropna(subset=["Tempo_s", "Cenario_Correto"])
    if df_valid.empty:
        print("⚠️  Sem dados de tempo para gerar gráfico.")
        return

    cenarios = sorted(df_valid["Cenario_Correto"].unique())
    dados_tempo = [df_valid[df_valid["Cenario_Correto"] == c]["Tempo_s"].values for c in cenarios]
    cores = [CORES_CENARIO.get(c, "#888888") for c in cenarios]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(
        dados_tempo, labels=cenarios, patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 5, "alpha": 0.6},
    )
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.75)

    ax.set_xlabel("Cenário Experimental", fontsize=12)
    ax.set_ylabel("Tempo de Processamento (s)", fontsize=12)
    ax.set_title("Distribuição do Tempo de Resposta por Cenário — PAND v2.0", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Salvo: {output_path}")


def gerar_relatorio_texto(df: pd.DataFrame):
    """Imprime o resumo de TAD no terminal."""
    print("\n" + "=" * 65)
    print("📊 RELATÓRIO DE AVALIAÇÃO — PAND v2.0 (MAPA CORRIGIDO)")
    print("=" * 65)

    for cenario in sorted(df["Cenario_Correto"].dropna().unique()):
        grp = df[df["Cenario_Correto"] == cenario]
        ac = grp["Acerto"].sum()
        tot = len(grp)
        esperado = GABARITO.get(cenario, "?")
        dist = grp["Decisao_PAND"].value_counts().to_dict()
        print(f"\n  [{cenario}]")
        print(f"    Esperado: {esperado}")
        print(f"    TAD: {ac}/{tot} = {ac/tot*100:.1f}%")
        print(f"    Distribuição de decisões: {dist}")

    tad_global = df["Acerto"].mean() * 100
    print(f"\n{'=' * 65}")
    print(f"  🏆 TAD GLOBAL: {df['Acerto'].sum()}/{len(df)} = {tad_global:.1f}%")
    print(f"{'=' * 65}\n")


def main():
    csv_resultado = "RESULTADO_FINAL_TCC.csv"
    json_mapa = "data/mapa_cenarios.json"
    pasta_saida = "data/figuras"
    os.makedirs(pasta_saida, exist_ok=True)

    if not os.path.exists(csv_resultado):
        print(f"❌ Arquivo '{csv_resultado}' não encontrado. Rode a Etapa 3 primeiro.")
        return
    if not os.path.exists(json_mapa):
        print(f"❌ Arquivo '{json_mapa}' não encontrado.")
        return

    print("📂 Carregando dados com mapa corrigido...")
    df = carregar_dados_corrigidos(csv_resultado, json_mapa)
    print(f"   {len(df)} registros carregados.\n")

    # Salva CSV corrigido
    df[["Painel", "Cenario_Correto", "Esperado", "Decisao_PAND", "Acerto", "Tempo_s"]].to_csv(
        "AVALIACAO_TAD_CORRIGIDA.csv", index=False
    )
    print("✅ CSV de avaliação corrigido salvo: AVALIACAO_TAD_CORRIGIDA.csv")

    # Gera figuras
    gerar_figura_tad(df, os.path.join(pasta_saida, "figura_TAD_por_cenario.png"))
    gerar_figura_confusao(df, os.path.join(pasta_saida, "figura_matriz_confusao.png"))
    gerar_figura_tempo(df, os.path.join(pasta_saida, "figura_tempo_resposta.png"))

    # Relatório no terminal
    gerar_relatorio_texto(df)


if __name__ == "__main__":
    main()
