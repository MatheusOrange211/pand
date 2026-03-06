"""
data_tool.py — PAND v2.0
Consulta a telemetria do painel no Digital Twin (solar_data.xlsx).

Melhorias v2.0:
  - Retorna os dados com INTERPRETAÇÃO já incluída (contexto para o LLM).
  - Classifica a eficiência em faixas nomeadas para guiar o raciocínio do agente.
  - Trata erros de forma informativa.
"""

import pandas as pd
import json
import os


def consultar_historico_painel(id_painel: str) -> str:
    """
    Busca os dados de telemetria do painel no Digital Twin e retorna
    um JSON enriquecido com interpretação das métricas.

    Retorna (JSON string):
      - id_painel
      - corrente_a
      - tensao_v
      - eficiencia (valor bruto 0.0–1.0)
      - eficiencia_pct (em %, ex: 87.3)
      - faixa_eficiencia: "CRITICA" | "BAIXA" | "CINZA" | "NORMAL" | "OTIMA"
      - status_inversor
      - interpretacao: texto curto para orientar o raciocínio do agente
      - erro (só presente em caso de falha)
    """
    caminho = os.path.join("data", "solar_data.xlsx")

    try:
        df = pd.read_excel(caminho)

        painel = df[df["id_painel"] == id_painel]
        if painel.empty:
            return json.dumps(
                {
                    "erro": f"Painel '{id_painel}' não encontrado na base de dados.",
                    "acao_sugerida": "Verificar se o ID está correto ou se o painel foi cadastrado.",
                }
            )

        row = painel.iloc[0]
        efic = float(row["eficiencia"])
        efic_pct = round(efic * 100, 1)

        # Classificação de eficiência em faixas de engenharia
        if efic < 0.50:
            faixa = "CRITICA"
            interpretacao = (
                f"Eficiência de {efic_pct}% — perda severa de geração. "
                "Verifique defeito elétrico, string aberta ou inversor em falha."
            )
        elif efic < 0.70:
            faixa = "BAIXA"
            interpretacao = (
                f"Eficiência de {efic_pct}% — abaixo do limiar mínimo aceitável (70%). "
                "Intervenção necessária, provável causa elétrica ou sujeira severa."
            )
        elif efic < 0.88:
            faixa = "CINZA"
            interpretacao = (
                f"Eficiência de {efic_pct}% — zona de ambiguidade (70–88%). "
                "Correlacionar com dados visuais. Monitorar antes de autorizar limpeza."
            )
        elif efic < 0.93:
            faixa = "NORMAL"
            interpretacao = (
                f"Eficiência de {efic_pct}% — operação normal. "
                "Limpeza NÃO justificada economicamente, salvo detecção visual severa."
            )
        else:
            faixa = "OTIMA"
            interpretacao = (
                f"Eficiência de {efic_pct}% — geração ótima. "
                "Qualquer detecção visual deve ser tratada como FALSO POSITIVO do sensor."
            )

        dados = {
            "id_painel": id_painel,
            "corrente_a": round(float(row["corrente_a"]), 3),
            "tensao_v": round(float(row["tensao_v"]), 3),
            "eficiencia": round(efic, 4),
            "eficiencia_pct": efic_pct,
            "faixa_eficiencia": faixa,
            "status_inversor": str(row["status_inversor"]),
            "interpretacao": interpretacao,
        }

        return json.dumps(dados, ensure_ascii=False)

    except FileNotFoundError:
        return json.dumps(
            {
                "erro": f"Arquivo '{caminho}' não encontrado.",
                "acao_sugerida": "Rodar etapa_2_gerar_dados_teste_telemetria.py primeiro.",
            }
        )
    except Exception as e:
        return json.dumps({"erro_leitura": str(e)})
