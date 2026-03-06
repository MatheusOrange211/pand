"""
etapa_3_agentes.py — PAND
Executa a bateria de testes com suporte a RETOMADA AUTOMÁTICA.

Novidades:
  - Se o CSV de saída já existir, retoma de onde parou (pula painéis já processados).
  - Trata rate limit (erro 429) com espera progressiva antes de parar.
  - Nunca apaga dados já coletados.
  - Ao final (ou ao retomar após pausa), exibe TAD parcial dos dados existentes.

Fluxo de uso:
  1. Execute normalmente: python etapa_3_agentes.py
  2. Se a API esgotar, o script salva e encerra com mensagem clara.
  3. Execute novamente: o script detecta o CSV existente e continua do ponto certo.
"""

import pandas as pd
import time
import re
import os
import sys
import json
from unittest.mock import patch
from pand_system.agent import run_agent_analysis


# --- GABARITO ---
GABARITO = {
    "A (Convergência Ruim)":  "LIMPEZA_CORRETIVA",
    "B (Falso Positivo)":     "FALSO_POSITIVO_IGNORADO",
    "C (Falha Oculta)":       "MANUTENCAO_ELETRICA",
    "D (Controle)":           "OPERACAO_NORMAL",
    "E (Ambiguidade Extrema)":"INSPECAO_AMBIGUIDADE",
}

ARQUIVO_SAIDA     = "RESULTADO_FINAL_TCC.csv"
ARQUIVO_AVALIACAO = "AVALIACAO_TAD.csv"
ARQUIVO_YOLO      = "data/resultados_yolo_reais.csv"
ARQUIVO_MAPA      = "data/mapa_cenarios.json"

# Tempo de espera (segundos) quando detecta rate limit antes de encerrar
ESPERA_RATE_LIMIT = 15


def extrair_codigo(texto: str) -> str:
    match = re.search(r'\[CÓDIGO:\s*([^\]]+)\]', str(texto))
    return match.group(1).strip() if match else "ERRO_FORMATACAO"


def carregar_ja_processados() -> set:
    """Retorna o conjunto de IDs de painéis já presentes no CSV de saída."""
    if not os.path.exists(ARQUIVO_SAIDA):
        return set()
    try:
        df = pd.read_csv(ARQUIVO_SAIDA, on_bad_lines="skip")
        if "Painel" in df.columns:
            ids = set(df["Painel"].dropna().tolist())
            print(f"♻️  Retomando: {len(ids)} painel(is) já processado(s) encontrado(s).")
            return ids
    except Exception:
        pass
    return set()


def exibir_tad_atual():
    """Lê o CSV existente e exibe a TAD parcial."""
    if not os.path.exists(ARQUIVO_SAIDA):
        return
    try:
        df = pd.read_csv(ARQUIVO_SAIDA, on_bad_lines="skip")
        if "Decisao_PAND" not in df.columns or "Esperado" not in df.columns:
            return
        df["Acerto"] = df["Decisao_PAND"] == df["Esperado"]
        total = len(df)
        acertos = df["Acerto"].sum()

        print(f"\n{'='*60}")
        print(f"📊 TAD PARCIAL — {total} painel(is) processado(s)")
        print(f"{'='*60}")
        for cenario, grp in df.groupby("Cenario_Injetado"):
            ac = grp["Acerto"].sum()
            tot = len(grp)
            print(f"  {cenario}: {ac}/{tot} = {ac/tot*100:.1f}%")
        print(f"  {'─'*40}")
        print(f"  GLOBAL: {acertos}/{total} = {acertos/total*100:.1f}%")
        print(f"{'='*60}")

        # Salva avaliação parcial
        linhas = []
        for cenario, grp in df.groupby("Cenario_Injetado"):
            ac = grp["Acerto"].sum(); tot = len(grp)
            linhas.append({"Cenario": cenario, "Acertos": int(ac),
                           "Total": tot, "TAD_pct": round(ac/tot*100, 1)})
        ac_g = df["Acerto"].sum(); tot_g = len(df)
        linhas.append({"Cenario": "GLOBAL", "Acertos": int(ac_g),
                       "Total": tot_g, "TAD_pct": round(ac_g/tot_g*100, 1)})
        pd.DataFrame(linhas).to_csv(ARQUIVO_AVALIACAO, index=False)
        print(f"✅ Avaliação salva em: {ARQUIVO_AVALIACAO}")

    except Exception as e:
        print(f"⚠️  Não foi possível calcular TAD parcial: {e}")


def is_rate_limit(erro: Exception) -> bool:
    msg = str(erro).lower()
    return "rate_limit" in msg or "429" in msg or "too many" in msg or "quota" in msg


def rodar_bateria_agentes():
    # --- Verificações de pré-condição ---
    for arq in [ARQUIVO_YOLO, ARQUIVO_MAPA]:
        if not os.path.exists(arq):
            print(f"❌ Arquivo '{arq}' não encontrado. Rode as etapas anteriores primeiro.")
            return

    # --- Carrega dados de entrada ---
    df_yolo = pd.read_csv(ARQUIVO_YOLO)
    dicionario_yolo = dict(zip(df_yolo["Painel"], df_yolo["YOLO_Output"]))

    with open(ARQUIVO_MAPA, "r", encoding="utf-8") as f:
        mapa_cenarios = json.load(f)

    # --- Detecta painéis já processados (retomada) ---
    ja_processados = carregar_ja_processados()
    pendentes = {p: c for p, c in mapa_cenarios.items() if p not in ja_processados}

    print(f"\n🤖 PAND v2.1 — ORQUESTRAÇÃO")
    print(f"   Total no mapa:      {len(mapa_cenarios)}")
    print(f"   Já processados:     {len(ja_processados)}")
    print(f"   Faltam processar:   {len(pendentes)}")

    if not pendentes:
        print("\n✅ Todos os painéis já foram processados!")
        exibir_tad_atual()
        return

    dist = pd.Series(pendentes.values()).value_counts().to_dict()
    print(f"   Pendentes por cenário: {dist}\n")

    erros_consecutivos = 0
    MAX_ERROS_CONSECUTIVOS = 3

    with patch("pand_system.agent.analisar_imagem_painel") as mock_vision:

        for painel_id, cenario in pendentes.items():

            if painel_id not in dicionario_yolo:
                print(f"⚠️  {painel_id}: sem dados YOLO, pulando.")
                continue

            mock_vision.return_value = dicionario_yolo[painel_id]
            esperado = GABARITO.get(cenario, "DESCONHECIDO")
            print(f"  [{cenario}] {painel_id} (esperado: {esperado})... ", end="")
            sys.stdout.flush()

            try:
                t0 = time.time()
                resposta = run_agent_analysis(b"", painel_id)
                tempo = round(time.time() - t0, 2)
                decisao = extrair_codigo(resposta)
                acerto = "✅" if decisao == esperado else "❌"
                print(f"{acerto} {decisao} ({tempo}s)")

                registro = {
                    "Painel":             painel_id,
                    "Cenario_Injetado":   cenario,
                    "Esperado":           esperado,
                    "Decisao_PAND":       decisao,
                    "Acerto":             decisao == esperado,
                    "Relatorio_Completo": resposta,
                    "Tempo_s":            tempo,
                }

                # Salva incrementalmente — NUNCA apaga o que já existe
                pd.DataFrame([registro]).to_csv(
                    ARQUIVO_SAIDA, mode="a", index=False,
                    header=not os.path.exists(ARQUIVO_SAIDA)
                )

                erros_consecutivos = 0
                time.sleep(3)

            except Exception as e:
                print(f"❌ ERRO")

                if is_rate_limit(e):
                    print(f"\n⏳ Rate limit da API atingido.")
                    print(f"   Aguardando {ESPERA_RATE_LIMIT}s antes de encerrar...")
                    time.sleep(ESPERA_RATE_LIMIT)
                    print(f"   💾 Progresso salvo em: {ARQUIVO_SAIDA}")
                    print(f"   ▶️  Execute o script novamente para continuar de onde parou.")
                    break
                else:
                    erros_consecutivos += 1
                    print(f"   Erro {erros_consecutivos}/{MAX_ERROS_CONSECUTIVOS}: {e}")
                    if erros_consecutivos >= MAX_ERROS_CONSECUTIVOS:
                        print(f"\n🛑 {MAX_ERROS_CONSECUTIVOS} erros consecutivos. Verifique suas chaves.")
                        break
                    time.sleep(5)
                    continue

    exibir_tad_atual()


if __name__ == "__main__":
    rodar_bateria_agentes()