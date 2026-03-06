"""
etapa_2_gerar_dados_teste_telemetria.py — PAND
Gera um dataset sintético de telemetria para os painéis solares, baseado nos resultados reais do YOLO da Etapa 1.
"""

import pandas as pd
import os
import random
import json


# SEED FIXO — OBRIGATÓRIO para reprodutibilidade
RANDOM_SEED = 42


def gerar_status_inversor(cenario: str, rng: random.Random) -> str:
    """
    Gera o status do inversor com probabilidade realista por cenário.
    Evita que o status seja 100% determinístico (simplificação excessiva).

    Cenário A (sujeira + queda): inversor em Alerta (80%) ou Normal (20%)
      — sujeira raramente causa falha total do inversor
    Cenário B (falso positivo): sempre Normal — geração está ótima
    Cenário C (falha oculta): Falha (80%) ou Alerta (20%)
      — defeito interno pode não ter atingido falha total ainda
    Cenário D (controle): sempre Normal — sistema saudável
    Cenário E (ambiguidade): Alerta (90%) ou Normal (10%)
      — zona cinzenta, inversor instável mas não em falha
    """
    distribuicoes = {
        "A": [("Alerta", 0.80), ("Normal", 0.20)],
        "B": [("Normal", 1.00)],
        "C": [("Falha", 0.80), ("Alerta", 0.20)],
        "D": [("Normal", 1.00)],
        "E": [("Alerta", 0.90), ("Normal", 0.10)],
    }
    opcoes, pesos = zip(*distribuicoes[cenario])
    return rng.choices(opcoes, weights=pesos, k=1)[0]


def gerar_telemetria_inteligente():
    print("⚙️  ETAPA 2: Gerando Telemetria Sintética com base nos resultados do YOLO...")
    print(f"   Seed aleatória fixada em: {RANDOM_SEED} (reprodutibilidade garantida)")

    # --- Verificação de pré-condição ---
    if not os.path.exists("data/resultados_yolo_reais.csv"):
        print("❌ Erro: 'data/resultados_yolo_reais.csv' não encontrado!")
        print("   Execute a Etapa 1 primeiro.")
        return

    df_yolo = pd.read_csv("data/resultados_yolo_reais.csv")
    rng = random.Random(RANDOM_SEED)

    # --- Classificação visual dos painéis ---
    paineis_com_deteccao = []
    paineis_limpos = []

    for _, row in df_yolo.iterrows():
        pid = row["Painel"]
        out = str(row["YOLO_Output"])
        is_clean = (
            ("clean" in out)
            or ("Non Defective" in out and "Dust" not in out and "Defective" not in out)
        )
        if is_clean:
            paineis_limpos.append(pid)
        else:
            paineis_com_deteccao.append(pid)

    rng.shuffle(paineis_com_deteccao)
    rng.shuffle(paineis_limpos)

    # --- Verificação de suficiência ---
    MIN_DETECCAO = 40  # A(15) + B(25) no mínimo
    MIN_LIMPOS = 20    # C(20) no mínimo
    if len(paineis_com_deteccao) < MIN_DETECCAO:
        print(f"⚠️  AVISO: Apenas {len(paineis_com_deteccao)} painéis com detecção YOLO.")
        print(f"   Mínimo recomendado: {MIN_DETECCAO}. Ajustando cenários A e B.")
        # Reduz B proporcionalmente se necessário
        n_b = max(10, len(paineis_com_deteccao) - 15)
        print(f"   Cenário B ajustado para {n_b} painéis.")
    else:
        n_b = 25  # valor padrão

    if len(paineis_limpos) < MIN_LIMPOS:
        print(f"❌ Erro crítico: apenas {len(paineis_limpos)} painéis limpos.")
        print(f"   Mínimo necessário para Cenário C: {MIN_LIMPOS}.")
        return

    print(f"\n   Painéis com detecção YOLO: {len(paineis_com_deteccao)}")
    print(f"   Painéis limpos (YOLO): {len(paineis_limpos)}")

    dados_finais = []
    mapa_cenarios = {}

    # =========================================================================
    # CENÁRIO A — Convergência: YOLO detecta + Eficiência BAIXA (45–65%)
    # Interpretação: sujeira visual confirmada por queda elétrica real.
    # Ação esperada do PAND: LIMPEZA_CORRETIVA
    # Referência: Maghami et al. (2016) — perda severa por sujeira
    # =========================================================================
    for p in paineis_com_deteccao[:15]:
        dados_finais.append({
            "id_painel": p,
            "corrente_a": rng.uniform(3.0, 5.0),
            "tensao_v": rng.uniform(25.0, 30.0),
            "eficiencia": rng.uniform(0.45, 0.65),
            "status_inversor": gerar_status_inversor("A", rng),
        })
        mapa_cenarios[p] = "A (Convergência Ruim)"

    # =========================================================================
    # CENÁRIO B — Falso Positivo: YOLO detecta + Eficiência ALTA (92–98%)
    # Interpretação: sensor visual detectou artefato/sombra, mas geração é nominal.
    # A limpeza seria custo sem retorno — o PAND DEVE ignorar o YOLO.
    # Ação esperada do PAND: FALSO_POSITIVO_IGNORADO
    # =========================================================================
    fim_b = 15 + n_b
    for p in paineis_com_deteccao[15:fim_b]:
        dados_finais.append({
            "id_painel": p,
            "corrente_a": rng.uniform(8.5, 9.8),
            "tensao_v": rng.uniform(38.0, 42.0),
            "eficiencia": rng.uniform(0.92, 0.98),
            "status_inversor": gerar_status_inversor("B", rng),
        })
        mapa_cenarios[p] = "B (Falso Positivo)"

    # =========================================================================
    # CENÁRIO E — Ambiguidade: YOLO detecta + Eficiência MÉDIA (70–82%)
    # Interpretação: sujeira presente mas perda não é crítica ainda.
    # Zona cinzenta — decisão depende de contexto adicional.
    # Ação esperada do PAND: INSPECAO_AMBIGUIDADE
    # =========================================================================
    for p in paineis_com_deteccao[fim_b:]:
        dados_finais.append({
            "id_painel": p,
            "corrente_a": rng.uniform(6.0, 7.5),
            "tensao_v": rng.uniform(32.0, 36.0),
            "eficiencia": rng.uniform(0.70, 0.82),
            "status_inversor": gerar_status_inversor("E", rng),
        })
        mapa_cenarios[p] = "E (Ambiguidade Extrema)"

    # =========================================================================
    # CENÁRIO C — Falha Oculta: YOLO limpo + Eficiência CRÍTICA (20–40%)
    # Interpretação: defeito elétrico interno — invisível ao sensor visual.
    # É o cenário de maior valor do PAND: só a fusão semântica detecta isso.
    # Ação esperada do PAND: MANUTENCAO_ELETRICA
    # Referência: Pillai & Rajasekar (2018) — microcrack, string aberta
    # =========================================================================
    for p in paineis_limpos[:20]:
        dados_finais.append({
            "id_painel": p,
            "corrente_a": rng.uniform(1.0, 2.5),
            "tensao_v": rng.uniform(18.0, 24.0),
            "eficiencia": rng.uniform(0.20, 0.40),
            "status_inversor": gerar_status_inversor("C", rng),
        })
        mapa_cenarios[p] = "C (Falha Oculta)"

    # =========================================================================
    # CENÁRIO D — Controle: YOLO limpo + Eficiência ÓTIMA (96–99%)
    # Interpretação: painel saudável, nenhuma anomalia em nenhum sensor.
    # Ação esperada do PAND: OPERACAO_NORMAL
    # =========================================================================
    for p in paineis_limpos[20:]:
        dados_finais.append({
            "id_painel": p,
            "corrente_a": rng.uniform(9.0, 10.0),
            "tensao_v": rng.uniform(40.0, 44.0),
            "eficiencia": rng.uniform(0.96, 0.99),
            "status_inversor": gerar_status_inversor("D", rng),
        })
        mapa_cenarios[p] = "D (Controle)"

    # --- Salva os arquivos ---
    os.makedirs("data", exist_ok=True)
    df_telemetria = pd.DataFrame(dados_finais).sort_values("id_painel")
    df_telemetria.to_excel("data/solar_data.xlsx", index=False)

    with open("data/mapa_cenarios.json", "w", encoding="utf-8") as f:
        json.dump(mapa_cenarios, f, ensure_ascii=False, indent=2)

    # --- Relatório de geração ---
    from collections import Counter
    dist = Counter(mapa_cenarios.values())

    print("\n" + "=" * 55)
    print("✅ TELEMETRIA GERADA COM SUCESSO")
    print("=" * 55)
    print(f"{'Cenário':<30} {'N':>5}  {'Efic (média)':>12}  {'Status Principal'}")
    print("-" * 55)

    for cenario in [
        "A (Convergência Ruim)", "B (Falso Positivo)", "C (Falha Oculta)",
        "D (Controle)", "E (Ambiguidade Extrema)"
    ]:
        paineis_c = [p for p, c in mapa_cenarios.items() if c == cenario]
        subset = df_telemetria[df_telemetria["id_painel"].isin(paineis_c)]
        efic_media = subset["eficiencia"].mean()
        status_dist = subset["status_inversor"].value_counts().to_dict()
        print(f"  {cenario:<28} {len(paineis_c):>5}  {efic_media:>11.1%}  {status_dist}")

    print("-" * 55)
    print(f"  {'TOTAL':<28} {len(dados_finais):>5}")
    print("=" * 55)
    print(f"\n📁 Arquivos salvos:")
    print(f"   data/solar_data.xlsx     ({len(dados_finais)} painéis)")
    print(f"   data/mapa_cenarios.json  ({len(mapa_cenarios)} entradas, seed={RANDOM_SEED})")


if __name__ == "__main__":
    gerar_telemetria_inteligente()