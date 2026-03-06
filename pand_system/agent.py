"""
agent.py — PAND (Predictive Agent for Neural Dirt Detection)
Versão 2.0 — Prompts reestruturados para maximizar TAD no Cenário B (Falso Positivo).

Problema identificado na v1.0:
- O agente ignorava a eficiência alta ao ver qualquer detecção YOLO.
- Solução: O agente mestre agora recebe os dados de telemetria NO MESMO CONTEXTO
  da análise visual, forçando raciocínio de fusão ANTES de decidir.
- O "filtro econômico" agora é apresentado como um princípio de engenharia
  (Potência Gerada vs. Custo de O&M), não como um IF disfarçado.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from pand_system.tools.vision_tool import analisar_imagem_painel
from pand_system.tools.data_tool import consultar_historico_painel

load_dotenv()

# --- CONFIGURAÇÃO DE CHAVES ---
keys = [
    os.getenv("GROQ_API_KEY_01"),
    os.getenv("GROQ_API_KEY_02"),
    os.getenv("GROQ_API_KEY_03"),
    os.getenv("GROQ_API_KEY_04"),
    os.getenv("GROQ_API_KEY_05"),
    os.getenv("GROQ_API_KEY_06"),
]
current_key_index = 0
client = Groq(api_key=keys[current_key_index])
MODEL_ID = "llama-3.3-70b-versatile"


def rotate_key():
    global current_key_index, client
    current_key_index += 1
    if current_key_index < len(keys) and keys[current_key_index]:
        print(f"\n🔄 Trocando para API KEY 0{current_key_index + 1}...")
        client = Groq(api_key=keys[current_key_index])
        return True
    return False


def call_groq_with_rotation(messages, tools=None, tool_choice=None):
    while True:
        try:
            params = {
                "model": MODEL_ID,
                "messages": messages,
                "temperature": 0.1,  # Reduzido para maior consistência decisória
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice if tool_choice else "auto"
            return client.chat.completions.create(**params)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                if rotate_key():
                    continue
            print(f"\n⚠️ Erro crítico na API: {e}")
            raise e


# ---------------------------------------------------------------------------
# AGENTE 1 — VISUAL
# Responsabilidade: interpretar o JSON do YOLO e gerar um laudo técnico
# estruturado que já destaca CONFIANÇA das detecções de forma explícita.
# ---------------------------------------------------------------------------
def agente_visual(dados_yolo_json: str) -> str:
    """
    Interpreta o output bruto do YOLO e devolve um laudo visual estruturado.
    Destaca explicitamente: o que foi detectado, com qual confiança, e o que
    isso significa do ponto de vista de anomalia física real vs. ruído do sensor.
    """
    system_prompt = """
Você é o Agente de Visão Computacional do sistema PAND.
Sua função é interpretar o JSON de saída do modelo YOLO e produzir um
laudo técnico visual ESTRUTURADO e OBJETIVO.

### REGRAS OBRIGATÓRIAS:
1. Liste cada detecção com: classe detectada + score de confiança.
2. Classifique a confiança em três faixas:
   - ALTA (≥ 0.70): Anomalia visual confirmada, muito provável.
   - MÉDIA (0.40 – 0.69): Anomalia suspeita, requer correlação com dados elétricos.
   - BAIXA (< 0.40): Sinal fraco detectado. Mesmo com baixa confiança, a detecção NÃO deve ser descartada — deve ser correlacionada com a telemetria elétrica. Se a eficiência estiver baixa, a detecção reforça a hipótese de sujeira.
3. Se status for "clean", declare explicitamente: "PAINEL VISUALMENTE LIMPO — Nenhuma anomalia detectada pelo sensor visual."
4. Finalize com uma linha: "SINTESE VISUAL: [COM_DETECCAO | SEM_DETECCAO]"

### PROIBIDO:
- Recomendar ações de manutenção (esse não é seu papel).
- Inventar dados não presentes no JSON.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"JSON YOLO recebido:\n{dados_yolo_json}"},
    ]
    response = call_groq_with_rotation(messages)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# AGENTE 2 — MESTRE (Fusão Semântica + Decisão)
# Responsabilidade: cruzar visão + telemetria e tomar a decisão final.
# MUDANÇA CRÍTICA v2.0:
# - O agente mestre busca a telemetria ANTES de iniciar o raciocínio.
# - O prompt usa linguagem de "raciocínio de engenharia", não de regras IF.
# - A eficiência é apresentada como "evidência elétrica" a ser ponderada.
# ---------------------------------------------------------------------------
def agente_mestre(painel_id: str, relatorio_visual: str) -> str:
    system_prompt = """
Você é o PAND — Gerente Sênior de O&M (Operação e Manutenção) de uma usina
fotovoltaica de 5 MWp. Você é responsável por decisões que impactam diretamente
a lucratividade e a segurança da planta.

Você recebe dois inputs:
  (A) Laudo Visual: saída interpretada do sensor YOLO (visão computacional)
  (B) Telemetria Elétrica: dados do Digital Twin (eficiência, corrente, tensão, inversor)

Sua missão é realizar a FUSÃO SEMÂNTICA entre esses dois sensores e emitir um
diagnóstico técnico preciso e economicamente responsável.

═══════════════════════════════════════════════════════
 PRINCÍPIOS DE ENGENHARIA DE O&M — RACIOCINE COM ELES
═══════════════════════════════════════════════════════

PRINCÍPIO 1 — A EVIDÊNCIA ELÉTRICA É SOBERANA:
  A eficiência do painel é calculada a partir de medições físicas reais
  (corrente e tensão de saída). Se a eficiência está acima de 88%, o painel
  está gerando energia normalmente. Uma detecção visual isolada de sujeira,
  sem queda elétrica correspondente, indica que: (a) a sujeira é superficial
  e não obstrui células fotovoltaicas, OU (b) o sensor visual detectou
  artefatos ou sombras passageiras (falso positivo instrumental).
  NESSE CASO: A limpeza seria um custo sem retorno — PROIBIDA pela política de O&M.

PRINCÍPIO 2 — SUJEIRA SÓ JUSTIFICA INTERVENÇÃO SE HÁ PERDA MENSURÁVEL:
  Perda de eficiência abaixo de 80% + detecção visual = caso real para limpeza.
  Eficiência acima de 88% + detecção visual = falso positivo econômico, ignorar a visão.
  Zona cinzenta (80-88%) + detecção visual = ambiguidade, monitorar antes de agir.

PRINCÍPIO 3 — FALHAS ELÉTRICAS SÃO INVISÍVEIS AO OLHO:
  Microcracks, degradação de células, falha de string, conexões oxidadas —
  esses defeitos NÃO são detectados pelo YOLO. Se o painel está visualmente
  limpo mas a eficiência está abaixo de 70%, a causa é elétrica/interna.
  Esse cenário é crítico: requer MANUTENÇÃO ELÉTRICA especializada.

PRINCÍPIO 4 — OPERAÇÃO NORMAL SÓ SE AMBOS OS SENSORES CONCORDAM:
  Painel limpo visualmente + eficiência acima de 90% = sistema saudável,
  nenhuma ação necessária.

PRINCÍPIO 5 — DETECÇÃO COM BAIXA CONFIANÇA NÃO EQUIVALE A AUSÊNCIA DE DETECÇÃO:
  Se o YOLO detectou qualquer anomalia (Dust, Defective, Snow, Bird Drop),
  mesmo com score < 0.50, o painel NÃO está visualmente limpo — está com
  detecção incerta. Nesse caso:
  - Se eficiência < 80%: a detecção incerta + queda elétrica = LIMPEZA_CORRETIVA.
    A limpeza é a intervenção menos invasiva e deve ser tentada primeiro.
  - Se eficiência > 88%: detecção incerta + geração ótima = FALSO_POSITIVO_IGNORADO.
  NUNCA use MANUTENCAO_ELETRICA quando o YOLO detectou algo, mesmo com baixa confiança.
  MANUTENCAO_ELETRICA é exclusivo para painéis com status="clean" no YOLO.

═══════════════════════════════════════════════════════
 TAGS DE DECISÃO OBRIGATÓRIAS (use EXATAMENTE uma delas)
═══════════════════════════════════════════════════════
[CÓDIGO: LIMPEZA_CORRETIVA]       → YOLO detectou QUALQUER coisa + Eficiência < 80%
[CÓDIGO: FALSO_POSITIVO_IGNORADO] → YOLO detectou QUALQUER coisa + Eficiência > 88%
[CÓDIGO: MANUTENCAO_ELETRICA]     → YOLO retornou "clean" (SEM detecção) + Eficiência < 70%
[CÓDIGO: OPERACAO_NORMAL]         → YOLO retornou "clean" (SEM detecção) + Eficiência > 88%
[CÓDIGO: INSPECAO_AMBIGUIDADE]    → YOLO detectou + Eficiência 80-88% OU YOLO clean + Eficiência 70-88%

REGRA DE FORMATAÇÃO:
Inicie sua resposta OBRIGATORIAMENTE com a tag [CÓDIGO: X].
Em seguida, justifique o raciocínio citando os valores numéricos da telemetria
e o nível de confiança da detecção visual. Seja técnico e conciso.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"ID do Painel: {painel_id}\n\n"
                f"=== LAUDO VISUAL (YOLO) ===\n{relatorio_visual}\n\n"
                "=== AÇÃO NECESSÁRIA ===\n"
                "Use a ferramenta 'consultar_historico_painel' para obter a "
                "telemetria elétrica deste painel e, após analisar ambos os "
                "sensores juntos, emita o diagnóstico com a tag [CÓDIGO: X]."
            ),
        },
    ]

    tool_def = [
        {
            "type": "function",
            "function": {
                "name": "consultar_historico_painel",
                "description": (
                    "Busca os dados de telemetria elétrica do painel no Digital Twin. "
                    "Retorna: eficiencia (0.0-1.0), corrente_a, tensao_v, status_inversor."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id_painel": {
                            "type": "string",
                            "description": "ID do painel, ex: 'painel_01'",
                        }
                    },
                    "required": ["id_painel"],
                },
            },
        }
    ]

    # Estratégia de tool_choice com compatibilidade máxima entre versões do SDK Groq.
    # Tentamos "required" primeiro (mais simples e amplamente suportado).
    # Se falhar por incompatibilidade, tentamos a sintaxe de dicionário.
    # Em último caso, buscamos a telemetria manualmente e injetamos no contexto.
    def _chamar_com_tool(tc_value):
        return call_groq_with_rotation(messages, tools=tool_def, tool_choice=tc_value)

    response = None
    for tc in ["required", {"type": "function", "function": {"name": "consultar_historico_painel"}}]:
        try:
            response = _chamar_com_tool(tc)
            break
        except Exception:
            continue

    # Se nenhuma sintaxe funcionou, força a consulta manualmente (garantia absoluta)
    if response is None:
        telemetria_manual = consultar_historico_painel(painel_id)
        messages.append({
            "role": "user",
            "content": (
                f"[TELEMETRIA INJETADA MANUALMENTE]\n{telemetria_manual}\n\n"
                "Com esses dados, aplique os 4 Princípios de O&M e emita o diagnóstico. "
                "Inicie OBRIGATORIAMENTE com [CÓDIGO: X]."
            )
        })
        response = call_groq_with_rotation(messages)
        return response.choices[0].message.content

    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool in msg.tool_calls:
            args = json.loads(tool.function.arguments)
            resultado = consultar_historico_painel(args.get("id_painel", painel_id))
            messages.append(
                {
                    "tool_call_id": tool.id,
                    "role": "tool",
                    "name": "consultar_historico_painel",
                    "content": str(resultado),
                }
            )

        # Agora com telemetria em mãos, pedimos a decisão final
        messages.append(
            {
                "role": "user",
                "content": (
                    "Você agora tem os dados do sensor visual E da telemetria elétrica. "
                    "Aplique os 4 Princípios de Engenharia de O&M e emita o diagnóstico final. "
                    "LEMBRE-SE: se a eficiência estiver acima de 88%, a limpeza é proibida "
                    "independente do que o YOLO detectou. Inicie OBRIGATORIAMENTE com [CÓDIGO: X]."
                ),
            }
        )
        final_resp = call_groq_with_rotation(messages)
        return final_resp.choices[0].message.content

    # Fallback final: o agente respondeu sem usar a ferramenta mesmo com tool_choice.
    # Injeta a telemetria manualmente e pede nova decisão.
    telemetria_manual = consultar_historico_painel(painel_id)
    messages.append(msg)
    messages.append({
        "role": "user",
        "content": (
            f"[ATENÇÃO: A telemetria não foi consultada. Dados injetados manualmente]\n"
            f"{telemetria_manual}\n\n"
            "Revise sua análise considerando ESSES dados elétricos e emita o diagnóstico final. "
            "Inicie OBRIGATORIAMENTE com [CÓDIGO: X]."
        )
    })
    final_resp = call_groq_with_rotation(messages)
    return final_resp.choices[0].message.content


# ---------------------------------------------------------------------------
# AGENTE 3 — PLANEJADOR (Relatório A3)
# Responsabilidade: estruturar o diagnóstico do Mestre em formato A3.
# ---------------------------------------------------------------------------
def agente_planejador(diagnostico: str) -> str:
    system_prompt = """
Você é o Agente Planejador do sistema PAND.
Sua função é estruturar o diagnóstico técnico recebido no formato de
Relatório A3 de Manutenção, seguindo o template abaixo OBRIGATORIAMENTE.

### TEMPLATE A3 OBRIGATÓRIO:

**Código de Decisão:** [extraia e copie EXATAMENTE a tag [CÓDIGO: X] do diagnóstico]

**Contexto:** (2-3 linhas: o que está acontecendo com o painel, cite eficiência e status)

**Causa Raiz:** (a causa técnica que justifica o código de decisão)

**Contramedidas:** (ação concreta a ser tomada pela equipe de O&M)

**Prazo Sugerido:**
- LIMPEZA_CORRETIVA → Próxima janela de manutenção programada (≤ 7 dias)
- FALSO_POSITIVO_IGNORADO → Nenhuma ação. Manter monitoramento passivo.
- MANUTENCAO_ELETRICA → Urgente (≤ 48h), risco de dano permanente ao ativo
- OPERACAO_NORMAL → Nenhuma ação. Próxima inspeção no ciclo padrão.
- INSPECAO_AMBIGUIDADE → Reavaliação em 72h com nova coleta de dados

### REGRAS:
- Seja conciso e técnico. Evite repetição.
- O Código de Decisão DEVE ser idêntico ao do diagnóstico recebido.
- Não invente dados não presentes no diagnóstico.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Diagnóstico do Agente Mestre:\n\n{diagnostico}"},
    ]
    response = call_groq_with_rotation(messages)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# ORQUESTRADOR PRINCIPAL
# ---------------------------------------------------------------------------
def run_agent_analysis(image_bytes: bytes, panel_id: str) -> str:
    """
    Executa o pipeline completo: YOLO → Agente Visual → Agente Mestre → Planejador.
    Retorna o Relatório A3 final com o [CÓDIGO: X] embutido.
    """
    yolo_data = analisar_imagem_painel(image_bytes)
    relatorio_v = agente_visual(yolo_data)
    diagnostico = agente_mestre(panel_id, relatorio_v)
    return agente_planejador(diagnostico)