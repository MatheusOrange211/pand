# PAND: Orquestração entre Visão Computacional e LLM para Tomada de Decisão na Manutenção Fotovoltaica

**Departamento de Ciência da Computação – Universidade Federal de Roraima (UFRR) – Boa Vista – RR – Brasil**  
matheusnaranjocorrea@gmail.com, filipe.dwan@ufrr.br

---

## Resumo
A expansão solar dificulta a manutenção, com inspeções visuais isoladas frequentemente imprecisas. Este estudo valida a arquitetura PAND para reduzir falsos positivos via fusão de dados visuais (YOLOv11m) e telemétricos, orquestrada por um núcleo agêntico (Llama 3.3). Em 100 testes simulados com conflito visual-elétrico, obteve Taxa de Assertividade de 92,0% e eficácia de 100% em casos críticos, com o tempo de inferência variando conforme a ambiguidade do cenário (média de 19,6s). Demonstra a viabilidade da fusão semântica para automação industrial, sugerindo futuras pesquisas em modelos de borda.

## Abstract
Solar expansion challenges maintenance, with isolated visual inspections often being inaccurate. This study validates the PAND architecture to reduce false positives through the fusion of visual (YOLOv11m) and telemetric data, orchestrated by an agentic core (Llama 3.3). In 100 simulated tests involving visual-electrical conflict, the system achieved a 92.0% Assertiveness Rate and 100% efficacy in critical cases, with inference time varying according to scenario ambiguity (19.6s average). The results indicate the technical feasibility of semantic fusion for industrial automation, suggesting future research into edge computing models.

---

## Estrutura do Projeto

O repositório `pandtcc` contém a implementação do sistema PAND. A seguir está a organização de diretórios e arquivos relevantes:

```
pandtcc/
├── app.py                   # Ponto de entrada para execução de serviços/GUI
├── main.py                  # Script principal para integrar agentes e fluxo de dados
├── etapa_1_gerar_resultados_teste_yolo.py  # Lote de inferência visual
├── etapa_2_gerar_dados_teste_telemetria.py # Processamento de dados telemétricos
├── etapa_3_agentes.py       # Orquestração de agentes LLM
├── analizar_resultados.py   # Análise final dos resultados de teste
├── requirements.txt         # Dependências Python do projeto
├── data/                    # Conjunto de dados de entrada e saída
│   ├── mapa_cenarios.json   # Configurações de cenários de teste
│   ├── imagens_teste/       # Fotos de painéis para inferência YOLO
│   ├── resultados_yolo_reais.csv # Saída do passo 1 (gabarito YOLO)
│   └── ...
├── models/                  # Modelos pré-treinados e artefatos gerados
│   ├── best_float32.tflite  # Exemplo de modelo de visão
│   └── ...
└── pand_system/             # Pacote interno contendo ferramentas
    ├── agent.py             # Definição do agente Llama
    └── tools/
        ├── vision_tool.py   # Funções para inferência visual
        └── data_tool.py     # Conversão/manipulação de telemetria
```

### Descrição dos Componentes
- **app.py / main.py**: scripts de alto nível usados para iniciar o sistema e demonstrar seu funcionamento.
- **etapa_1_gerar_resultados_teste_yolo.py**: percorre imagens de teste, chama o YOLO via `vision_tool` e salva resultados.
- **etapa_2_gerar_dados_teste_telemetria.py**: similar ao primeiro, mas para processar dados de telemetria.
- **etapa_3_agentes.py**: implementa a lógica de fusão e orquestração usando Llama 3.3 como agente central.
- **pand_system/**: código de suporte modularizado em ferramentas de visão e dados, além da classe de agente.

## Instalação e Configuração

1. **Clonar o repositório**
   ```powershell
   git clone https://seu-repositorio.git
   cd pandtcc
   ```

2. **Criar e ativar ambiente virtual**
   ```powershell
   python -m venv .venv
   & .\.venv\Scripts\Activate.ps1     # Windows PowerShell
   ```

3. **Instalar dependências**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Preparar dados de entrada**
   - Copie imagens de teste para `data/imagens_teste/`
   - Garanta que `mapa_cenarios.json` contenha os cenários desejados

5. **Executar etapas de processamento**
   - Visual: ```python etapa_1_gerar_resultados_teste_yolo.py```  
   - Telemetria: ```python etapa_2_gerar_dados_teste_telemetria.py```  
   - Orquestração: ```python etapa_3_agentes.py```  
   - Resultados: ```python analisar_resultados.py```  

6. **Configuração adicional**
   - Ajuste de parâmetros do agente Llama no arquivo `pand_system/agent.py` conforme necessidade.
   - Modelos YOLO ou TFLite podem ser substituídos ou recalibrados em `models/`.

---

📌 *Este README serve como guia para pesquisadores e desenvolvedores interessados na fusão semântica de visão computacional e LLMs para manutenção fotovoltaica.*


_Em homenagem a Pandora_
