# importa módulos do sistema e bibliotecas de terceiros
import os  # manipulação de caminhos e operações do sistema de arquivos
import pandas as pd  # estrutura de dados tabular para resultados
import time  # usado para marcar o tempo de execução
# função customizada que chama o modelo YOLO na infraestrutura pand_system
from pand_system.tools.vision_tool import analisar_imagem_painel


def rodar_bateria_yolo():
    # caminho base onde estão as imagens de teste
    pasta_imagens = os.path.join("data", "imagens_teste")
    # lista para armazenar dicionários com saída de cada inferência
    resultados_yolo = []
    
    # sinaliza o início do processamento em lote
    print("👁️ INICIANDO ETAPA 1: INFERÊNCIA YOLO EM LOTE...")
    t0_total = time.time()  # guarda o tempo inicial para cálculo posterior
    
    # percorre 100 painéis numerados de 1 a 100
    for i in range(1, 101):
        painel_id = f"painel_{i:02d}"  # formata com dois dígitos, ex. painel_01
        caminho_imagem = os.path.join(pasta_imagens, f"{painel_id}.jpg")
        
        # verifica se o arquivo existe antes de tentar abrir
        if not os.path.exists(caminho_imagem):
            print(f"⚠️ Faltando: {painel_id}.jpg")
            continue  # pula para o próximo painel se faltar a imagem
            
        print(f"Processando visão de {painel_id}...", end=" ")
        
        # abre o arquivo em binário para enviar ao analisador YOLO
        with open(caminho_imagem, "rb") as f:
            image_bytes = f.read()
            
        # realiza a inferência e obtém o JSON de saída do YOLO
        yolo_json = analisar_imagem_painel(image_bytes)
        
        # adiciona o resultado à lista, mantendo referência ao painel
        resultados_yolo.append({
            "Painel": painel_id,
            "YOLO_Output": yolo_json
        })
        print("✅ Concluído")

    # após o loop, transforma os resultados em DataFrame e salva em CSV
    df = pd.DataFrame(resultados_yolo)
    df.to_csv("data/resultados_yolo_reais.csv", index=False)
    
    # calcula tempo total e imprime mensagem de conclusão
    tempo_total = round(time.time() - t0_total, 2)
    print(f"🎯 Etapa 1 finalizada em {tempo_total}s! Arquivo salvo em data/resultados_yolo_reais.csv")

# se o script for executado diretamente, chama a função principal
if __name__ == "__main__":
    rodar_bateria_yolo()