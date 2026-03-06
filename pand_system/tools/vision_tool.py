import json
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

def analisar_imagem_painel(image_input):
    """
    Recebe bytes da imagem ou caminho, roda o YOLO e retorna JSON.
    """
    try:
        # Tenta carregar o modelo. Se não existir, avisa.
        try:
            model = YOLO('models/best.onnx')
        except:
            return json.dumps({"status": "erro", "msg": "Modelo YOLO (best.onnx) não encontrado."})

        # Processamento da imagem (converte PIL/Bytes para formato OpenCV)
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            return json.dumps({"erro": "Formato de imagem inválido"})

        # Inferência
        results = model(img)
        
        # Formata saída
        deteccoes = []
        found = False
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                
                deteccoes.append({
                    "objeto": class_name,
                    "confianca": round(conf, 2)
                })
                found = True

        if found:
            return json.dumps({
                "status": "detection_found",
                "detalhes": deteccoes
            })
        else:
            return json.dumps({"status": "clean", "msg": "Nenhuma anomalia visual detectada."})

    except Exception as e:
        return json.dumps({"erro_visao": str(e)})