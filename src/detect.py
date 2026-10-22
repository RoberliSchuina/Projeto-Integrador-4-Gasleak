"""Detecção de vazamento de gás em imagens, vídeos ou câmera com o modelo YOLOv8 treinado.

Exemplos:
    python src/detect.py --source caminho/para/imagem.jpg
    python src/detect.py --source caminho/para/video.mp4 --conf 0.4
    python src/detect.py --source 0 --show            # webcam / câmera OGI conectada
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
PESOS_PADRAO = RAIZ / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Detector de vazamento de gás (YOLOv8)")
    parser.add_argument("--source", required=True, help="Imagem, vídeo, pasta, URL ou índice da câmera (ex.: 0)")
    parser.add_argument("--weights", default=str(PESOS_PADRAO), help="Pesos do modelo (.pt ou .onnx)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confiança mínima (Colab: 0.5 imagens, 0.4 vídeos)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho de entrada da rede")
    parser.add_argument("--show", action="store_true", help="Exibe o resultado em uma janela")
    parser.add_argument("--project", default=str(RAIZ / "runs" / "detect"), help="Pasta de saída")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    modelo = YOLO(args.weights)
    resultados = modelo.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        show=args.show,
        save=True,
        project=args.project,
        stream=True,  # processa quadro a quadro, sem acumular vídeos inteiros na memória
    )

    for r in resultados:
        n = len(r.boxes)
        if n:
            print(f"[ALERTA] {n} vazamento(s) de gás detectado(s) em {Path(r.path).name}")

    print(f"Resultados salvos em: {args.project}")


if __name__ == "__main__":
    main()
