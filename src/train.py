"""Treinamento do detector de vazamento de gás (YOLOv8s) com os mesmos parâmetros do Colab.

Antes de rodar, baixe o dataset do Roboflow no formato "YOLOv8" e extraia em data/dataset/
(com as pastas train/ e valid/). Ajuste o campo `path` de configs/yolov8/configs_modelo.yaml
se usar outro local.

Exemplo:
    python src/train.py --data configs/yolov8/configs_modelo.yaml --epochs 500
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Treino YOLOv8 - Gas Leak")
    parser.add_argument("--data", default=str(RAIZ / "configs" / "yolov8" / "configs_modelo.yaml"))
    parser.add_argument(
        "--model",
        default="yolov8s.yaml",
        help="yolov8s.yaml = treino do zero (como no Colab); yolov8s.pt = transfer learning a partir do COCO",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100, help="Early stopping")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="yolov8s_modelo")
    parser.add_argument("--export-onnx", action="store_true", help="Exporta o melhor modelo para ONNX ao final")
    args = parser.parse_args()

    modelo = YOLO(args.model)
    modelo.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        project=str(RAIZ / "runs" / "detect"),
        plots=True,
    )
    modelo.val()

    if args.export_onnx:
        caminho = modelo.export(format="onnx", simplify=True)
        print(f"Modelo ONNX exportado em: {caminho}")


if __name__ == "__main__":
    main()
