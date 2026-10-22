# Dataset

O dataset **não** é versionado neste repositório. Ele foi criado e anotado no Roboflow:

- Roboflow: https://app.roboflow.com/roberli/detector-de-gas
- Classe única: `gasleak`

Para treinar localmente, exporte o dataset no formato **YOLOv8** e extraia aqui:

```
data/dataset/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

Depois ajuste o campo `path` em [`configs/yolov8/configs_modelo.yaml`](../configs/yolov8/configs_modelo.yaml)
para `data/dataset/` (no Colab o caminho usado era `/content/dataset/`).

Vídeos de teste, vídeos de resultado e demais arquivos ficam na pasta do Google Drive do projeto:
https://drive.google.com/drive/folders/1fbmdwq0HorSvJ0A4UIHge39IE8-hVHQb
