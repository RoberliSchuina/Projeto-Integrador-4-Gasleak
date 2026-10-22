# Dataset

Os datasets **não** são versionados neste repositório.

| Dataset | Onde está | Conteúdo |
|---|---|---|
| Treino do modelo final (YOLOv8s) | [Google Drive](https://drive.google.com/drive/folders/1fbmdwq0HorSvJ0A4UIHge39IE8-hVHQb): `obj.zip` (treino) e `valid.zip` (validação) | Classe única `gasleak` |
| Gas Leak v1 (detecção) | [Roboflow Universe](https://universe.roboflow.com/roberli/gas-leak-x7van-qfi52/dataset/1), licença CC BY 4.0 | 166 imagens (118 treino, 28 validação, 20 teste), classe `gas` |
| gas-leak v1 (segmentação) | [Roboflow Universe](https://universe.roboflow.com/roberli/gas-leak-ka6fi-uahui/dataset/1), licença CC BY 4.0 | 1.995 imagens extraídas de vídeo |

Para treinar localmente, extraia o dataset no formato **YOLOv8** aqui:

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

Vídeos de teste, vídeos de resultado e demais arquivos ficam na mesma pasta do Google Drive.
