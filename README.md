# Projeto Integrador 4 - Gasleak

**IA para detecção e monitoramento de vazamento de gás em plataformas de petróleo usando visão computacional.**

Projeto Integrador 4 (FAESA, 2026.02), com base no trabalho da optativa *Uso de Inteligência Artificial na Descarbonização* (D010397).

<p align="center">
  <img src="results/exemplos/deteccao_04.jpg" width="32%">
  <img src="results/exemplos/deteccao_05.jpg" width="32%">
  <img src="results/exemplos/deteccao_06.jpg" width="32%">
</p>

## Contexto

As **emissões fugitivas** (vazamentos em tubulações, flanges, válvulas, instrumentos, selos de bomba, juntas e conexões) estão entre as principais fontes de gases de efeito estufa (GEE) na indústria de óleo e gás, ao lado da queima em tocha (*flaring*), do *venting* e da geração de energia a bordo.

As soluções tradicionais (detectores portáteis de lapela, detectores fixos, ultrassom, sensores eletroquímicos e analisadores de metano) são pontuais: só detectam o gás que chega até o sensor. As **câmeras OGI** (*Optical Gas Imaging*) tornam a pluma de gás visível, captando no infravermelho o contraste térmico entre o gás e o ambiente, mas ainda dependem de um operador olhando a imagem.

**Proposta:** treinar um modelo de detecção de objetos (YOLO) que identifique automaticamente plumas de vazamento de gás em imagens e vídeos de câmeras OGI/térmicas, viabilizando o monitoramento contínuo e alarmes automáticos.

## Como funciona

```
Câmera OGI / vídeo / imagem ──► YOLOv8s (classe "gasleak") ──► caixas + confiança ──► alerta
```

| Item | Escolha |
|---|---|
| Técnica | Visão computacional, detecção de objetos |
| Algoritmo | YOLOv8s (Ultralytics), classe única `gasleak` |
| Ambiente | Google Colab com GPU |
| Bibliotecas | Ultralytics, OpenCV |
| Dataset | Imagens de vazamentos anotadas no [Roboflow](https://app.roboflow.com/roberli/detector-de-gas) |
| Versão anterior | YOLOv4 + Darknet (configs em [`configs/darknet`](configs/darknet)) |

## Resultados

Treino de **500 épocas** (imgsz 640, batch 16, partindo do zero com `yolov8s.yaml`), com *early stopping* na época 475 (paciência 100). O melhor modelo ([`weights/best.pt`](weights/best.pt)) é o da **época 375**:

| Precisão | Recall | mAP@50 | mAP@50-95 |
|:---:|:---:|:---:|:---:|
| 0,905 | 0,789 | 0,866 | 0,546 |

Tempo total de treino: cerca de 76 min. Curvas de perda e métricas por época estão em [`results/treinamento`](results/treinamento).

![Curvas de treino](results/treinamento/results.png)

| Rótulos reais (validação) | Predições do modelo |
|---|---|
| ![labels](results/treinamento/labels.jpg) | ![pred](results/treinamento/val_batch0_pred.jpg) |

O vídeo com o resultado da detecção está na [pasta do projeto no Google Drive](https://drive.google.com/drive/folders/1fbmdwq0HorSvJ0A4UIHge39IE8-hVHQb).

## Estrutura do repositório

```
├── notebooks/
│   └── YOLOv8_Treinamento_Customizado_GasLeak.ipynb   # notebook original do Colab (sem saídas)
├── src/
│   ├── train.py      # treino com os mesmos parâmetros do Colab
│   └── detect.py     # inferência em imagem, vídeo, pasta ou câmera
├── configs/
│   ├── yolov8/       # configs_modelo.yaml (dataset) e args.yaml (hiperparâmetros do treino)
│   └── darknet/      # obj.data e obj.names da versão YOLOv4/Darknet
├── weights/
│   └── best.pt       # modelo treinado (YOLOv8s, 22 MB)
├── results/
│   ├── treinamento/  # results.csv, curvas e lotes de validação
│   └── exemplos/     # imagens de teste com as detecções
└── data/             # instruções para baixar o dataset (não versionado)
```

## Como usar

### No Google Colab (recomendado para treinar)

Abra o [notebook no Colab](https://colab.research.google.com/drive/1HVmY2yG92IVbBllGE5xDI1HWjclknZVJ) ou a cópia em [`notebooks/`](notebooks), selecione um ambiente com GPU e execute as células em ordem. O notebook espera os arquivos `obj.zip` (treino) e `valid.zip` (validação) em `MyDrive/GasLeak/`, disponíveis na [pasta do Google Drive](https://drive.google.com/drive/folders/1fbmdwq0HorSvJ0A4UIHge39IE8-hVHQb).

### Localmente

```bash
git clone https://github.com/RoberliSchuina/Projeto-Integrador-4-Gasleak.git
cd Projeto-Integrador-4-Gasleak
pip install -r requirements.txt
```

Detectar vazamentos com o modelo treinado:

```bash
python src/detect.py --source results/exemplos/deteccao_04.jpg
python src/detect.py --source video.mp4 --conf 0.4
python src/detect.py --source 0 --show        # câmera conectada ao computador
```

As saídas são salvas em `runs/detect/`. Treinar novamente (veja [`data/README.md`](data/README.md) para baixar o dataset):

```bash
python src/train.py --epochs 500 --export-onnx
```

## Desafios e próximos passos

- **Dados:** há poucas imagens públicas de vazamentos em câmeras OGI. O dataset foi montado a partir de fontes públicas (TensorFlow, ImageNet, Kaggle, GitHub, Google) e anotado manualmente.
- **Generalização:** o recall de 0,79 indica que parte das plumas ainda escapa. Ampliar o dataset com vídeos reais de plataformas e testar transfer learning (`--model yolov8s.pt`) são caminhos naturais.
- **Implantação:** rodar o modelo em ONNX na borda (junto à câmera) e integrar os alertas a um sistema de supervisão.

## Links

- Notebook no Colab: https://colab.research.google.com/drive/1HVmY2yG92IVbBllGE5xDI1HWjclknZVJ
- Dataset no Roboflow: https://app.roboflow.com/roberli/detector-de-gas
- Arquivos do projeto (vídeos, pesos, dataset zipado): https://drive.google.com/drive/folders/1fbmdwq0HorSvJ0A4UIHge39IE8-hVHQb
