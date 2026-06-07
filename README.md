# 🛰️ SafeOrbit — Classificação de Terreno por Visão Computacional

> Componente de **Visão Computacional (ACV)** do projeto integrado **SafeOrbit** —
> Global Solution · Indústria Espacial · FIAP.

Classificação de imagens de satélite (Sentinel-2) para identificar o **tipo de terreno**
onde um sinal de SOS foi disparado, orientando a equipe de resgate antes de partir.
Duas redes neurais convolucionais (CNNs) **criadas do zero**, comparadas entre si.

---

## 👥 Integrantes

| Nome | RM | Turma |
|------|----|-------|
| Breno Silva Rubio | RM97864 | 4ESR |
| Enrico Marquez | RM99325 | 4ESR |
| Gustavo Dias | RM550820 | 4ESR |
| Joel Barros | RM550378 | 4ESR |
| Leonardo Moreira | RM550988 | 4ESR |

🎥 **Vídeo de apresentação (até 3 min):https://youtu.be/2TBT9USoxR4

---

## 1. Problema e conexão com a Indústria Espacial

O **SafeOrbit** é uma plataforma de coordenação de resgate em áreas remotas via
constelação de satélites LEO. Quando alguém em perigo dispara um sinal de SOS, um
satélite retransmite a localização ao centro de operações. **Saber o tipo de terreno**
da coordenada muda a estratégia: helicóptero, barco, trilha a pé ou apoio de moradores.

**Tarefa de Visão Computacional:** dada uma imagem de satélite (tile RGB) da localização,
**classificar o terreno** entre 10 classes de cobertura do solo.

**Dado orbital usado:** imagens reais do satélite **Sentinel-2** (programa Copernicus/ESA),
via dataset **EuroSAT** — exatamente o tipo de dado orbital aberto que o desafio sugere.

**ODS atendidos:** 9 (Inovação e infraestrutura), 11 (Comunidades), 3 (salvar vidas), 10 (redução de desigualdades).

## 2. Dataset — EuroSAT (Sentinel-2)

- **27.000** imagens RGB de **64×64 px**, **10 classes** de terreno.
- Origem: imagens orbitais do **Sentinel-2** (ESA/Copernicus), rotuladas. Licença aberta.
- Baixado da **Hugging Face** (`blanchon/EuroSAT_RGB`) — espelho confiável, já com splits.
  > ⚠️ Não usamos `tensorflow_datasets` porque o servidor original do EuroSAT
  > (`madm.dfki.de`) costuma retornar **HTTP 403** e quebra o download.
- **Divisão (predefinida):** ≈ **16.200 treino · 5.400 validação · 5.400 teste** (~60/20/20).
- **Pré-processamento:** normalização dos pixels para `[0, 1]`; `class_weight` para o leve desbalanceamento.

**Classes (ordem padrão EuroSAT):** Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, SeaLake.

## 3. Arquiteturas (CNNs do zero)

Definidas em [`src/models.py`](src/models.py). **Nenhum modelo pré-treinado.**

| | Modelo A — Baseline | Modelo B — Aprimorada |
|---|---|---|
| Blocos conv | 3 × (Conv + MaxPool) | 3 × (2×Conv + BN + MaxPool + Dropout) |
| Cabeça | Flatten + Dense | GlobalAveragePooling + Dense |
| Regularização | nenhuma | BatchNorm + Dropout + Data Augmentation |
| Parâmetros | ~1,14 M | ~0,33 M |

> Curiosidade: o Modelo A tem **mais** parâmetros (por causa do `Flatten`), mas o B
> generaliza melhor — mostra que **arquitetura importa mais que tamanho bruto**.


## 4. Resultados

| Modelo | Acurácia (teste) | Observação |
|--------|------------------|------------|
| A — Baseline   | 83,4% | sofre overfitting (treino ~98% ≫ validação ~84%) |
| B — Aprimorada | **94,3%** | melhor generalização; **modelo escolhido** |

Meta de referência: **88%** — superada pelo Modelo B com folga. As matrizes de confusão
e os exemplos de erro estão no notebook; os poucos erros do Modelo B concentram-se em
classes visualmente parecidas (ex.: `Industrial Buildings` × `Residential Buildings` e a
família `Annual Crop` / `Permanent Crop` / `Herbaceous Vegetation`).

## 5. Como executar

### Opção A — Treinar no Google Colab (recomendado)
1. Abra `notebook/safeorbit_cnn.ipynb` no [Google Colab](https://colab.research.google.com).
2. `Ambiente de execução → Alterar tipo → GPU`.
3. Execute todas as células (`Ambiente de execução → Executar tudo`). O dataset é baixado
   automaticamente da Hugging Face (~174 MB, ~1-2 min).
4. Baixe `models/melhor_modelo.keras`, `models/class_names.json` e `sample_images/`.

### Opção B — Demonstração local (Streamlit)
```bash
pip install -r requirements.txt
# coloque melhor_modelo.keras e class_names.json em models/
streamlit run app/streamlit_app.py
```
Envie uma imagem (use as de `sample_images/`) → o app mostra o terreno + orientação de resgate.

## 6. Estrutura do repositório
```
safeorbit-acv/
├── README.md
├── requirements.txt
├── notebook/
│   └── safeorbit_cnn.ipynb     # treino + avaliação + comparação (deliverable principal)
├── src/
│   └── models.py               # arquiteturas das 2 CNNs (do zero)
├── app/
│   └── streamlit_app.py        # demonstração funcional
├── models/                     # melhor_modelo.keras + class_names.json (após treinar)
└── sample_images/              # amostras de teste (após treinar)
```

## 7. Limitações e melhorias futuras
- Resolução de 64×64 limita detalhes; EuroSAT cobre a Europa (um dataset Sentinel local seria ideal).
- Futuro: usar as bandas multiespectrais (não só RGB) e maior resolução.

---
_Projeto acadêmico — FIAP / Global Solution._
