"""
SafeOrbit — Demonstração funcional (Streamlit)
==============================================
App simples: o usuário envia uma imagem de satélite (tile RGB) e o modelo
classifica o tipo de terreno, devolvendo a orientação operacional de resgate.

Como rodar localmente:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

Pré-requisitos:
    models/melhor_modelo.keras   (gerado pelo notebook)
    models/class_names.json
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 64
MODEL_PATH = "models/melhor_modelo.keras"
CLASSES_PATH = "models/class_names.json"

# Tradução + contexto operacional de resgate, por ÍNDICE (ordem padrão do EuroSAT)
INFO_PT = [
    ("Lavoura anual",       "Área agrícola aberta. Acesso por estradas rurais; bom para pouso de helicóptero."),  # 0
    ("Floresta",            "Vegetação densa. Acesso terrestre difícil; considerar resgate aéreo."),              # 1
    ("Vegetação herbácea",  "Campo aberto/arbustos. Acesso moderado a pé."),                                      # 2
    ("Rodovia",             "Acesso rodoviário rápido por veículo terrestre."),                                   # 3
    ("Área industrial",     "Área construída. Possível apoio logístico e estrutura próxima."),                    # 4
    ("Pastagem",            "Área aberta e plana. Excelente para pouso de helicóptero."),                         # 5
    ("Cultivo permanente",  "Plantação (pomares/vinhedos). Acesso por vias rurais."),                             # 6
    ("Área residencial",    "Local povoado. Apoio de moradores e serviços locais disponível."),                   # 7
    ("Rio",                 "Curso d'água. Risco de correnteza; possível acesso por embarcação."),                # 8
    ("Mar/Lago",            "Corpo d'água extenso. Resgate aquático (barco/mergulho)."),                          # 9
]


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


st.set_page_config(page_title="SafeOrbit — Terreno", page_icon="🛰️", layout="centered")

st.title("🛰️ SafeOrbit — Classificação de Terreno")
st.caption(
    "Coordenação de resgate em áreas remotas via satélite. "
    "Envie a imagem de satélite da localização do sinal de SOS."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Modelo não encontrado em `{MODEL_PATH}`.\n\n"
        "Rode o notebook `notebook/safeorbit_cnn.ipynb` primeiro e coloque "
        "`melhor_modelo.keras` e `class_names.json` na pasta `models/`."
    )
    st.stop()

model, CLASS_NAMES = load_model_and_classes()

arquivo = st.file_uploader(
    "Imagem de satélite (RGB)", type=["png", "jpg", "jpeg"]
)

if arquivo is not None:
    img = Image.open(arquivo).convert("RGB")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Imagem recebida", use_container_width=True)

    # Pré-processamento idêntico ao treino
    arr = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)), dtype="float32") / 255.0
    prob = model.predict(arr[None, ...], verbose=0)[0]
    idx = int(prob.argmax())
    nome_pt, contexto = INFO_PT[idx] if idx < len(INFO_PT) else (CLASS_NAMES[idx], "—")

    with col2:
        st.metric("Terreno detectado", nome_pt, f"{prob[idx] * 100:.1f}% de confiança")
        st.info(f"🚁 **Orientação à equipe:** {contexto}")

    st.subheader("Probabilidade por classe")
    ordem = np.argsort(prob)[::-1]
    st.bar_chart(
        {(INFO_PT[i][0] if i < len(INFO_PT) else CLASS_NAMES[i]): float(prob[i]) for i in ordem}
    )
else:
    st.write("⬆️ Envie uma imagem para começar. Você pode usar as amostras de `sample_images/`.")
