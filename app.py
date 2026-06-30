
#Classification chien vs chat — Webapp Streamlit



import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


MODEL_PATH = "cats_dogs_model.keras"
IMG_SIZE = (224, 224)          # TL_SIZE dans le notebook
# class_indices appris : {'cats': 0, 'dogs': 1}
# => sortie sigmoid : proba proche de 1 = chien, proche de 0 = chat
THRESHOLD = 0.5

st.set_page_config(page_title="Chien ou Chat ?", page_icon="🐾", layout="centered")


@st.cache_resource
def load_model():
    """Charge le modèle une seule fois et le garde en cache."""
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(pil_img: Image.Image) -> np.ndarray:
    """Prépare une image PIL pour le modèle (même pipeline qu'à l'entraînement)."""
    # Corrige l'orientation EXIF (photos de téléphone) et force le RGB
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    pil_img = pil_img.resize(IMG_SIZE)
    arr = np.asarray(pil_img, dtype=np.float32)
    arr = preprocess_input(arr)          # <-- prétraitement MobileNetV2
    return np.expand_dims(arr, axis=0)   # ajoute la dimension batch


# --------------------------- Interface ---------------------------
st.title("🐾 Chien ou Chat ?")
st.write(
    "Chargez une image de chien ou de chat. Je vous donne ma prédiction."
)

uploaded = st.file_uploader(
    "Choisissez une image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Image chargée", use_container_width=True)

    with st.spinner("Analyse en cours..."):
        model = load_model()
        proba_dog = float(model.predict(preprocess(image), verbose=0)[0][0])

    is_dog = proba_dog > THRESHOLD
    label = "Chien" if is_dog else "Chat"
    confidence = proba_dog if is_dog else 1.0 - proba_dog

    st.subheader(f"Prédiction : **{label}**")
    
    

    # Avertissement si le modèle hésite (proba proche du seuil)
    if 0.40 < proba_dog < 0.60:
        st.warning(
            "Le modèle hésite (probabilité proche de 0,5). "
            "L'image est peut-être ambiguë, mal cadrée ou éloignée des cas vus à l'entraînement."
        )

    with st.expander("Détails techniques"):
        st.write(f"Probabilité brute (proba que ce soit un chien) : `{proba_dog:.4f}`")
        st.write(f"Seuil de décision : `{THRESHOLD}`")
        st.write(f"Taille d'entrée du modèle : `{IMG_SIZE}`")
else:
    st.info("En attente d'une image...")

st.caption("Projet — Classification d'images (dogs vs cats) · MobileNetV2")
