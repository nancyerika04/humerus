import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_my_model():
    return load_model("mobilenetv2_humerus_model.h5")

model = load_my_model()

# Fonction de prédiction
def predict(image, model):
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)  # Ajuster la taille
    image_array = np.asarray(image) / 255.0  # Normaliser les pixels
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension batch
    prediction = model.predict(image_array)
    return prediction[0][0]  # Retourne la probabilité

# Interface utilisateur Streamlit
st.title("Détection de fractures de l'humérus")
st.write("Téléchargez une image pour prédire si elle contient une fracture.")

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée.", use_column_width=True)
    st.write("Analyse en cours...")
    prediction = predict(image, model)
    if prediction > 0.5:
        st.write(f"### Résultat : Fracture détectée (Probabilité : {prediction:.2f})")
    else:
        st.write(f"### Résultat : Aucune fracture détectée (Probabilité : {prediction:.2f})")
