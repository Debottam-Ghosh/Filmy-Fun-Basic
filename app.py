import pickle
import numpy as np
import streamlit as st
from PIL import Image
import os
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import gdown

st.title("Which Bollywood Celebrity Do You Look Like???")

# Ensure 'uploads' folder exists
os.makedirs('uploads', exist_ok=True)

def save_your_image(your_image):
    try:
        with open(os.path.join('uploads', your_image.name), 'wb') as f:
            f.write(your_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Failed to save image: {e}")
        return False

@st.cache_resource
def extract_features(img_path):
    try:
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name='VGG-Face',  # Options: 'Facenet', 'ArcFace', etc.
            enforce_detection=True
        )
        return embedding[0]['embedding']
    except Exception as e:
        st.error(f"Face detection failed: {e}")
        return None

def load_embeddings():
    if "feature_list" not in st.session_state:
        if not os.path.exists('embedding.pkl'):
            file_id = '1v820bfltcX85TPnQ7OKKrxpUDI7Z8ciq'
            url = f'https://drive.google.com/uc?id={file_id}'
            output = 'embedding.pkl'
            gdown.download(url, output, quiet=False)
        with open('embedding.pkl', 'rb') as f:
            st.session_state.feature_list = pickle.load(f)
    return st.session_state.feature_list

@st.cache_data
def load_filenames():
    with open('filenames.pkl', 'rb') as f:
        return pickle.load(f)

def recommend(feature_list, feature):
    similarity = [
        cosine_similarity(np.array(feature).reshape(1, -1), np.array(feature_list).reshape(1, -1))[0][0]
        for feature_list in feature_list
    ]
    index_pos = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)[0][0]
    return index_pos

# UI
your_image = st.file_uploader("Choose your image :)")

if your_image is not None:
    if save_your_image(your_image):
        display_image = Image.open(your_image)
        with st.spinner("Analyzing your face..."):
            features = extract_features(os.path.join('uploads', your_image.name))

        if features is None:
            st.error("No face detected in the image.")
        else:
            feature_list = load_embeddings()
            filenames = load_filenames()
            index_pos = recommend(feature_list, features)
            display_output = Image.open(filenames[index_pos])

            col1, col2 = st.columns(2)
            with col1:
                st.header("Your uploaded image")
                st.image(display_image)

            with col2:

                celebrity = " ".join(os.path.basename(filenames[index_pos]).split('_'))
                st.header(f"You look like {celebrity}")
                st.image(display_output)
