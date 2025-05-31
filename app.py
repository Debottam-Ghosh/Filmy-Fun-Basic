import pickle
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import os
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
import gdown

st.title("Which Bollywood Celebrity Do You Look Like???")

def save_your_image(your_image):
    try:
        with open(os.path.join('uploads', your_image.name), 'wb') as f:
            f.write(your_image.getbuffer())
        return True
    except:
        return False

@st.cache_resource
def load_model_and_detector():
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    detector = MTCNN()
    return model, detector

@st.cache_resource
def extract_features(img_path):
    model, detector = load_model_and_detector()
    sample_img = cv2.imread(img_path)
    result = detector.detect_faces(sample_img)
    if not result:
        return None
    x, y, width, height = result[0]['box']
    face = sample_img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def load_embeddings():
    if "feature_list" not in st.session_state:
        file_id = '1LdBa-zfv3LmufXZA87MRWHf9BgCQHzpz'
        url = f'https://drive.google.com/uc?id={file_id}'
        output = 'embedding.pkl'
        gdown.download(url, output, quiet=False)
        with open(output, 'rb') as f:
            st.session_state.feature_list = pickle.load(f)
    return st.session_state.feature_list

@st.cache_data
def load_filenames():
    with open('filenames.pkl', 'rb') as f:
        return pickle.load(f)

def recommend(feature_list, feature):
    similarity = [
        cosine_similarity(feature.reshape(1, -1), feat.reshape(1, -1))[0][0]
        for feat in feature_list
    ]
    index_pos = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)[0][0]
    return index_pos

your_image = st.file_uploader("Choose your image :)")

if your_image is not None:
    if save_your_image(your_image):
        display_image = Image.open(your_image)
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
                celebrity = " ".join(filenames[index_pos].split('\\')[1].split('_'))
                st.header(f"You look like {celebrity}")
                st.image(display_output)
