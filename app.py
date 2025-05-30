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

# Download and load the embedding.pkl from Google Drive
file_id = '1LdBa-zfv3LmufXZA87MRWHf9BgCQHzpz'
url = f'https://drive.google.com/uc?id={file_id}'

# Output file name
output = 'embedding.pkl'

# Download file
gdown.download(url, output, quiet=False)

# Load the pickle file
with open(output, 'rb') as f:
    feature_list = pickle.load(f)


st.title("Which Bollywood Celebrity Do You Look Like???")

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = pickle.load(open('embedding.pkl', 'rb'))
feature_list = np.array(feature_list)
filenames = pickle.load(open('filenames.pkl', 'rb'))


def save_your_image(your_image):
    try:
        with open(os.path.join('uploads',your_image.name), 'wb') as f:
            f.write(your_image.getbuffer())
        return True
    except:
        return False


def extract_features(img_path, model):

    detector = MTCNN()
    test_image = img_path
    sample_img = cv2.imread(test_image)
    result = detector.detect_faces(sample_img)
    x, y, width, height = result[0]['box']
    face = sample_img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    return result

def recommend(feature_list, feature):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(feature.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos



your_image = st.file_uploader("Choose your image :)")

if your_image is not None:
    if save_your_image(your_image):
        display_image = Image.open(your_image)
        features = extract_features(os.path.join('uploads', your_image.name), model)
        index_pos = recommend(feature_list,features)
        display_output = Image.open(filenames[index_pos])

        col1, col2 = st.columns(2)
        with col1:
            st.header("Your uploaded image")
            st.image(display_image )

        with col2:
            celebrity = " ".join(filenames[index_pos].split('\\')[1].split('_'))
            st.header(f"You look like {celebrity}")
            st.image(display_output)

