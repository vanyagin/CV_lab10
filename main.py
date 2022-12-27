import cv2
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from glob import glob
from tqdm import tqdm

def ModelTrain(n):
    descriptors = []
    for file in tqdm(glob('data500/*')):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        d = sift.detectAndCompute(img, None)[1]
        if d is None:
            continue
        descriptors += list(d)
    descriptors = np.array(descriptors)
    print('Training:')
    model = KMeans(init="k-means++", n_clusters=n, n_init=4)
    model.fit(descriptors)
    print('Saving:')
    with open('model.pickle', 'wb') as f:
        pickle.dump((model, n), f)
    print('Done!')


def ToVector(img, model, n):
    sift = cv2.SIFT_create()
    d = sift.detectAndCompute(img, None)[1]
    if d is None:
        return None
    classes = model.predict(d)
    hist = np.histogram(classes, n, density=True)[0]
    return hist

def DBCreate():
    with open('model.pickle', 'rb') as f:
        model, n = pickle.load(f)
    paths = []
    vectors = []
    for path in tqdm(glob('voc2012/*')):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        vec = ToVector(img, model, n)
        if vec is None:
            continue
        paths.append(path)
        vectors.append(vec)
    print('Saving:')
    df = pd.DataFrame({'path': paths, 'vec': vectors})
    df.to_pickle('db.pickle')
    print('Done!')


@st.cache(allow_output_mutation=True)
def load_db():
    df = pd.read_pickle('db.pickle')
    nbrs = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
    nbrs.fit(np.stack(df['vec'].to_numpy()))
    return df, nbrs

@st.cache(allow_output_mutation=True)
def load_model():
    with open('model.pickle', 'rb') as f:
        return pickle.load(f)


db, neighbours = load_db()
model, n = load_model()
uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg', 'webp', 'tiff'])
if uploaded_file is not None:
    file_arr = np.frombuffer(uploaded_file.getbuffer(), dtype='uint8')
    img = cv2.imdecode(file_arr, cv2.IMREAD_GRAYSCALE)
    vec = ToVector(img, model, n)
    indices = neighbours.kneighbors(vec.reshape(1, -1), return_distance=False)[0]
    paths = np.hstack(db.loc[indices, ['path']].values)
    for path in paths:
        st.image(path, caption=path)
