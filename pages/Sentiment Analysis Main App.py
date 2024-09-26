import nltk
# nltk.download('all')
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model


st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊")

st.markdown(
    """
    <style>
    .main {
        background-color: #dce2e3;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = load_model('./model_GRU.h5')

with open('./tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def data_processing(review):
    review = review.lower()
    review = re.sub(r"https\S+|www\S+https\S+", '', review, flags=re.MULTILINE)  
    review = re.sub(r'\@w+|\#','', review) 
    review = re.sub(r'[^\w\s]', '', review) 
    text_tokens = word_tokenize(review) 
    filtered_text = [w for w in text_tokens if not w in stop_words] 
    stemmed_text = [stemmer.stem(word) for word in filtered_text] 
    return " ".join(stemmed_text)


def preprocess_input(text):
    processed_review = data_processing(text)  
    sequences = tokenizer.texts_to_sequences([processed_review])  
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='pre', truncating='pre')  
    return padded_sequences


def get_sentiment_label(prediction):
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return labels[np.argmax(prediction)]


column1, column2 = st.columns([1, 6])  
with column1:
    st.image("sentimentlogo.png", width=100)
with column2:
    st.title("Sentiment Analysis App")


st.write("This app predicts the sentiment of text based on different sources.")

user_input = st.text_area("Please Write a Review:","")

if st.button("Predict Sentiment"):
    if user_input:
        processed_input = preprocess_input(user_input)  
        prediction = model.predict(processed_input) 
        sentiment = get_sentiment_label(prediction)  
        if sentiment == "Positive":
            st.success(f"The sentiment is {sentiment} 😊")
        elif sentiment == "Negative":
            st.error(f"The sentiment is {sentiment} 😞")
        elif sentiment == "Neutral":
            st.info(f"The sentiment is {sentiment} 😐")
        else:
            st.warning(f"Received unexpected sentiment: {sentiment}")
    else:
        st.warning("Please enter a review.")


data = {
    'Reviews': [
        'This product is amazing!',
        'Not what I expected.',
        'Satisfied with the quality.',
        'Could be better.',
        'Worst experience ever.'
    ]
}

df = pd.DataFrame(data)

st.subheader("Sample Reviews For Testing:")

for index, row in df.iterrows():
    with st.container():
        col1,col2 = st.columns([3,1])
        # col2.write(row['Sentiment'])
        with col1:
            st.code(row['Reviews'], language='')

