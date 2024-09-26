import nltk
# nltk.download('all')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
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
import pyperclip


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


st.write("This app predicts the sentiment of text based reviews.")

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
        'The company provides a dynamic and innovative work environment that encourages creativity and continuous improvement. Employees are constantly challenged with exciting projects that push them to think outside the box and develop new skills. There’s a strong focus on fostering a culture of collaboration, where team members work together to achieve shared goals.',
        'The office provides a stable working environment, but career growth opportunities are limited.',
        'Work-life balance is acceptable, though the workload can vary based on the project.',        
        'Management is okay, not too strict, but not particularly motivating either.',
        'The benefits are average, nothing exceptional but sufficient for basic needs.',
        'The office culture is professional, but there’s little in terms of team bonding activities.',
        'Great place to work with excellent career development opportunities and supportive management.',
        'The benefits package is fantastic, and the company truly cares about employee well-being.',
        'The leadership team is inspiring, and there are always new challenges to keep you engaged.',
        'Amazing work-life balance, and the flexibility to work remotely is a huge bonus.',
        'The company provides a dynamic and innovative work environment with plenty of growth potential.',
        'Toxic environment with no opportunities for advancement in technology, worst training provided at the time of joining this organization. Employees are overworked and completely undervalued by the company.',
        'Worst experience ever in this company. No career growth, no raises, and zero employee development over the last five years.',
        'Awful management, no promotions, no career growth, and a lack of respect for employee contributions. Avoid at all costs.',
        'An amazing company with a supportive work environment, excellent growth opportunities, and a great work-life balance. The leadership is approachable, and the team culture fosters collaboration and innovation. Overall, its a fantastic place to build a rewarding career.'
    ]
}

df = pd.DataFrame(data)
st.markdown(
    """
    <style>
    .custom-box {
        background-color: #f5f5f5;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        color: black;
        word-wrap: break-word;
        white-space: normal;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("Sample Reviews For Testing:")

for index, row in df.iterrows():
    with st.container():
        col1,col2 = st.columns([3,1])
        # col2.write(row['Sentiment'])
        with col1:
            st.code(row['Reviews'], language='')
            st.markdown(f"<div class='custom-box'>{row['Reviews']}</div>", unsafe_allow_html=True)

