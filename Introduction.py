import streamlit as st

st.set_page_config(page_title="Sentiment Analysis - Introduction", page_icon="📖")

# Set custom page style
st.markdown(
    """
    <style>
    .main {
        background-color: #fae8e1;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)



column1, column2 = st.columns([1, 6])  
with column1:
    st.image("./sentimentlogo.png", width=100)
with column2:
    st.title("Welcome to the Sentiment Analysis App")

st.subheader("Analyzing Employee Reviews for Better Insights")

# Introduction
st.write("""
This application is designed to help users analyze the sentiment of employee reviews.
Using cutting-edge deep learning techniques, we classify reviews into three distinct categories:
- Positive 😊
- Neutral 😐
- Negative 😞

Employee feedback can provide valuable insights into company culture, job satisfaction, and more.
Understanding this sentiment is crucial for employers seeking to improve their workplace environment
and for potential employees researching company dynamics.
""")

# Motivation
st.header("Why Sentiment Analysis?")
st.write("""
Sentiment analysis, also known as opinion mining, enables businesses and individuals to gauge public sentiment
towards products, services, or internal dynamics within organizations. Specifically, in the context of employee
reviews, sentiment analysis provides valuable information about:
- *Workplace satisfaction*: Understanding what employees love and what they find challenging.
- *Employer branding*: Monitoring and improving public perception based on employee feedback.
- *Retention strategies*: Identifying areas of dissatisfaction to reduce turnover.

This app is aimed at making sentiment analysis of employee feedback easier, faster, and more accurate.
""")

# Dataset Information
st.header("Dataset Used for Training")
st.write("""
The dataset used to train this model was web-scraped from *AmbitionBox*, a popular platform for employee reviews.
The reviews were collected from the top 20 companies in three major sectors:
- *IT Sector*
- *Healthcare Sector*
- *Real Estate Sector*

These reviews provide a diverse set of employee feedback, ensuring that the model can accurately analyze sentiment
across various industries. By using this real-world dataset, the model is well-equipped to handle a variety of review
styles and content.
""")

# How to Use the App
st.header("How to Use This App")
st.write("""
To get started, simply follow these steps:
1. *Enter a review*: In the text area provided, write or paste an employee review that you would like to analyze.
2. *Click the 'Predict Sentiment' button*: The app will process your input using advanced Natural Language Processing (NLP) techniques and display the predicted sentiment.
    - If the review is positive, you’ll see a success message with a 😊 emoji.
    - For neutral reviews, an informational message with a 😐 emoji will appear.
    - If the review is negative, you’ll see a warning message with a 😞 emoji.
3. *View Sample Reviews*: Below the input area, we’ve included some sample reviews for you to test the app's functionality. Feel free to try them out!

Behind the scenes, the app uses a pre-trained *GRU (Gated Recurrent Unit)* model to classify each review. This model has been trained on thousands of employee reviews and can effectively determine the sentiment based on the content.
""")

# Footer
st.write("""
Whether you're an employer wanting to gauge internal sentiment, or a job seeker looking to understand company culture,
this tool can help you gain insights into employee experiences. Get started now and see how it works for yourself!
""")
