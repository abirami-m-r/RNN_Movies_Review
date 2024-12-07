# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load the IMDB dataset word index
word_index = imdb.get_word_index()

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

## Save model file
model = load_model('simple_rnn_imdb.h5')
model.summary()

def decode_review(encoded):
    review = [reverse_word_index.get(i - 3,'?') for i in encoded]
    print(review)

##Step 2: # Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3  for word in words] ##Based on [reverse_word_index.get(i - 3, '?') for i in encoded_review]
    decode_review(encoded_review)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

### Prediction  function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]


# Step 4: User Input and Prediction Using STREAMLIT

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify as + or -:')

#user input
u_ip = st.text_area('Movie Review')

if st.button('Classify'):
    sentiment,score=predict_sentiment(u_ip)
    st.write(f'Review: {u_ip}')
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')

else:
    st.write('Please Enter a movie review')


##streamlit run main.py