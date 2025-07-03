import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

import streamlit as st
import tensorflow as tf
import numpy as np
import gzip
import pickle
import re
import snscrape.modules.twitter as sntwitter

# ================== Load Model dan Tokenizer ==================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/sentiment_bilstm_cnn.h5")

@st.cache_data
def load_tokenizer():
    tokenizer_path = 'model/tokenizer.pkl.gz'
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer tidak ditemukan di: {tokenizer_path}")
    
    with gzip.open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    if not hasattr(tokenizer, 'texts_to_sequences'):
        raise ValueError("File tokenizer tidak valid.")
    
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()
MAX_LEN = 100  # Ubah sesuai saat training

# ================== Fungsi Preprocessing ==================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@[\w]+', '', text)
    text = re.sub(r'\#[\w]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ================== Fungsi Prediksi ==================

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0]

    labels = ["Negative", "Neutral", "Positive"]
    sentiment = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return sentiment, confidence

# ================== Fungsi Ambil Tweet dari URL ==================

def get_tweet_text_from_url(url):
    try:
        tweet_id = url.strip().split("/")[-1]
        scraper = sntwitter.TwitterTweetScraper(tweet_id)
        tweet = next(scraper.get_items())  # ambil tweet pertama dari generator
        return tweet.content
    except Exception as e:
        return f"[ERROR] Gagal ambil tweet: {e}"

# ================== UI Streamlit ==================

st.set_page_config(page_title="Tweetlyze", layout="centered")
st.title("💬 Tweetlyze - Analisis Sentimen Twitter")

# --- Input manual teks ---
st.subheader("✏ Analisis Teks Langsung")
user_input = st.text_input("Masukkan teks tweet atau opini:")

if st.button("🔍 Analisis Teks"):
    if user_input:
        cleaned_input = clean_text(user_input)
        sentiment, confidence = predict_sentiment(cleaned_input)
        st.success(f"Sentimen: {sentiment} ({confidence:.2f}%)")
    else:
        st.warning("Masukkan teks terlebih dahulu.")

st.markdown("---")

# --- Input URL tweet ---
st.subheader("🔗 Analisis Sentimen dari URL Tweet")
tweet_url = st.text_input("Masukkan URL Tweet:")

if st.button("🌐 Ambil & Analisis URL"):
    if tweet_url:
        tweet_text = get_tweet_text_from_url(tweet_url)
        if "[ERROR]" in tweet_text:
            st.error(tweet_text)
        else:
            cleaned_tweet = clean_text(tweet_text)
            st.markdown(f"Teks dari Tweet: {tweet_text}")
            sentiment, confidence = predict_sentiment(cleaned_tweet)
            st.success(f"Sentimen: {sentiment} ({confidence:.2f}%)")
    else:
        st.warning("Masukkan URL tweet terlebih dahulu.")