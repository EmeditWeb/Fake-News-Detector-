# fake_news_detector_app.py

import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
import pandas as pd

# --- Streamlit Page Configuration (for sleek design and wider layout) ---
st.set_page_config(
    page_title="Fake News Detector | Emeditweb",
    page_icon="📰", # A newspaper emoji as the favicon
    layout="wide", # Use a wide layout for better visual appeal
    initial_sidebar_state="collapsed" # Optionally hide sidebar by default
)

# --- Download NLTK data (only if not already downloaded by Streamlit Cloud) ---
@st.cache_resource # Cache NLTK downloads to prevent re-downloading on every rerun
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet')

download_nltk_data()

# --- Load the trained model and TF-IDF Vectorizer ---
# These files should be placed in the same directory as this Streamlit app file
@st.cache_resource # Cache the model and vectorizer loading for performance
def load_resources():
    try:
        model = joblib.load('linearsvc_fake_news_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"**Error:** Model or Vectorizer file not found. "
                 f"Please ensure `linearsvc_fake_news_model.joblib` and `tfidf_vectorizer.joblib` "
                 f"are in the same directory as the app. 💾")
        return None, None

model, tfidf_vectorizer = load_resources()

# --- Preprocessing functions (IDENTICAL to training) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def normalize_whitespace_and_strip(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def convert_to_lowercase(text):
    if isinstance(text, str):
        return text.lower()
    return text

def remove_punctuation(text):
    if isinstance(text, str):
        return re.sub(r'[^\w\s]', '', text)
    return text

def remove_numbers(text):
    if isinstance(text, str):
        return re.sub(r'\d+', '', text)
    return text

def remove_stopwords_and_tokenize_alpha(text):
    if isinstance(text, str):
        tokens = [word for word in text.split() if word.isalpha() and word not in stop_words]
        return ' '.join(tokens)
    return text

def lemmatize_text(text):
    if isinstance(text, str):
        tokens = text.split()
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmas)
    return text

def preprocess_text_for_prediction(text):
    # Apply the same cleaning steps in order
    text = normalize_whitespace_and_strip(text)
    text = convert_to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords_and_tokenize_alpha(text)
    text = lemmatize_text(text)
    text = normalize_whitespace_and_strip(text) # Final cleanup
    return text

# --- Streamlit App Layout ---

st.title("📰 **AI-Powered Fake News Detector**")
st.markdown("""
    <p style='font-size: 1.1em; color: #666; text-align: center;'>
        Leveraging Natural Language Processing to identify misinformation.
        Enter a news article's title and/or content below to get an instant prediction.
    </p>
    """, unsafe_allow_html=True)

st.divider() # A subtle separator

# fake_news_detector_app.py

import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
import pandas as pd

# --- Streamlit Page Configuration (for sleek design and wider layout) ---
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰", # A newspaper emoji as the favicon
    layout="wide", # Use a wide layout for better visual appeal
    initial_sidebar_state="collapsed" # Optionally hide sidebar by default
)

# --- Download NLTK data (only if not already downloaded by Streamlit Cloud) ---
@st.cache_resource # Cache NLTK downloads to prevent re-downloading on every rerun
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet')

download_nltk_data()

# --- Load the trained model and TF-IDF Vectorizer ---
# These files should be placed in the same directory as this Streamlit app file
@st.cache_resource # Cache the model and vectorizer loading for performance
def load_resources():
    try:
        model = joblib.load('linearsvc_fake_news_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"**Error:** Model or Vectorizer file not found. "
                 f"Please ensure `linearsvc_fake_news_model.joblib` and `tfidf_vectorizer.joblib` "
                 f"are in the same directory as the app. 💾")
        return None, None

model, tfidf_vectorizer = load_resources()

# --- Preprocessing functions (IDENTICAL to training) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def normalize_whitespace_and_strip(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def convert_to_lowercase(text):
    if isinstance(text, str):
        return text.lower()
    return text

def remove_punctuation(text):
    if isinstance(text, str):
        return re.sub(r'[^\w\s]', '', text)
    return text

def remove_numbers(text):
    if isinstance(text, str):
        return re.sub(r'\d+', '', text)
    return text

def remove_stopwords_and_tokenize_alpha(text):
    if isinstance(text, str):
        tokens = [word for word in text.split() if word.isalpha() and word not in stop_words]
        return ' '.join(tokens)
    return text

def lemmatize_text(text):
    if isinstance(text, str):
        tokens = text.split()
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmas)
    return text

def preprocess_text_for_prediction(text):
    # Apply the same cleaning steps in order
    text = normalize_whitespace_and_strip(text)
    text = convert_to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords_and_tokenize_alpha(text)
    text = lemmatize_text(text)
    text = normalize_whitespace_and_strip(text) # Final cleanup
    return text

# --- Streamlit App Layout ---

st.title("📰 **AI-Powered Fake News Detector**")
st.markdown("""
    <p style='font-size: 1.1em; color: #666; text-align: center;'>
        Leveraging Natural Language Processing to identify misinformation.
        Enter a news article's title and/or content below to get an instant prediction.
    </p>
    """, unsafe_allow_html=True)

st.divider() # A subtle separator

# Input Form
with st.container(border=True): # Use a container with a border for a cleaner look
    st.subheader("📝 Article Input")
    title_input = st.text_input("Article Title (optional)",
                                 placeholder="e.g., 'Breaking News: AI takes over the world!'")
    text_input = st.text_area("Article Content",
                              height=250,
                              placeholder="Paste the full news article content here...",
                              help="The more content, the better the prediction accuracy.")

    predict_button_col, _ = st.columns([1, 4]) # Use columns for button alignment
    with predict_button_col:
        predict_button = st.button("✨ **Get Prediction** ✨", use_container_width=True)


# --- Rolling Word Part / Undergoing Training Disclaimer ---
st.markdown("---") # Another separator for clarity
st.warning("""
    <div style="width: 100%; overflow: hidden;">
        <marquee behavior="scroll" direction="left" scrollamount="4">
            **Important Note:** This model is currently undergoing continuous training and optimization.
            While highly accurate, results could occasionally be incorrect. Use with discretion. 🛠️
        </marquee>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---") # And another one

# --- Prediction Logic ---
if predict_button:
    if not model or not tfidf_vectorizer:
        st.error("Cannot make predictions. Model or Vectorizer resources are not loaded. "
                 "Please ensure files are present in the GitHub repository.")
    elif not title_input and not text_input:
        st.warning("🧐 Please enter either a **title** or **article content** to make a prediction.")
    else:
        with st.spinner("Analyzing article..."):
            # Combine title and text just like in training
            full_text_combined = f"{title_input} {text_input}".strip()

            if not full_text_combined:
                st.warning("After combining, the input text is empty. Please provide more substantial text.")
            else:
                # Preprocess the combined text
                processed_text = preprocess_text_for_prediction(full_text_combined)

                if not processed_text:
                    st.warning("After preprocessing, the input text became empty. "
                               "This might happen if the text contained only numbers, punctuation, or stopwords. "
                               "Please try different text. 🤷‍♀️")
                else:
                    # Transform using the *trained* TF-IDF vectorizer
                    text_vectorized = tfidf_vectorizer.transform([processed_text])

                    # Make prediction
                    prediction = model.predict(text_vectorized)
                    # For LinearSVC, higher decision_function implies higher confidence in the positive class (class_1)
                    # Assuming 1 is FAKE, 0 is REAL based on common binary classification setups.
                    decision_score = model.decision_function(text_vectorized)[0]

                    st.subheader("🔮 Prediction Result:")
                    if prediction[0] == 1: # Assuming 1 = FAKE
                        st.error(f"## 🚨 This article is **FAKE NEWS!** 🚨")
                        st.markdown(f"<p style='font-size: 1.1em; color: red;'>Confidence (decision value): <strong>{decision_score:.2f}</strong></p>", unsafe_allow_html=True)
                        st.balloons() # Add some celebratory balloons for detecting fake news (ironic, but fun)
                    else: # Assuming 0 = REAL
                        st.success(f"## ✅ This article is **REAL NEWS!** ✅")
                        st.markdown(f"<p style='font-size: 1.1em; color: green;'>Confidence (decision value): <strong>{decision_score:.2f}</strong></p>", unsafe_allow_html=True)

                    st.markdown("---")
                    st.info("💡 **Note on Confidence:** For LinearSVC, the 'confidence' is represented by a raw decision function value, not a probability. A higher absolute value indicates stronger confidence.")

# --- Footer or additional info ---
st.markdown("""
    <br><br><br>
    <div style='text-align: center; font-size: 0.9em; color: #888;'>
        Powered by Streamlit & Scikit-learn. <br>
        Developed as an NLP project.
    </div>
    """, unsafe_allow_html=True)￼Enter# Input Form
with st.container(border=True): # Use a container with a border for a cleaner look
    st.subheader("📝 Article Input")
    title_input = st.text_input("Article Title (optional)",
                                 placeholder="e.g., 'Breaking News: AI takes over the world!'")
    text_input = st.text_area("Article Content",
                              height=250,
                              placeholder="Paste the full news article content here...",
                              help="The more content, the better the prediction accuracy.")

    predict_button_col, _ = st.columns([1, 4]) # Use columns for button alignment
    with predict_button_col:
        predict_button = st.button("✨ **Get Prediction** ✨", use_container_width=True)


# --- Rolling Word Part / Undergoing Training Disclaimer ---
st.markdown("---") # Another separator for clarity
st.warning("""
    <div style="width: 100%; overflow: hidden;">
        <marquee behavior="scroll" direction="left" scrollamount="4">
            **Important Note:** This model is currently undergoing continuous training and optimization.
            While highly accurate, results could occasionally be incorrect. Use with discretion. 🛠️
        </marquee>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---") # And another one
 Prediction Logic ---
if predict_button:
    if not model or not tfidf_vectorizer:
        st.error("Cannot make predictions. Model or Vectorizer resources are not loaded. "
                 "Please ensure files are present in the GitHub repository.")
    elif not title_input and not text_input:
        st.warning("🧐 Please enter either a **title** or **article content** to make a prediction.")
    else:
        with st.spinner("Analyzing article..."):
            # Combine title and text just like in training
            full_text_combined = f"{title_input} {text_input}".strip()

            if not full_text_combined:
                st.warning("After combining, the input text is empty. Please provide more substantial text.")
            else:
                # Preprocess the combined text
                processed_text = preprocess_text_for_prediction(full_text_combined)

                if not processed_text:
                    st.warning("After preprocessing, the input text became empty. "
                               "This might happen if the text contained only numbers, punctuation, or stopwords. "
                               "Please try different text. 🤷‍♀️")
                else:
                    # Transform using the *trained* TF-IDF vectorizer
                    text_vectorized = tfidf_vectorizer.transform([processed_text])
