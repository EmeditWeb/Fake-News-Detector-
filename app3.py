# fake_news_detector_app.py

import streamlit as st
import joblib
import re
# Removed 'time' import as it's no longer needed in setup_nltk_data
import nltk
import os
import pandas as pd # Included for general ML project context, though not directly used in the app's core logic

# --- Streamlit Page Configuration (for sleek design and wider layout) ---
st.set_page_config(
    page_title="Fake News Detector | EmeditWeb",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Configure NLTK data path for pre-packaged data ---
# @st.cache_resource caches the result of this function, so NLTK data path
# is set and verified only once across all user sessions and app reruns.
@st.cache_resource
def setup_nltk_data():
    # Define the path where NLTK data is expected to be in the deployed repo.
    # In Streamlit Cloud, os.getcwd() points to the root of your cloned repository.
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')

    # Add this path to NLTK's data search path, ensuring it's at the beginning.
    # This is crucial for NLTK to find the data you pre-packaged.
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)

    st.info(f"NLTK data path configured to: `{nltk_data_path}`. Verifying pre-packaged data...")

    try:
        # Just verify that the resources are now findable from the pre-packaged data.
        # These lines will raise a LookupError if the data is not found locally
        # within the expected path.
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')

        st.success("NLTK data (stopwords, wordnet) found from pre-packaged repository! ✅")
    except LookupError as e: # Catch specifically NLTK's LookupError for missing resources
        st.error(f"**CRITICAL ERROR:** Required NLTK data (stopwords/wordnet) not found in the repository at `{nltk_data_path}`. "
                 f"This means the 'nltk_data' folder was not correctly added to your GitHub repo, "
                 f"or Git LFS is not configured properly. Error: {e}")
        st.stop() # Stop the app if data is missing, as it's essential for preprocessing.
    except Exception as e: # Catch any other unexpected errors during setup
        st.error(f"An unexpected error occurred while setting up NLTK data: {e}")
        st.stop()

# Call the NLTK setup function very early in the script,
# before any NLTK modules that rely on downloaded data are initialized.
setup_nltk_data()

# --- Now it's safe to import specific NLTK modules that rely on downloaded data ---
# These imports MUST be AFTER setup_nltk_data() to ensure the data path is set.
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Load the trained model and TF-IDF Vectorizer ---
# @st.cache_resource ensures the model and vectorizer are loaded only once
# when the app starts, not on every user interaction or app rerun.
MODEL_PATH = 'linearsvc_fake_news_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'

@st.cache_resource
def load_ml_resources():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"**Error:** Model or Vectorizer file not found. "
                 f"Please ensure '{MODEL_PATH}' and '{VECTORIZER_PATH}' are in the same directory as the app. 💾")
        st.stop() # Stop the app if crucial ML resources are missing
        # return None, None # Should not be reached due to st.stop()

model, tfidf_vectorizer = load_ml_resources()

# --- Preprocessing functions (IDENTICAL to training) ---
# These functions must exactly mirror the preprocessing applied during model training.
# Initialize NLTK components *after* data is ensured to be present
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
        # Remove all punctuation using a regex that matches non-word, non-space characters
        return re.sub(r'[^\w\s]', '', text)
    return text

def remove_numbers(text):
    if isinstance(text, str):
        return re.sub(r'\d+', '', text)
    return text

def remove_stopwords_and_tokenize_alpha(text):
    if isinstance(text, str):
        # Tokenize by splitting words, convert to lowercase, and remove non-alphabetic tokens and stopwords
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
    # Apply the same cleaning steps in order as in training
    text = normalize_whitespace_and_strip(text)
    text = convert_to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords_and_tokenize_alpha(text)
    text = lemmatize_text(text)
    text = normalize_whitespace_and_strip(text) # Final cleanup for any extra spaces
    return text

# --- Streamlit App Layout ---
st.title("📰 **AI-Powered Fake News Detector**")
st.markdown("""
    <p style='font-size: 1.1em; color: #666; text-align: center; margin-bottom: 20px;'>
        Leveraging Natural Language Processing to identify misinformation.
        Enter a news article's title and/or content below to get an instant prediction.
    </p>
    """, unsafe_allow_html=True)

st.divider() # A subtle separator

# Input Form Section
# Using st.container with border for a visually distinct input area
with st.container(border=True):
    st.subheader("📝 Article Input")
    title_input = st.text_input("Article Title (optional)",
                                 placeholder="e.g., 'Breaking News: AI takes over the world!'")
    text_input = st.text_area("Article Content",
                              height=250,
                              placeholder="Paste the full news article content here...",
                              help="The more content, the better the prediction accuracy. A minimum amount of meaningful text is recommended.")

    # Centering the button or placing it in a column for better layout
    predict_button_col, _ = st.columns([1, 4]) # 1 unit for button column, 4 units for empty space
    with predict_button_col:
        predict_button = st.button("✨ **Get Prediction** ✨", use_container_width=True)


# --- Rolling Word Part / Undergoing Training Disclaimer ---
st.markdown("---") # Separator before the disclaimer
st.warning("""
    <div style="width: 100%; overflow: hidden;">
        <marquee behavior="scroll" direction="left" scrollamount="4">
            **Important Note:** This model is currently undergoing continuous training and optimization.
            While highly accurate, results could occasionally be incorrect. Use with discretion. 🛠️
        </marquee>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---") # Separator after the disclaimer

# --- Prediction Logic ---
if predict_button:
    if not model or not tfidf_vectorizer:
        # This case is largely handled by st.stop() in load_ml_resources(), but good for robustness
        st.error("Cannot make predictions. Model or Vectorizer resources are not loaded. "
                 "Please ensure files are present in the GitHub repository and accessible.")
    elif not title_input and not text_input:
        st.warning("🧐 Please enter either a **title** or **article content** to make a prediction. Both fields cannot be empty.")
    else:
        # Use st.spinner for visual feedback during prediction
        with st.spinner("Analyzing article... This might take a moment..."):
            # Combine title and text exactly as was done during training
            full_text_combined = f"{title_input} {text_input}".strip()

            if not full_text_combined:
                st.warning("After combining, the input text became empty. This might happen if your input consisted only of spaces. Please provide more substantial text.")
            else:
                # Preprocess the combined text using the defined functions
                processed_text = preprocess_text_for_prediction(full_text_combined)

                if not processed_text:
                    # If preprocessing strips all meaningful content (e.g., only numbers, punctuation, stopwords)
                    st.warning("After preprocessing, the input text became empty. "
                               "This might happen if the text contained only numbers, punctuation, or common stopwords. "
                               "Please try different text. 🤷‍♀️")
                else:
                    # Transform the preprocessed text using the loaded TF-IDF vectorizer
                    # Note: .transform, not .fit_transform as the vectorizer is already trained
                    text_vectorized = tfidf_vectorizer.transform([processed_text])

                    # Make prediction using the loaded model
                    prediction = model.predict(text_vectorized)
                    # For LinearSVC, decision_function provides a "distance" from the hyperplane.
                    # A positive value typically means confidence in the positive class (often 1),
                    # a negative value confidence in the negative class (often 0).
                    decision_score = model.decision_function(text_vectorized)[0]

                    st.subheader("🔮 Prediction Result:")
                    if prediction[0] == 1: # Assuming 1 corresponds to 'FAKE'
                        st.error(f"## 🚨 This article is **FAKE NEWS!** 🚨")
                        st.markdown(f"<p style='font-size: 1.1em; color: red;'>Confidence (decision value): <strong>{decision_score:.2f}</strong></p>", unsafe_allow_html=True)
                        st.balloons() # Add some fun confetti for detecting fake news (a bit ironic!)
                    else: # Assuming 0 corresponds to 'REAL'
                        st.success(f"## ✅ This article is **REAL NEWS!** ✅")
                        st.markdown(f"<p style='font-size: 1.1em; color: green;'>Confidence (decision value): <strong>{decision_score:.2f}</strong></p>", unsafe_allow_html=True)

                    st.markdown("---") # Separator after prediction result
                    st.info("💡 **Note on Confidence:** For LinearSVC, the 'confidence' is represented by a raw decision function value. A higher absolute value indicates stronger confidence in the predicted class. Positive values lean towards FAKE, negative towards REAL.")

# --- Footer or additional info ---
st.markdown("""
    <br><br><br>
    <div style='text-align: center; font-size: 0.9em; color: #888;'>
        Powered by Streamlit & Scikit-learn. <br>
        Developed as an NLP project. Source code available on GitHub.
    </div>
    """, unsafe_allow_html=True)
