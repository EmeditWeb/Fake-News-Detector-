# 📰 AI-Powered Fake News Detector

![Fake News Detector Screenshot](https://via.placeholder.com/800x450?text=App+Screenshot+Here) 

## 🚀 Project Overview

This project aims to combat the pervasive issue of misinformation by developing a robust and accurate machine learning model to classify news articles as either 'REAL' or 'FAKE'. Leveraging Natural Language Processing (NLP) techniques, this system provides an automated and user-friendly solution for identifying potentially misleading content.

The core of the system is a highly optimized `LinearSVC` (Linear Support Vector Classifier) model, trained on a comprehensive dataset of news articles, and deployed as an interactive web application using Streamlit.

## ✨ Features

* **Intelligent Text Preprocessing:** Advanced NLP techniques including lowercasing, punctuation removal, number removal, stop word filtering, and lemmatization to clean and prepare raw text data.
* **TF-IDF Feature Engineering:** Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features, capturing the importance of words within the context of the entire news corpus.
* **High-Performance Classification:** Implements and evaluates multiple machine learning models (Logistic Regression, Multinomial Naive Bayes, Random Forest, LinearSVC), with **LinearSVC** consistently outperforming others.
* **Optimized Model:** The chosen LinearSVC model undergoes hyperparameter tuning using GridSearchCV to ensure optimal performance and generalization.
* **Robustness Validation:** Performance validated through K-Fold Cross-Validation, demonstrating consistent high accuracy across diverse data subsets.
* **Interactive Web Application:** Deployed as a user-friendly web interface using Streamlit, allowing users to input news article titles and content to receive instant fake/real predictions.
* **Clear User Feedback:** Provides clear visual indicators (green for REAL, red for FAKE) and informative messages within the deployed application.

## ⚙️ How It Works (Technical Overview)

1.  **Text Input:** Users provide a news article's title and/or content through the Streamlit web interface.
2.  **Preprocessing:** The input text undergoes the same rigorous preprocessing steps applied during model training (cleaning, tokenization, stop word removal, lemmatization).
3.  **Feature Transformation:** The preprocessed text is transformed into numerical feature vectors using the pre-trained `TfidfVectorizer`.
4.  **Prediction:** The vectorized input is fed into the loaded `LinearSVC` model, which then predicts whether the article is 'REAL' or 'FAKE'.
5.  **Result Display:** The prediction is displayed prominently on the web page with appropriate styling and a decision function score.

## 📊 Model Performance

During development, various models were rigorously evaluated. The `LinearSVC` model consistently demonstrated superior performance, making it the chosen algorithm for this detector.

* **Model Used:** Linear Support Vector Classifier (LinearSVC)
* **Test Accuracy (after tuning):** Approximately **94.08%**
* **Mean Cross-Validation Accuracy (5-fold):** **93.51%** (with a standard deviation of 0.73%), indicating high and consistent performance.

## 🚀 Getting Started (Local Development)

Follow these steps to get a local copy of the project up and running on your machine.

### Prerequisites

* Python 3.8+
* pip (Python package installer)
* Git

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/fake-news-detector-streamlit.git](https://github.com/your-username/fake-news-detector-streamlit.git)
cd fake-news-detector-streamlit
