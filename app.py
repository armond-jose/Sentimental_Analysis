import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/logistic_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# App title
st.title("🎬 Movie Review Sentiment Analyzer")

# User input
review = st.text_area("Enter your movie review:")

# Predict sentiment
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Predicted Sentiment: **{sentiment}**")
