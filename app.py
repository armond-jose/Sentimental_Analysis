import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/logistic_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.title("🎬 Movie Review Sentiment Analyzer (with Neutral)")

# Text input
review = st.text_area("Enter a movie review below:")

# Analyze button
if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a review first!")
    else:
        # Vectorize and predict probability
        vector = vectorizer.transform([review])
        prob = model.predict_proba(vector)[0]
        pos_prob = prob[1]  # Probability of positive sentiment

        # Classify based on probability
        if 0.45 <= pos_prob <= 0.55:
            sentiment = "Neutral 😐"
        elif pos_prob > 0.55:
            sentiment = "Positive 😊"
        else:
            sentiment = "Negative 😞"

        # Show probability (optional)
        st.write(f"🧠 Confidence: Positive = {pos_prob:.2f}, Negative = {1 - pos_prob:.2f}")
        st.success(f"🎯 Sentiment: **{sentiment}**")
