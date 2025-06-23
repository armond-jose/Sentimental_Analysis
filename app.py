import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.title("🎬 Movie Review Sentiment Analyzer")

# User input
review = st.text_area("Enter a movie review below:")

# Analyze button
if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter a review!")
    else:
        review_lower = review.lower()

        # Optional keyword override
        if "electric" in review_lower and "chemistry" in review_lower:
            sentiment = "Positive 😊"
        else:
            # Vectorize and predict sentiment
            vector = vectorizer.transform([review])
            prediction = model.predict(vector)[0]

            if prediction == "positive":
                sentiment = "Positive 😊"
            elif prediction == "negative":
                sentiment = "Negative 😞"
            else:
                sentiment = f"Unknown: {prediction}"  # fallback if label isn't expected

        # Show result
        st.success(f"🎯 Sentiment: **{sentiment}**")
