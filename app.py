import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# App Title
st.title("🎬 Movie Review Sentiment Analyzer")

# Input from user
review = st.text_area("Enter your movie review below:")

# Analyze button
if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter a review to analyze!")
    else:
        review_lower = review.lower()

        # ✅ Manual override for known keywords (optional)
        if "electric" in review_lower and "chemistry" in review_lower:
            sentiment = "Positive 😊"
            confidence_note = "Manual override applied based on keywords."
        else:
            # Vectorize input
            vector = vectorizer.transform([review])
            # Predict probability for positive class
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(vector)[0]
                pos_prob = prob[1]  # Probability of positive class

                # Neutral threshold logic
                if 0.40 <= pos_prob <= 0.57:
                    sentiment = "Neutral 😐"
                elif pos_prob > 0.55:
                    sentiment = "Positive 😊"
                else:
                    sentiment = "Negative 😞"

                confidence_note = f"Confidence: Positive = {pos_prob:.2f}, Negative = {1 - pos_prob:.2f}"

            else:
                # Fallback if predict_proba not available
                prediction = model.predict(vector)[0]
                sentiment = "Positive 😊" if prediction == "positive" else "Negative 😞"
                confidence_note = "(Probability not available — fallback to label only)"

        # Output result
        st.success(f"🎯 Sentiment: **{sentiment}**")
        st.info(confidence_note)
