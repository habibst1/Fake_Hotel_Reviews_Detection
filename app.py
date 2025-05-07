# app.py
import streamlit as st
from transformers import pipeline
import torch
import shap
from google import genai

api_key = "AIzaSyDiKvwkTPGA9lDdesVcr3dvEzqz-7qBPuU"
gemini = genai.Client(api_key=api_key)



def explain_review_with_shap_details(text, model, tokenizer, class_names=["truthful", "deceptive"], top_k=5):
    """
    Predicts label and explains it using SHAP.
    Returns:
      - predicted label
      - confidence
      - class probabilities
      - top contributing words for 'truthful' and 'deceptive' classes (only positive contributions)
    """
    # Set up pipeline
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=0 if torch.cuda.is_available() else -1)

    # Get probabilities
    pred_scores = pipe(text)[0]
    probs = {entry["label"]: entry["score"] for entry in pred_scores}
    
    # Sort by probability to get prediction
    sorted_probs = sorted(pred_scores, key=lambda x: x["score"], reverse=True)
    prediction = sorted_probs[0]["label"]
    confidence = sorted_probs[0]["score"]

    # SHAP explanation
    explainer = get_explainer(pipe)
    shap_values = explainer([text])[0]  # only one input text

    # Get index of each class
    class_to_index = {entry["label"]: i for i, entry in enumerate(pred_scores)}

    # Contributions for truthful class
    truthful_index = class_to_index["truthful"]
    truthful_contributions = list(zip(shap_values.data, shap_values.values[:, truthful_index]))
    truthful_top_words = sorted(
        [(tok, val) for tok, val in truthful_contributions if val > 0],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    # Contributions for deceptive class
    deceptive_index = class_to_index["deceptive"]
    deceptive_contributions = list(zip(shap_values.data, shap_values.values[:, deceptive_index]))
    deceptive_top_words = sorted(
        [(tok, val) for tok, val in deceptive_contributions if val > 0],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "top_truthful_words": [(tok, round(val, 4)) for tok, val in truthful_top_words],
        "top_deceptive_words": [(tok, round(val, 4)) for tok, val in deceptive_top_words],
    }


def generate_gemini_explanation(review, prediction, top_truthful_words, top_deceptive_words, probs):
    truthful_score = probs.get("truthful", 0)
    deceptive_score = probs.get("deceptive", 0)

    # Filter only high-impact words (above a threshold, e.g., 0.01)
    threshold = 0.01
    top_truthful_words_filtered = [(w, v) for w, v in top_truthful_words if abs(v) >= threshold]
    top_deceptive_words_filtered = [(w, v) for w, v in top_deceptive_words if abs(v) >= threshold]

    prompt = f"""
Here is a customer review:
\"\"\"{review}\"\"\"

The AI model analyzed this review and predicted the label:
- **{prediction.upper()}**, with the following confidence scores:
    - Truthful: {truthful_score:.4f}
    - Deceptive: {deceptive_score:.4f}

The model was influenced by the presence of certain key words according to SHAP:
- Strong indicators of Truthful: {', '.join([f"{w} (+{v})" for w, v in top_truthful_words_filtered]) or "None"}
- Strong indicators of Deceptive: {', '.join([f"{w} (+{v})" for w, v in top_deceptive_words_filtered]) or "None"}

Your goal is to help a user understand why the model predicted {prediction.upper()}.

Start your explanation with the sentence: "The model predicted **{prediction.lower()}** because"

Then write one or two paragraphs (based on how much patterns you found meaningful) explaining the decision. Focus on meaningful patterns in the text. For example:
- If a specific event or experience (e.g., a "dirty room" or "friendly staff") is mentioned, explain why that contributed to the label.
- Only mention words that directly add context to the sentiment or experience described in the review, avoiding general words like “I” or “was” unless they have a significant impact on the prediction.
- Be mindful to explain if a particular word, phrase, or sentiment swayed the model toward one label (truthful or deceptive). Don't just mention every word with a positive or negative score without explaining its relevance.

Avoid analyzing generic terms (like “hotel” or “service”) unless they were part of a specific context or description that clearly influenced the label.
Also, if the model shows low confidence (for example, if the confidence scores are close to each other), mention this and explain why the model might be uncertain.
Use plain English, avoid technical terms like SHAP, and make the explanation sound like a human trying to understand and explain the decision.

"""

    response = gemini.models.generate_content(
        model = "gemini-2.0-flash",
        contents = prompt,
        config={
            "max_output_tokens": 256,
            "temperature": 0.7
        }
    )
    return response.text




# Load model and tokenizer once
@st.cache_resource
def load_pipeline():
    model_name = "./kaggle/deception_detector_bert_cased"
    pipe = pipeline("text-classification", model=model_name, tokenizer=model_name,
                    return_all_scores=True, device=0 if torch.cuda.is_available() else -1)
    return pipe


@st.cache_resource
def get_explainer(_pipe):
    return shap.Explainer(pipe)

# Load pipeline
pipe = load_pipeline()

explainer = get_explainer(pipe)

# App UI
st.title("🛎️ Hotel Review Truthfulness Checker")

review = st.text_area("Paste a hotel review here:", height=200)

if st.button("Analyze Review"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        # Run model prediction + SHAP
        with st.spinner("Analyzing..."):
            result = explain_review_with_shap_details(
                text=review,
                model=pipe.model,
                tokenizer=pipe.tokenizer,
                top_k=5  # just for SHAP use internally
            )
            prediction = result["prediction"]
            confidence = result["confidence"]
            probs = result["probabilities"]
            truthful_words = result["top_truthful_words"]
            deceptive_words = result["top_deceptive_words"]

            # Gemini explanation
            gemini_explanation = generate_gemini_explanation(
                review, prediction, truthful_words, deceptive_words , probs
            )

        # Display results
        st.subheader("🧠 Model Prediction")
        st.markdown(f"**Prediction:** `{prediction.upper()}`")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
        st.markdown(f"**Truthful Probability:** {probs.get('truthful', 0):.4f}")
        st.markdown(f"**Deceptive Probability:** {probs.get('deceptive', 0):.4f}")

        # Gemini explanation
        st.subheader("🪄 Why the model predicted this?")
        st.markdown(gemini_explanation)





