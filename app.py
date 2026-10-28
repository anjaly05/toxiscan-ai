
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.getenv("MODEL_PATH", "your-huggingface-username/your-model-repo")

LABELS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

COLORS = [
    "#E85D75", "#F08A5D", "#9B72CF",
    "#D94F70", "#E5A04B", "#6C83D9"
]

st.set_page_config(
    page_title="ToxiScan AI",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8F6FF, #FFFFFF, #F3F0FF);
}
.block-container {
    max-width: 780px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}
.hero {
    background: white;
    padding: 30px;
    border-radius: 22px;
    border: 1px solid #EAE4FF;
    text-align: center;
    box-shadow: 0 8px 30px #6650A015;
    margin-bottom: 24px;
}
.logo { font-size: 42px; }
.hero h1 {
    color: #49327D;
    font-size: 38px;
    margin: 8px 0;
}
.hero p {
    color: #77718A;
    font-size: 16px;
}
.pill {
    display: inline-block;
    background: #EEE8FF;
    color: #6946B8;
    border-radius: 30px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
}
.section-title {
    color: #49327D;
    font-size: 21px;
    font-weight: 700;
    margin-top: 24px;
}
div[data-testid="stTextArea"] textarea {
    background: white;
    border: 1px solid #DCD2F5;
    border-radius: 14px;
    color: #302747;
}
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    background: #7654C8;
    color: white;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background: #5D3CA9;
    color: white;
    border: none;
}
.footer {
    color: #9189A5;
    text-align: center;
    font-size: 12px;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

st.markdown("""
<div class="hero">
    <div class="logo">🛡️</div>
    <div class="pill">AI-POWERED TEXT ANALYSIS</div>
    <h1>ToxiScan AI</h1>
    <p>Understand online language with intelligent toxicity detection.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">✍️ Analyze a comment</div>',
    unsafe_allow_html=True
)
st.write("Enter a comment below to check for potentially harmful language.")

comment = st.text_area(
    "Comment",
    placeholder="Type or paste your comment here...",
    height=140,
    label_visibility="collapsed"
)

if st.button("✨ Analyze Comment", type="primary"):
    if not comment.strip():
        st.warning("Please enter a comment first.")
    else:
        inputs = tokenizer(
            comment,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        probabilities = torch.sigmoid(logits)[0].cpu().numpy()
        detected = [
            label for label, score in zip(LABELS, probabilities)
            if score >= 0.5
        ]

        st.markdown(
            '<div class="section-title">📊 Analysis results</div>',
            unsafe_allow_html=True
        )

        if detected:
            st.error("Potentially harmful language detected.")
        else:
            st.success("No toxicity categories crossed the detection threshold.")

        for label, score, color in zip(LABELS, probabilities, COLORS):
            st.markdown(
                f"<span style='color:{color};font-weight:700'>"
                f"{label.replace('_', ' ').title()}</span>",
                unsafe_allow_html=True
            )
            st.progress(float(score))
            st.caption(f"Model score: {score * 100:.2f}%")

        st.caption(
            "Scores are model estimates, not definitive judgments. "
            "A low score does not guarantee that a comment is harmless."
        )

with st.expander("ℹ️ About this project"):
    st.write(
        "ToxiScan AI uses a fine-tuned DistilBERT model to estimate "
        "six categories: toxic, severe toxic, obscene, threat, insult "
        "and identity hate."
    )
    st.write("Detection threshold: 50% per category.")

st.markdown("""
<div class="footer">
    ToxiScan AI · Machine Learning Project · Built with Streamlit
</div>
""", unsafe_allow_html=True)
