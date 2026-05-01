import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Spam Classifier Dashboard",
    page_icon="📧",
    layout="wide"
)

# =========================
# FORCE WHITE BACKGROUND
# =========================
st.markdown("""
<style>
    .stApp {
        background-color: white;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.spam-badge {
    background-color: #ff4b4b;
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
}
.ham-badge {
    background-color: #00cc66;
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("sms_spam.tsv.tsv", sep="\t", header=None, names=["label", "message"])

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model(df):
    df = df.copy()
    df['clean_msg'] = df['message'].apply(clean_text)
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(df['clean_msg'])
    y = df['label_enc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return model, tfidf, X_test, y_test, y_pred, y_proba

# =========================
# LOAD + TRAIN
# =========================
with st.spinner("Loading data..."):
    df = load_data()
    model, tfidf, X_test, y_test, y_pred, y_proba = train_model(df)

# =========================
# HEADER
# =========================
st.title("📧 SMS Spam Classifier Dashboard")
st.markdown("---")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🔍 Classify a Message")

    user_message = st.text_area("Enter message")

    if st.button("🚀 Classify"):
        if user_message.strip():
            cleaned = clean_text(user_message)
            vec = tfidf.transform([cleaned])
            prob = model.predict_proba(vec)[0][1]

            if prob >= 0.5:
                st.markdown('<p class="spam-badge">🚫 SPAM</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="ham-badge">✅ HAM</p>', unsafe_allow_html=True)

            st.metric("Spam Probability", f"{prob:.2%}")
        else:
            st.warning("Enter a message")

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Samples", len(df))

# =========================
# CONFUSION MATRIX
# =========================
fig = px.imshow(cm, text_auto=True)
st.plotly_chart(fig, use_container_width=True)

# =========================
# REPORT
# =========================
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())
