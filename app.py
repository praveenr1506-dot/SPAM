import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import re

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Spam Classifier Dashboard",
    page_icon="📧",
    layout="wide"
)

# ============================================
# CUSTOM CSS (same as before)
# ============================================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
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

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    return pd.read_csv('sms_spam.tsv.tsv', sep='\t', header=None, names=['label', 'message'])

# ============================================
# CLEAN TEXT
# ============================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================
# TRAIN MODEL
# ============================================
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

    model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return model, tfidf, X_test, y_test, y_pred, y_proba

# ============================================
# LOAD DATA + TRAIN
# ============================================
with st.spinner('Loading data and training model...'):
    df = load_data()
    model, tfidf, X_test, y_test, y_pred, y_proba = train_model(df)

# ============================================
# HEADER
# ============================================
st.title("📧 SMS Spam Classifier Dashboard")
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("🔍 Classify a Message")

    user_message = st.text_area(
        "Enter a message to classify:",
        height=100
    )

    if st.button("🚀 Classify"):
        if user_message.strip():
            cleaned = clean_text(user_message)
            vec = tfidf.transform([cleaned])
            prob = model.predict_proba(vec)[0][1]

            st.markdown("---")
            st.subheader("Result")

            if prob >= 0.5:
                st.markdown('<p class="spam-badge">🚫 SPAM</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="ham-badge">✅ HAM</p>', unsafe_allow_html=True)

            st.metric("Spam Probability", f"{prob:.1%}")
        else:
            st.warning("Please enter a message.")

# ============================================
# METRICS
# ============================================
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("🎯 Accuracy", f"{accuracy:.1%}")
col2.metric("📊 Precision", f"{precision:.1%}")
col3.metric("🔍 Recall", f"{recall:.1%}")
col4.metric("⚖️ F1 Score", f"{f1:.1%}")

st.markdown("---")

# ============================================
# CHARTS
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Distribution")
    fig_pie = px.pie(df, names='label', hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("🎯 Confusion Matrix")
    fig_cm = px.imshow(cm, text_auto=True)
    st.plotly_chart(fig_cm, use_container_width=True)

# ============================================
# REPORT
# ============================================
st.markdown("---")
st.subheader("📋 Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df, use_container_width=True)
