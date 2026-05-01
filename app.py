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


# Page configuration
st.set_page_config(
    page_title="Spam Classifier Dashboard",
    page_icon="📧",
    layout="wide"
)

# Custom CSS
st.markdown("""
""", unsafe_allow_html=True)

# ============================================
# CACHED FUNCTIONS FOR MODEL AND DATA
# ============================================

@st.cache_data
def load_data():
    df = pd.read_csv('sms_spam.tsv.tsv', sep='\t', header=None, names=['label', 'message'])
    return df


def clean_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def train_model(df):
    """Train the spam classifier model."""

    # Preprocessing
    df = df.copy()
    df['clean_msg'] = df['message'].apply(clean_text)
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(df['clean_msg'])
    y = df['label_enc']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return model, tfidf, X_test, y_test, y_pred, y_proba


# ============================================
# LOAD DATA AND TRAIN MODEL
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
# SIDEBAR - MESSAGE CLASSIFIER
# ============================================

with st.sidebar:
    st.header("🔍 Classify a Message")

    user_message = st.text_area(
        "Enter a message to classify:",
        height=100,
        placeholder="Type or paste a message here..."
    )

    if st.button("🚀 Classify", type="primary", use_container_width=True):
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

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4b4b" if prob >= 0.5 else "#00cc66"},
                    'steps': [
                        {'range': [0, 50], 'color': "#e8f5e9"},
                        {'range': [50, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': 50
                    }
                },
                title={'text': "Spam Score"}
            ))

            fig_gauge.update_layout(height=250, margin=dict(t=50, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        else:
            st.warning("Please enter a message to classify.")

    st.markdown("---")
    st.subheader("📝 Try These Examples")

    examples = [
        "Congratulations! You won a free iPhone!",
        "Hey, are we meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised!",
        "Can you pick up milk on your way home?",
        "Click here to claim your $1000 prize NOW!"
    ]

    for ex in examples:
        if st.button(ex[:40] + "..." if len(ex) > 40 else ex, key=ex):
            st.session_state['example_msg'] = ex
            st.rerun()


# Check if example was clicked
if 'example_msg' in st.session_state:
    user_message = st.session_state.pop('example_msg')

# ============================================
# MAIN CONTENT - METRICS ROW
# ============================================

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🎯 Accuracy", f"{accuracy:.1%}")
with col2:
    st.metric("📊 Precision", f"{precision:.1%}")
with col3:
    st.metric("🔍 Recall", f"{recall:.1%}")
with col4:
    st.metric("⚖️ F1 Score", f"{f1:.1%}")

st.markdown("---")
