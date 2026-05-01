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
# CACHED FUNCTIONS FOR MODEL AND DATA
# ============================================

@st.cache_data
def load_data():
    """Load and return the SMS spam dataset."""
    url = 'https://raw.githubusercontent.com/justmarkham/pydata-book/master/data/sms_spam.tsv'
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
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

# ============================================
# CHARTS ROW 1
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Distribution")
    
    label_counts = df['label'].value_counts()
    fig_pie = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        color=label_counts.index,
        color_discrete_map={'ham': '#00cc66', 'spam': '#ff4b4b'},
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label+value')
    fig_pie.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=350
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("🎯 Confusion Matrix")
    
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Ham', 'Spam'],
        y=['Ham', 'Spam'],
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig_cm.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=350
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ============================================
# CHARTS ROW 2
# ============================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Spam Probability Distribution")
    
    prob_df = pd.DataFrame({
        'Probability': y_proba,
        'Actual': ['Spam' if y == 1 else 'Ham' for y in y_test]
    })
    
    fig_hist = px.histogram(
        prob_df,
        x='Probability',
        color='Actual',
        nbins=30,
        color_discrete_map={'Ham': '#00cc66', 'Spam': '#ff4b4b'},
        barmode='overlay',
        opacity=0.7
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="black", 
                       annotation_text="Decision Boundary")
    fig_hist.update_layout(
        xaxis_title="Spam Probability",
        yaxis_title="Count",
        margin=dict(t=20, b=20, l=20, r=20),
        height=350
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("📏 Message Length Analysis")
    
    df_temp = df.copy()
    df_temp['length'] = df_temp['message'].apply(len)
    
    fig_box = px.box(
        df_temp,
        x='label',
        y='length',
        color='label',
        color_discrete_map={'ham': '#00cc66', 'spam': '#ff4b4b'}
    )
    fig_box.update_layout(
        xaxis_title="Message Type",
        yaxis_title="Character Count",
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=350
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ============================================
# DETAILED METRICS
# ============================================

st.markdown("---")
st.subheader("📋 Detailed Classification Report")

report_dict = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df = report_df.round(3)

st.dataframe(
    report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']),
    use_container_width=True
)

# ============================================
# SAMPLE DATA
# ============================================

st.markdown("---")
st.subheader("📝 Sample Messages from Dataset")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Ham Messages (Legitimate)**")
    ham_samples = df[df['label'] == 'ham'].sample(5, random_state=42)['message'].tolist()
    for msg in ham_samples:
        st.info(msg[:100] + "..." if len(msg) > 100 else msg)

with col2:
    st.markdown("**Spam Messages**")
    spam_samples = df[df['label'] == 'spam'].sample(5, random_state=42)['message'].tolist()
    for msg in spam_samples:
        st.error(msg[:100] + "..." if len(msg) > 100 else msg)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit • Model: Logistic Regression with TF-IDF • Dataset: SMS Spam Collection</p>
</div>
""", unsafe_allow_html=True)
