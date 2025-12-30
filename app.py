import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="StyleFit AI Pro", layout="wide")

# Load Backend Data
if not os.path.exists('model_stats.pkl'):
    st.error("⚠️ Please run 'train_models.py' first!")
    st.stop()

scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
stats = joblib.load('model_stats.pkl')

# --- 2. DYNAMIC BACKGROUND & STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    [data-testid="stVerticalBlock"] > div:has(div.stMetric), .stTabs, .stTable, .stDataFrame {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. THE MUSIC PLAYER (Updated Track) ---
st.sidebar.title("🎵 Boutique Radio")
# Updated path with your new filename
music_path = "the-fashion-music-409865.mp3"

if os.path.exists(music_path):
    st.sidebar.audio(music_path, format="audio/mp3")
    st.sidebar.caption("Now Playing: Fashion Trends")
else:
    st.sidebar.error("❌ New music file not found. Check the filename!")

# --- 4. SIDEBAR SELECTION ---
st.sidebar.divider()
item_type = st.sidebar.selectbox("Clothing Item", ["T-Shirt", "Sweater", "Jacket", "Dress"])
image_dict = {"T-Shirt": "t-shirt.jpg", "Sweater": "sweater.jpg", "Jacket": "jacket.jpg", "Dress": "dress.jpg"}
img_path = image_dict[item_type]

if os.path.exists(img_path):
    st.sidebar.image(img_path, caption=f"Current Item: {item_type}", use_container_width=True)

model_name = st.sidebar.selectbox("AI Algorithm", ["Random Forest", "KNN", "Decision Tree", "Logistic Regression", "SVM"])
m_key = model_name.lower().replace(" ", "_")

# --- 5. MAIN INTERFACE ---
tab1, tab2 = st.tabs(["🎯 Size Predictor", "📊 Dashboard Insights"])

with tab1:
    st.title(f"Smart {item_type} Prediction")
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        if os.path.exists(img_path):
            st.image(img_path, width=280)
        
        w = st.number_input("Weight (kg)", 30.0, 150.0, 70.0)
        h = st.number_input("Height (cm)", 100.0, 220.0, 170.0)
        a = st.number_input("Age", 10, 100, 25)
        
        if st.button("Predict Size"):
            model = joblib.load(f"{m_key}_model.pkl")
            res = model.predict(scaler.transform([[w, a, h]]))
            size = le.inverse_transform(res)[0]
            st.success(f"Recommended Size: **{size}**")

    with col_right:
        st.subheader(f"{model_name} Intelligence Map")
        viz_path = f"boundary_{m_key}.png"
        if os.path.exists(viz_path):
            st.image(viz_path)
        st.metric("Model Accuracy", f"{stats[m_key]['Accuracy']:.2%}")

with tab2:
    st.title("Project Technical Analytics")
    
    # Accuracy Comparison Table
    st.subheader("🏆 Algorithm Comparison")
    acc_data = {name: f"{data['Accuracy']:.2%}" for name, data in stats.items()}
    comparison_df = pd.DataFrame(list(acc_data.items()), columns=['Algorithm', 'Accuracy'])
    st.table(comparison_df)

    col_a, col_b = st.columns(2)
    with col_a:
        st.image("correlation_heatmap.png", caption="Feature Correlation")
    with col_b:
        st.subheader(f"Classification Report: {model_name}")
        report_df = pd.DataFrame(stats[m_key]['Report']).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='yellow'))