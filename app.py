import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="StyleFit AI Pro", layout="wide")

# Load Backend Data
# When you deploy, these .pkl files must be in your main GitHub folder
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    stats = joblib.load('model_stats.pkl')
    return scaler, le, stats

try:
    scaler, le, stats = load_assets()
except:
    st.error("⚠️ Data files not found. Please upload .pkl files to GitHub.")
    st.stop()

# --- 2. DYNAMIC BACKGROUND & GLASS-MORPHISM STYLING ---
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
    /* White glass effect for readability */
    [data-testid="stVerticalBlock"] > div:has(div.stMetric), .stTabs, .stTable, .stDataFrame, .stWarning {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. THE MUSIC PLAYER (Cloud Path) ---
st.sidebar.title("🎵 Boutique Radio")
# Relative path for Cloud Deployment
music_file = "the-fashion-music-409865.mp3"

if os.path.exists(music_file):
    st.sidebar.audio(music_file, format="audio/mp3")
    st.sidebar.caption("Now Playing: Fashion Trends")
else:
    st.sidebar.warning("Music file not found in repository.")

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
    st.title(f"Predict your {item_type} Size")
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        if os.path.exists(img_path):
            st.image(img_path, width=280)
        
        weight = st.number_input("Weight (kg)", 30.0, 150.0, 70.0)
        height = st.number_input("Height (cm)", 100.0, 220.0, 170.0)
        age = st.number_input("Age", 10, 100, 25)
        
        if st.button("Calculate Best Fit"):
            model_file = f"{m_key}_model.pkl"
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                input_data = scaler.transform([[weight, age, height]])
                res = model.predict(input_data)
                size = le.inverse_transform(res)[0]
                st.success(f"Recommended {item_type} Size: **{size}**")
            else:
                st.error("Model file missing!")

    with col_right:
        st.subheader(f"{model_name} Intelligence Map")
        viz_path = f"boundary_{m_key}.png"
        if os.path.exists(viz_path):
            st.image(viz_path, caption="Mathematics of Size Prediction")
        st.metric("Model Accuracy", f"{stats[m_key]['Accuracy']:.2%}")

with tab2:
    st.title("Project Technical Analytics")
    
    # Accuracy Comparison Table
    st.subheader("🏆 Algorithm Performance Summary")
    acc_data = {name.replace('_',' ').title(): f"{data['Accuracy']:.2%}" for name, data in stats.items()}
    comparison_df = pd.DataFrame(list(acc_data.items()), columns=['Algorithm', 'Accuracy'])
    st.table(comparison_df)

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists("correlation_heatmap.png"):
            st.image("correlation_heatmap.png", caption="Feature Correlation (Weight is the strongest predictor)")
    with col_b:
        st.subheader(f"Classification Report: {model_name}")
        report_df = pd.DataFrame(stats[m_key]['Report']).transpose()
        # The yellow highlight the user requested
        st.dataframe(report_df.style.highlight_max(axis=0, color='yellow'))

    st.warning("""
    **💡 Presentation Conclusion:** By applying 'Label Consolidation' to clean the dataset, we achieved competition-grade accuracy. 
    The Random Forest and KNN models show the highest precision for the boutique market.
    """)