import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CineSense | Thai Movie Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        baseline_model = joblib.load("model.joblib")
        improved_model = joblib.load("model_v2.joblib")
        dataset = pd.read_csv("8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv")
        return baseline_model, improved_model, dataset
    except Exception as exc:
        st.error(f"Failed to load project assets: {exc}")
        return None, None, None


baseline_model, improved_model, dataset = load_assets()

# --- STYLING ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Kanit:wght@300;400;600&display=swap');

    .stApp {
        background-color: #ffffff;
        font-family: 'Inter', 'Kanit', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    .data-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .sentiment-tag {
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 8px 0;
    }

    .pos-tag { background-color: #dcfce7; color: #166534; }
    .neg-tag { background-color: #fee2e2; color: #991b1b; }
    .neu-tag { background-color: #fef9c3; color: #854d0e; }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎬 CineSense")
    st.caption("Thai Movie Sentiment Analysis")
    st.markdown("---")
    page = st.radio("Navigation", ["🔍 Analyze Review", "📊 Project Info"])

# --- ANALYZE PAGE ---
if page == "🔍 Analyze Review":
    st.header("Thai Movie Sentiment Classifier")
    st.write("พิมพ์รีวิวภาพยนตร์ภาษาไทยเพื่อเปรียบเทียบผลจากโมเดล Baseline และ Improved")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Review Input")

        if "review_text" not in st.session_state:
            st.session_state.review_text = ""

        if st.button("🎲 Random Sample"):
            if dataset is not None and "text" in dataset.columns:
                sample = dataset.sample(1).iloc[0]
                st.session_state.review_text = str(sample["text"])
                st.rerun()

        review_text = st.text_area(
            "Thai Movie Review",
            value=st.session_state.review_text,
            height=220,
            placeholder="พิมพ์รีวิวหนังที่นี่...",
        )

        analyze = st.button(
            "Analyze Sentiment",
            type="primary",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.subheader("Prediction Results")

        if analyze and review_text.strip():
            result_columns = st.columns(2)

            models = [
                (baseline_model, result_columns[0], "Baseline Model"),
                (improved_model, result_columns[1], "Improved Model"),
            ]

            for model, column, model_name in models:
                with column:
                    st.markdown(f"**{model_name}**")

                    if model is None:
                        st.error("Model could not be loaded.")
                        continue

                    prediction = model.predict([review_text])[0]

                    confidence = None
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba([review_text])[0]
                        confidence = float(np.max(probabilities) * 100)

                    tag_class = (
                        "pos-tag"
                        if prediction == "Positive"
                        else "neg-tag"
                        if prediction == "Negative"
                        else "neu-tag"
                    )

                    st.markdown(
                        f'<div class="sentiment-tag {tag_class}">{prediction}</div>',
                        unsafe_allow_html=True,
                    )

                    if confidence is not None:
                        st.write(f"Confidence: `{confidence:.1f}%`")
                        st.progress(int(round(confidence)))
        elif analyze:
            st.warning("กรุณาพิมพ์รีวิวก่อนเริ่มวิเคราะห์")
        else:
            st.info("ระบบพร้อมทำงาน กรุณาใส่รีวิวในช่องด้านซ้าย")

        st.markdown("</div>", unsafe_allow_html=True)

# --- PROJECT INFO PAGE ---
elif page == "📊 Project Info":
    st.header("Project Information")

    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Overview")
    st.write(
        """
        CineSense is an academic Natural Language Processing and Machine Learning
        project for classifying Thai movie reviews into sentiment categories.
        The application compares a baseline model with an improved model through
        a Streamlit interface.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Project Components")
    st.markdown(
        """
        - **Dataset:** 5,000 synthetic Thai movie reviews
        - **Task:** Sentiment classification
        - **Interface:** Streamlit
        - **Models:** Baseline and improved trained classifiers
        - **Libraries:** scikit-learn, Pandas, NumPy, Joblib
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="data-card">', unsafe_allow_html=True)
    st.subheader("Team")
    st.markdown(
        """
        - Jirapat Pattanatetham
        - Peerawit Lattisak
        - Woramet Chiaochan
        """
    )
    st.caption("NLP & Machine Learning course project • University of Phayao")
    st.markdown("</div>", unsafe_allow_html=True)
