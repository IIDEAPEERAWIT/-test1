# 🎬 Thai Movie Sentiment Analysis

A Natural Language Processing and Machine Learning project for classifying **Thai movie reviews** into sentiment categories through an interactive Streamlit application.

The project compares two trained sentiment-classification models — a baseline model and an improved model — and presents their predictions and confidence scores side by side.

## ✨ Features

- Classifies Thai movie reviews into **Positive, Negative, or Neutral** sentiment
- Compares predictions from two trained models
- Displays confidence scores for each prediction
- Includes random samples from a 5,000-review synthetic Thai movie review dataset
- Uses Thai tokenization with **PyThaiNLP (newmm)**
- Provides an interactive **Streamlit** web interface

## 🧠 Machine Learning Pipeline

The application uses:

- **PyThaiNLP** for Thai word tokenization
- **TF-IDF** text vectorization
- **Logistic Regression** for sentiment classification
- Two serialized models for baseline vs. improved model comparison

## 🛠 Tech Stack

- Python
- Streamlit
- scikit-learn
- PyThaiNLP
- Pandas
- NumPy
- Joblib

## 📁 Project Structure

```text
.
├── app.py
├── requirements.txt
├── model.joblib
├── model_v2.joblib
└── 8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/IIDEAPEERAWIT/-test1.git
cd -test1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
streamlit run app.py
```

## 🖥 Application Pages

### 🔍 Analyze Review
Enter a Thai movie review and compare the sentiment prediction and confidence score from both models.

### 📊 Project Info
View a summary of the dataset, preprocessing pipeline, model approach, and evaluation section.

### 💬 AI Assistant
A simple chat-style interface included in the application UI.

## 👨‍💻 Team

- Jirapat Pattanatetham
- Peerawit Lattisak
- Woramet Chiaochan

## 🎓 Academic Project

Developed as part of an **NLP & Machine Learning** course at the **University of Phayao**.

---

**Peerawit Lattisak**  
Computer Science, University of Phayao  
📧 peerawitidea@gmail.com
