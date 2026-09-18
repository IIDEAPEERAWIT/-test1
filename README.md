# 🎬 Thai Movie Sentiment Analysis

A Natural Language Processing and Machine Learning project for classifying **Thai movie reviews** into sentiment categories through an interactive **Streamlit** application.

The application compares two trained sentiment-classification models — a **Baseline Model** and an **Improved Model** — and shows their predictions and confidence scores side by side.

## ✨ Features

- Classifies Thai movie reviews into **Positive, Negative, or Neutral** sentiment
- Compares two trained models in the same interface
- Displays prediction confidence when the model supports `predict_proba`
- Includes random review samples from a synthetic Thai movie review dataset
- Provides a clean Streamlit interface for demonstration and portfolio use
- Includes a project information page with the project scope and team

## 🧠 Machine Learning Approach

The project uses a text-classification workflow based on:

- **TF-IDF** text vectorization
- **Logistic Regression** sentiment classification
- A baseline model and an improved model saved with **Joblib**
- Evaluation-focused comparison through the web application

## 🛠 Tech Stack

**Language**
- Python

**Machine Learning / Data**
- scikit-learn
- Pandas
- NumPy
- Joblib

**Application**
- Streamlit

## 📁 Project Structure

```text
.
├── app.py
├── requirements.txt
├── model.joblib
├── model_v2.joblib
├── 8.synthetic_netflix_like_thai_reviews_3class_hard_5000.csv
├── .gitignore
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/IIDEAPEERAWIT/thai-movie-sentiment-analysis.git
cd thai-movie-sentiment-analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

## 🖥 Application Pages

### 🔍 Analyze Review
Enter a Thai movie review and compare sentiment predictions from the Baseline Model and Improved Model.

### 📊 Project Info
View the dataset scope, application purpose, technologies, and project team.

## 👨‍💻 Team

- Jirapat Pattanatetham
- **Peerawit Lattisak**
- Woramet Chiaochan

## 🎓 Academic Project

Developed as part of an **NLP & Machine Learning** course at the **University of Phayao**.

---

### Contact

**Peerawit Lattisak**  
B.Sc. Computer Science, University of Phayao  
Email: peerawitidea@gmail.com
