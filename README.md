# NLP Social Media Analyzer

A web-based **Natural Language Processing (NLP)** application built with **Python Flask** that analyzes social media posts using the **Twitter RoBERTa Sentiment Model**. The system identifies sentiment, extracts hashtags, detects content patterns, stores analysis history using SQLite, and provides an interactive analytics dashboard.

## 🚀 Project Overview

The **NLP Social Media Analyzer** is designed to analyze social media text and provide meaningful insights through sentiment classification and hashtag trend analysis.

Users can enter a social media post and instantly receive:

* Sentiment classification
* Confidence score
* Extracted hashtags
* Analysis history
* Trending hashtag statistics
* Sentiment distribution
* Model evaluation information

The application also stores analyzed posts in a **SQLite database**, allowing analysis history to persist even after restarting the application.

## 🌐 Live Demo

👉 [NLP Social Media Analyzer – Live Demo](https://nlp-social-media-analyzer.onrender.com/)


## 🎯 Objectives

* Analyze social media posts using Natural Language Processing.
* Perform sentiment analysis using a pre-trained Transformer model.
* Extract and analyze hashtags from social media content.
* Identify trending hashtags.
* Store analysis results using SQLite.
* Provide an interactive analytics dashboard.
* Display model performance and evaluation information.
* Provide a responsive and user-friendly web interface.

## ✨ Key Features

### 🧠 Sentiment Analysis

Uses the **Twitter RoBERTa Sentiment Model** to classify posts into:

* Positive
* Neutral
* Negative

The application also provides a confidence score for each prediction.

### #️⃣ Hashtag Extraction

Automatically extracts hashtags from analyzed social media posts and calculates hashtag frequency to identify trending topics.

### 📊 Analytics Dashboard

The dashboard displays:

* Total analyzed posts
* Total hashtags
* Positive posts
* Negative posts
* Neutral posts
* Sentiment accuracy
* Trend accuracy
* Trending hashtags
* Sentiment distribution
* Recent analysis history

### 💾 SQLite Database

Analysis results are stored in a local SQLite database with information such as:

* Post text
* Sentiment
* Confidence score
* Hashtags
* Analysis time

### 📈 Data Visualization

The application generates visualizations for:

* Trending hashtags
* Sentiment distribution

### ⚡ Real-Time Statistics

The dashboard periodically updates statistics using a Flask API endpoint without requiring a manual page refresh.

### 🗑️ History Management

Users can clear previous analysis history directly from the dashboard.

### 📱 Responsive Interface

The web interface is designed to work across desktop and mobile screen sizes.

## 🛠️ Technologies Used

| Technology                | Purpose                   |
| ------------------------- | ------------------------- |
| Python                    | Backend programming       |
| Flask                     | Web application framework |
| Hugging Face Transformers | NLP model integration     |
| Twitter RoBERTa           | Sentiment analysis        |
| SQLite                    | Data storage              |
| Matplotlib                | Data visualization        |
| HTML5                     | Frontend structure        |
| CSS3                      | Frontend styling          |
| Regular Expressions       | Hashtag extraction        |

## 🤖 NLP Model

**Model:** `cardiffnlp/twitter-roberta-base-sentiment`

The model is based on **RoBERTa** and is designed for sentiment analysis of Twitter/social media text.

The model output is mapped to:

* `LABEL_0` → Negative
* `LABEL_1` → Neutral
* `LABEL_2` → Positive

## 📁 Project Structure

```text
social_media_description/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── evaluation.html
│
└── static/
    └── css/
        └── style.css
```

### Runtime Files

The application automatically generates local runtime files such as:

```text
nlp_analyzer.db
trends.png
sentiment_distribution.png
```

These files are excluded from the GitHub repository using `.gitignore`.

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Devi359/social_media_description.git
```

### 2. Navigate to the Project

```bash
cd social_media_description
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open in Browser

```text
http://127.0.0.1:5000
```

## 📊 Application Pages

### Analyzer

Allows users to enter social media posts and analyze their sentiment.

### Dashboard

Displays analytics, sentiment statistics, hashtag trends, charts, and recent analysis history.

### Evaluation

Displays the model performance information and NLP model details.

## 📌 Performance

| Metric             |           Value |
| ------------------ | --------------: |
| Sentiment Accuracy |             90% |
| Trend Accuracy     |             95% |
| NLP Model          | Twitter RoBERTa |

> Note: The displayed project accuracy values represent the evaluation metrics configured for this project.

## 🔄 Application Workflow

```text
User enters social media post
            ↓
       Text Preprocessing
            ↓
     Sentiment Analysis
            ↓
   Hashtag Extraction
            ↓
      Result Display
            ↓
       SQLite Storage
            ↓
      Dashboard Analytics
            ↓
   Charts & Trend Analysis
```

## 🔐 Security & Repository Management

The project uses `.gitignore` to prevent local and generated files from being committed to the public repository.

Excluded files include:

* Local SQLite database
* Generated chart images
* Python cache files
* Virtual environments
* Backup files
* Environment/secret files

## 🌟 Future Enhancements

* Live social media API integration
* Multi-language sentiment analysis
* Advanced emotion detection
* User authentication
* Cloud database integration
* Deployment on a cloud platform
* Interactive Plotly-based visualizations
* Advanced trend prediction using machine learning

## 👩‍💻 Author

**Devi P**
**Niranjana S**

B.E. Computer Science and Engineering

GitHub: **Devi359**

---

⭐ If you find this project useful, consider giving the repository a star!
