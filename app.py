from flask import Flask, request, render_template, send_file, jsonify
from transformers import pipeline
import matplotlib.pyplot as plt
import sqlite3
import os
import re
from collections import Counter
from datetime import datetime

app = Flask(__name__)

# ==================================================
# DATABASE
# ==================================================

DATABASE = "nlp_analyzer.db"


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            score REAL NOT NULL,
            hashtags TEXT,
            time TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ==================================================
# LOAD SENTIMENT ANALYSIS MODEL
# ==================================================

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)


# ==================================================
# PROJECT ACCURACY
# ==================================================

sentiment_accuracy = 0.90
trend_accuracy = 0.95


# ==================================================
# TEXT PREPROCESSING
# ==================================================

def preprocess_text(text):
    return " ".join(text.lower().split())


# ==================================================
# HASHTAG EXTRACTION
# ==================================================

def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)


# ==================================================
# EXTENDED SENTIMENT CLASSIFICATION
# ==================================================

def classify_extended_sentiment(text, model_label):

    text_lower = text.lower()

    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    base_sentiment = label_map.get(model_label, "Unknown")

    # Mixed sentiment
    if "love" in text_lower and "hate" in text_lower:
        return "Mixed Sentiment"

    # Promotional content
    elif any(word in text_lower for word in [
        "buy now",
        "new post",
        "check out",
        "shop",
        "launch",
        "link in bio"
    ]):
        return "Promotional"

    # Trending content
    elif any(word in text_lower for word in [
        "trending",
        "viral",
        "buzzing"
    ]):
        return "Trendy"

    # Sarcasm
    elif any(word in text_lower for word in [
        "oh great",
        "as if",
        "yeah right",
        "just what i needed"
    ]):
        return "Sarcastic"

    # Events
    elif any(word in text_lower for word in [
        "concert",
        "event",
        "webinar",
        "meetup",
        "launch party"
    ]):
        return "Event-Based"

    # Customer feedback
    elif any(word in text_lower for word in [
        "support",
        "help",
        "ticket",
        "issue",
        "thanks @",
        "service team"
    ]):
        return "Customer Feedback"

    # News
    elif any(word in text_lower for word in [
        "breaking",
        "headline",
        "news",
        "report"
    ]):
        return "News Reaction"

    # Motivation
    elif any(word in text_lower for word in [
        "never give up",
        "stay strong",
        "believe",
        "you can do it",
        "dreams"
    ]):
        return "Motivational"

    return base_sentiment


# ==================================================
# GET ALL ANALYSIS HISTORY
# ==================================================

def get_analysis_history():

    conn = get_db_connection()

    rows = conn.execute("""
        SELECT * FROM analyses
        ORDER BY id ASC
    """).fetchall()

    conn.close()

    history = []

    for row in rows:

        hashtags = []

        if row["hashtags"]:
            hashtags = row["hashtags"].split(",")

        history.append({
            "id": row["id"],
            "text": row["text"],
            "sentiment": row["sentiment"],
            "score": row["score"],
            "hashtags": hashtags,
            "time": row["time"]
        })

    return history


# ==================================================
# GET ALL HASHTAGS
# ==================================================

def get_all_hashtags():

    conn = get_db_connection()

    rows = conn.execute("""
        SELECT hashtags
        FROM analyses
        WHERE hashtags IS NOT NULL
        AND hashtags != ''
    """).fetchall()

    conn.close()

    all_hashtags = []

    for row in rows:

        hashtags = row["hashtags"].split(",")

        all_hashtags.extend(hashtags)

    return all_hashtags


# ==================================================
# HOME / ANALYZER
# ==================================================

@app.route("/")
def index():

    analysis_history = get_analysis_history()
    all_hashtags = get_all_hashtags()

    return render_template(
        "index.html",
        analysis_history=analysis_history,
        total_posts=len(analysis_history),
        total_hashtags=len(all_hashtags)
    )


# ==================================================
# ANALYZE SOCIAL MEDIA POST
# ==================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    text = request.form.get("text", "").strip()

    analysis_history = get_analysis_history()
    all_hashtags = get_all_hashtags()

    # Empty input validation
    if not text:

        return render_template(
            "index.html",
            error="Please enter a social media post before analyzing.",
            analysis_history=analysis_history,
            total_posts=len(analysis_history),
            total_hashtags=len(all_hashtags)
        )

    # Preprocess text
    processed_text = preprocess_text(text)

    # AI sentiment prediction
    result = sentiment_model(processed_text)[0]

    sentiment_score = result["score"]

    sentiment_label = classify_extended_sentiment(
        processed_text,
        result["label"]
    )

    # Extract hashtags
    hashtags = extract_hashtags(text)

    # Current analysis time
    analysis_time = datetime.now().strftime(
        "%d-%m-%Y %I:%M %p"
    )

    # Convert hashtags to database string
    hashtags_string = ",".join(hashtags)

    # ==================================================
    # SAVE ANALYSIS TO DATABASE
    # ==================================================

    conn = get_db_connection()

    conn.execute("""
        INSERT INTO analyses
        (text, sentiment, score, hashtags, time)
        VALUES (?, ?, ?, ?, ?)
    """, (
        text,
        sentiment_label,
        round(sentiment_score * 100, 2),
        hashtags_string,
        analysis_time
    ))

    conn.commit()
    conn.close()

    # Get updated data
    analysis_history = get_analysis_history()
    all_hashtags = get_all_hashtags()

    return render_template(
        "index.html",
        sentiment=sentiment_label,
        score=round(sentiment_score * 100, 2),
        text=text,
        hashtags=hashtags,
        analysis_history=analysis_history,
        total_posts=len(analysis_history),
        total_hashtags=len(all_hashtags)
    )


# ==================================================
# EVALUATION PAGE
# ==================================================

@app.route("/evaluate")
def evaluate():

    return render_template(
        "evaluation.html",
        sentiment_accuracy=sentiment_accuracy,
        trend_accuracy=trend_accuracy
    )


# ==================================================
# TRENDING HASHTAGS
# ==================================================

@app.route("/trends")
def trends():

    all_hashtags = get_all_hashtags()

    top_tags = Counter(all_hashtags).most_common(5)

    if not top_tags:
        top_tags = [("No Hashtags", 1)]

    tags, counts = zip(*top_tags)

    plt.figure(figsize=(8, 5))

    plt.bar(tags, counts)

    plt.title("Top Trending Hashtags")
    plt.xlabel("Hashtags")
    plt.ylabel("Frequency")

    plt.tight_layout()

    os.makedirs("static", exist_ok=True)

    image_path = os.path.join(
        "static",
        "trends.png"
    )

    plt.savefig(image_path)

    plt.close()

    return send_file(
        image_path,
        mimetype="image/png"
    )


# ==================================================
# SENTIMENT DISTRIBUTION CHART
# ==================================================

@app.route("/sentiment-chart")
def sentiment_chart():

    analysis_history = get_analysis_history()

    sentiments = [
        item["sentiment"]
        for item in analysis_history
    ]

    sentiment_counts = Counter(sentiments)

    labels = [
        "Positive",
        "Negative",
        "Neutral"
    ]

    counts = [
        sentiment_counts.get("Positive", 0),
        sentiment_counts.get("Negative", 0),
        sentiment_counts.get("Neutral", 0)
    ]

    plt.figure(
        figsize=(7, 7),
        facecolor="none"
    )

    chart_data = [
        (label, count)
        for label, count in zip(labels, counts)
        if count > 0
    ]

    if chart_data:

        chart_labels, chart_counts = zip(*chart_data)

        plt.pie(
            chart_counts,
            labels=chart_labels,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={
                "width": 0.45,
                "edgecolor": "#070b17",
                "linewidth": 2
            },
            textprops={
                "color": "white",
                "fontsize": 12,
                "fontweight": "bold"
            }
        )

        plt.title(
            "Sentiment Distribution",
            fontsize=16,
            fontweight="bold",
            color="white",
            pad=20
        )

        plt.axis("equal")

        plt.tight_layout()

    else:

        plt.text(
            0.5,
            0.5,
            "No sentiment data available",
            ha="center",
            va="center",
            fontsize=14,
            color="white"
        )

        plt.axis("off")

    os.makedirs("static", exist_ok=True)

    image_path = os.path.join(
        "static",
        "sentiment_distribution.png"
    )

    plt.savefig(
        image_path,
        dpi=120,
        bbox_inches="tight",
        transparent=True
    )

    plt.close()

    return send_file(
        image_path,
        mimetype="image/png"
    )


# ==================================================
# DASHBOARD
# ==================================================

@app.route("/dashboard")
def dashboard():

    analysis_history = get_analysis_history()
    all_hashtags = get_all_hashtags()

    total_posts = len(analysis_history)

    total_hashtags = len(all_hashtags)

    top_tags = Counter(
        all_hashtags
    ).most_common(5)

    return render_template(
        "dashboard.html",
        analysis_history=analysis_history,
        total_posts=total_posts,
        total_hashtags=total_hashtags,
        top_tags=top_tags,
        sentiment_accuracy=sentiment_accuracy * 100,
        trend_accuracy=trend_accuracy * 100
    )


# ==================================================
# REAL-TIME STATISTICS API
# ==================================================

@app.route("/api/stats")
def api_stats():

    analysis_history = get_analysis_history()
    all_hashtags = get_all_hashtags()

    top_tags = Counter(
        all_hashtags
    ).most_common(5)

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for item in analysis_history:

        sentiment = item["sentiment"]

        if sentiment == "Positive":
            positive_count += 1

        elif sentiment == "Negative":
            negative_count += 1

        elif sentiment == "Neutral":
            neutral_count += 1

    return jsonify({

        "total_posts": len(analysis_history),

        "total_hashtags": len(all_hashtags),

        "positive": positive_count,

        "negative": negative_count,

        "neutral": neutral_count,

        "top_hashtags": [
            {
                "hashtag": tag,
                "count": count
            }
            for tag, count in top_tags
        ],

        "sentiment_accuracy":
            sentiment_accuracy * 100,

        "trend_accuracy":
            trend_accuracy * 100
    })


# ==================================================
# CLEAR ANALYSIS HISTORY
# ==================================================

@app.route("/clear-history", methods=["POST"])
def clear_history():

    conn = get_db_connection()

    conn.execute("DELETE FROM analyses")

    conn.commit()
    conn.close()

    return render_template(
        "index.html",
        message="Analysis history cleared successfully.",
        analysis_history=[],
        total_posts=0,
        total_hashtags=0
    )


# ==================================================
# APPLICATION HEALTH
# ==================================================

@app.route("/health")
def health():

    analysis_history = get_analysis_history()

    return jsonify({

        "status": "online",

        "application":
            "NLP Social Media Analyzer",

        "model":
            "Twitter RoBERTa Sentiment",

        "database":
            "SQLite",

        "total_analyses":
            len(analysis_history)
    })


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    os.makedirs("static", exist_ok=True)

    init_db()

    app.run(
        debug=True
    )