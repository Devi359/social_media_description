from flask import Flask, request, render_template, send_file 
from transformers import pipeline 
import matplotlib.pyplot as plt 
import os 
import re 
from collections import Counter 
app = Flask(__name__) 
# Load sentiment model 
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter
roberta-base-sentiment") 
# Dummy accuracy values 
sentiment_accuracy = 0.90 
trend_accuracy = 0.95 
# Store hashtags for trend graph 
all_hashtags = [] 
# Preprocessing 
def preprocess_text(text): 
    return ' '.join(text.lower().split()) 
def extract_hashtags(text): 
    return re.findall(r"#(\w+)", text) 
def classify_extended_sentiment(text, model_label): 
    text_lower = text.lower() 
    label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 
'Positive'} 
    base_sentiment = label_map.get(model_label, 'Unknown') 
    # Additional sentiment heuristics 
    if "love" in text_lower and "hate" in text_lower: 
        return "Mixed Sentiment" 
    elif any(w in text_lower for w in ["buy now", "new post", "check out", 
"shop", "launch", "link in bio"]): 
        return "Promotional" 
    elif any(w in text_lower for w in ["trending", "viral", "buzzing"]): 
        return "Trendy" 
    elif any(w in text_lower for w in ["oh great", "as if", "yeah right", "just what i 
needed"]): 
        return "Sarcastic" 
    elif any(w in text_lower for w in ["concert", "event", "webinar", "meetup", 
"launch party"]): 
        return "Event-Based" 
    elif any(w in text_lower for w in ["support", "help", "ticket", "issue", "thanks 
@", "service team"]): 
        return "Customer Feedback" 
    elif any(w in text_lower for w in ["breaking", "headline", "news", "report"]): 
        return "News Reaction" 
    elif any(w in text_lower for w in ["never give up", "stay strong", "believe", 
"you can do it", "dreams"]): 
        return "Motivational" 
        return base_sentiment 
@app.route('/') 
def index(): 
    return render_template('index.html') 
@app.route('/analyze', methods=['POST']) 
def analyze(): 
    text = request.form.get("text", "") 
    processed_text = preprocess_text(text) 
  result = sentiment_model(processed_text)[0] 
    sentiment_score = result['score'] 
    sentiment_label =classify_extended_sentiment(processed_text, result['label']) 
    hashtags = extract_hashtags(text) 
    all_hashtags.extend(hashtags) 
  return render_template('index.html', sentiment=sentiment_label, 
score=sentiment_score, text=text, hashtags=hashtags) 
@app.route('/evaluate') 
def evaluate(): 
    return-render_template('evaluation.html', 
sentiment_accuracy=sentiment_accuracy, trend_accuracy=trend_accuracy) 
@app.route('/trends') 
def trends(): 
    # Count the frequency of each hashtag 
    top_tags = Counter(all_hashtags).most_common(5) 
    if not top_tags: 
        top_tags = [("None", 1)] 
        tags, counts = zip(*top_tags) 
# Generate the bar chart for trending hashtag 
   plt.figure(figsize=(6, 4)) 
    plt.bar(tags, counts, color="#2980b9") 
    plt.title("Top Trending Hashtags") 
    plt.xlabel("Hashtags") 
    plt.ylabel("Frequency") 
    plt.tight_layout() 
  # Save the graph as a static image 
    image_path = "static/trends.png" 
    plt.savefig(image_path) 
    plt.close() 
  # Return the image as a response 
    return send_file(image_path, mimetype='image/png') 
if __name__ == '__main__': 
    os.makedirs('static', exist_ok=True) 
    app.run(debug=True)