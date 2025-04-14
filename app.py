from flask import Flask, render_template, request, jsonify
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Define the list of sentiment categories based on your original mapping.
emo = ['happiness', 'hate', 'love', 'relief', 'sadness', 'surprise', 'worry']

# Function to clean input text; same as in your original code.
def clean_text(text):
    text = re.sub(r'@[\w]+', '', text)  # Remove mentions
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Global variables that will store the trained model and vectorizer.
vectorizer = None
model = None

# Train the model when the Flask application starts.
@app.before_request
def load_model():
    global vectorizer, model
    # Read CSV file (ensure tweet_emotions.csv is in the same folder or adjust the path)
    df = pd.read_csv("tweet_emotions.csv")
    
    # Filter out unwanted classes
    classes_to_remove = ['anger', 'boredom', 'empty', 'neutral', 'fun', 'enthusiasm']
    df = df[~df['sentiment'].isin(classes_to_remove)]
    
    # Clean the text content
    df['content'] = df['content'].apply(clean_text)
    
    # Encode the sentiment labels
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
    
    # Train a TF-IDF vectorizer using all available content from the CSV
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df['content'])
    
    # Train the classifier (using Logistic Regression here)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, df['sentiment'])
    
    # You can print out the classes if needed for debugging:
    print("Label classes:", label_encoder.classes_)

# Home route renders the main page.
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Prediction endpoint which receives JSON input and returns the predicted sentiment.
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    content = data.get("content", "")
    cleaned = clean_text(content)
    text_transform = vectorizer.transform([cleaned])
    pred_index = model.predict(text_transform)[0]
    
    # Map the predicted index to the sentiment label.
    sentiment = emo[pred_index]
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)