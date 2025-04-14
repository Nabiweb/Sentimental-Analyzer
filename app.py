from flask import Flask, render_template, request, jsonify
import re
import pickle

# Load model and vectorizer
with open("sentiment_analysis_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Labels (make sure this matches your model's output index)
emo = ['happiness', 'hate', 'love', 'relief', 'sadness', 'surprise', 'worry']

app = Flask(__name__)

def clean_text(text):
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    content = data.get("content", "")
    cleaned = clean_text(content)
    transformed = vectorizer.transform([cleaned])
    prediction_index = model.predict(transformed)[0]
    sentiment = emo[prediction_index]
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
