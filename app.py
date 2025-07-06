
from flask import Flask, render_template, request
import pickle
from preprocess import clean_text

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['resume']
    cleaned = clean_text(text)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)
    return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
