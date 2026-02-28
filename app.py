from flask import Flask, render_template, request
import pickle

app = Flask(_name_)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    data = [message]
    data_features = vectorizer.transform(data)
    prediction = model.predict(data_features)

    if prediction[0] == 1:
        result = "Spam"
    else:
        result = "Ham"

    return render_template("index.html", prediction_text=result)

if _name_ == "_main_":
    app.run(debug=True)