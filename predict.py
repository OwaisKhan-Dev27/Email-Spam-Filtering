import pickle 
with open("spam_model.pkl","rb") as f: model = pickle.load(f)
with open("tfidf_vectorizer.pkl","rb") as f: vectorizer = pickle.load(f)
input_mail = input("Enter message: ")
input_data = [input_mail]
input_features = vectorizer.transform(input_data)
prediction = model.predict(input_features)
if prediction[0] == 1:
    print("Spam Mail")
else:
    print("Ham Mail")