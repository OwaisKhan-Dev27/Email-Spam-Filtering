import pandas as pd
df = pd.read_csv("dataset/spam.csv",
encoding='latin-1')
df = df[['v1','v2']]
df.columns = ['label','message']
df['label'] = df['label'].map({'ham':0,'spam':1})
df = df.reset_index(drop=True)
print(df.head(10))
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
X = df['message']
Y = df['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_tfidf, Y_train)
Y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Confusion Matrix:\n",
confusion_matrix(Y_test, Y_pred))
input_mail = input("Enter a message tocheck spam or ham:")
input_data = [input_mail]
input_data_features = vectorizer.transform(input_data)
prediction = model.predict(input_data_features)
if prediction[0] == 1:
    print("Spam Mail")
else:
    print("Ham Mail")
import pickle
with open("spam_model.pkl","wb") as f: pickle.dump(model, f)
with open("tfidf_vectorizer.pkl","wb") as f: pickle.dump(vectorizer, f)
print("Model and vectorizer saved successfully")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, Y_train)
lr_pred = lr.predict(X_test_tfidf)
print("\nLogistic Regression Accuracy:",accuracy_score(Y_test, lr_pred))
import pickle
pickle.dump(model, open("spam_model.pkl","wb"))