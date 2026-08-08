import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

categories = {
    'rec.sport.baseball': 'Sports',
    'rec.sport.hockey': 'Sports',
    'talk.politics.misc': 'Politics',
    'talk.politics.guns': 'Politics',
    'comp.graphics': 'Tech',
    'comp.sys.ibm.pc.hardware': 'Tech'
}

data = fetch_20newsgroups(
    subset='all',
    categories=list(categories.keys()),
    remove=('headers', 'footers', 'quotes')
)

df = pd.DataFrame({
    'text': data.data,
    'category': [categories[data.target_names[t]] for t in data.target]
})

df = df[df['text'].str.strip().str.len() > 20]

X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['category'],
    test_size=0.2,
    random_state=42,
    stratify=df['category']
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model saved successfully!")
