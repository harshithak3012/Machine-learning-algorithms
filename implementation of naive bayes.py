import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = pd.read_csv("D:/Harshitha docs/machine learning/data.csv")
len(data)
X=data['Sentence']
Y=data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)
y_pred = nb_classifier.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')