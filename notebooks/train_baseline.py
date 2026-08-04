import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load your labelled validation set
df = pd.read_csv("data/validation_labelled.csv")

# Drop any rows missing labels, just in case
df = df.dropna(subset=["true_category", "true_urgency"])

print(f"Training on {len(df)} labelled tickets")

X = df["ticket_text"]

# --- Category model ---
y_category = df["true_category"]

vectorizer_cat = TfidfVectorizer(max_features=200, stop_words="english")
X_vec_cat = vectorizer_cat.fit_transform(X)

category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_vec_cat, y_category)

# --- Urgency model ---
y_urgency = df["true_urgency"]

vectorizer_urg = TfidfVectorizer(max_features=200, stop_words="english")
X_vec_urg = vectorizer_urg.fit_transform(X)

urgency_model = LogisticRegression(max_iter=1000)
urgency_model.fit(X_vec_urg, y_urgency)

# Save models and vectorizers for reuse
joblib.dump(category_model, "app/category_model.pkl")
joblib.dump(vectorizer_cat, "app/vectorizer_cat.pkl")
joblib.dump(urgency_model, "app/urgency_model.pkl")
joblib.dump(vectorizer_urg, "app/vectorizer_urg.pkl")

print("Baseline models trained and saved.")