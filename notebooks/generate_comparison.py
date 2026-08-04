import pandas as pd
import joblib

# Load classification results (has ticket_text, true labels, LLM predictions)
df = pd.read_csv("data/classification_results.csv")

# Load baseline models
category_model = joblib.load("app/category_model.pkl")
vectorizer_cat = joblib.load("app/vectorizer_cat.pkl")
urgency_model = joblib.load("app/urgency_model.pkl")
vectorizer_urg = joblib.load("app/vectorizer_urg.pkl")

# Predict category and urgency using baseline model for ALL tickets
X_cat = vectorizer_cat.transform(df["ticket_text"])
df["baseline_category"] = category_model.predict(X_cat)

X_urg = vectorizer_urg.transform(df["ticket_text"])
df["baseline_urgency"] = urgency_model.predict(X_urg)

# Save combined results
df.to_csv("data/full_comparison.csv", index=False)

print("Done. Saved to data/full_comparison.csv")
print(df[["ticket_id", "true_category", "predicted_category", "baseline_category",
          "true_urgency", "predicted_urgency", "baseline_urgency"]].head(10))