import pandas as pd
import joblib  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- PRE-REQUISITE: LOAD DATA ---
# Load the CLEANED data from Week 1
df = pd.read_csv('../data/cleaned_toxic_comments.csv')
df = df.dropna(subset=['cleaned_text']) # Remove any empty rows after cleaning

# (i) Vectorize data
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text']) # Removed .toarray() to save memory
y = df['is_toxic'] # Use the binary label we created in Week 1

# (ii) Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# (iii) Performance Metrics
y_pred = classifier.predict(X_test)

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# (iv) ADDITION: Save the model and vectorizer for Deployment (Week 3)

joblib.dump(classifier, 'toxic_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("\nModel and Vectorizer saved successfully!")

# Optional
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('../reports/confusion_matrix.png')
