import pandas as pd
import string
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Download necessary NLTK data files (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Paths and columns - update path to your dataset
csv_path = r"data\home\sdf\dataset.csv"

# Choose columns for text and label
text_col = "Job Title"      # column with text data to train on
label_col = "Categories"    # target labels column

# Load dataset
df = pd.read_csv(csv_path)
print(f"📄 Columns in dataset: {df.columns.tolist()}")

# Drop rows with missing or empty text or label
df[text_col] = df[text_col].astype(str).str.strip()
df[label_col] = df[label_col].astype(str).str.strip()
df_clean = df[(df[text_col] != "") & (df[label_col] != "")]
print(f"✅ Non-empty rows before cleaning: {len(df)}")
print(f"✅ Rows after dropping empty '{text_col}' and '{label_col}': {len(df_clean)}")

# Preprocessing function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = wordpunct_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

print("🧼 Preprocessing text...")
df_clean[text_col] = df_clean[text_col].apply(preprocess)

# Features and labels
X = df_clean[text_col].values
y = df_clean[label_col].values

# Vectorize text data using TF-IDF for better performance
vectorizer = TfidfVectorizer(max_features=10000)
X_vec = vectorizer.fit_transform(X)
print(f"Vectorized data shape: {X_vec.shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training completed.")

# Evaluate model on test data
y_pred = model.predict(X_test)
print("📊 Classification report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model/ai_vs_human_model.joblib")
joblib.dump(vectorizer, "model/ai_vs_human_vectorizer.joblib")
print("💾 Model and vectorizer saved to 'model/' directory.")
