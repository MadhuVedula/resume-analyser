import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Paths and column names - adjust as needed
csv_path = r"C:\Users\madhu\resume_analyser\data\home\sdf\dataset.csv"

# Pick the text column you want to train on (use a non-empty one)
text_col = "Job Title"           # example: use "Job Title" instead of "Job Description"
label_col = "Categories"         # example label column - adjust if needed

# Load data
data = pd.read_csv(csv_path)
print("Columns found in dataset:", data.columns.tolist())

# Check initial row counts
print("Total rows in dataset:", len(data))
print(f"Rows with non-null '{text_col}':", data[text_col].notnull().sum())
print(f"Rows with non-null '{label_col}':", data[label_col].notnull().sum())

# Drop rows where text or label is null or empty (after stripping)
data[text_col] = data[text_col].astype(str).str.strip()
data[label_col] = data[label_col].astype(str).str.strip()

data_clean = data[(data[text_col] != "") & (data[label_col] != "")]
print("Number of rows after cleaning:", len(data_clean))

if len(data_clean) == 0:
    raise ValueError(f"No data left after cleaning '{text_col}' and '{label_col}' columns.")

# Extract features and labels
X = data_clean[text_col].values
y = data_clean[label_col].values

print("Sample text data:")
print(X[:5])
print("Sample labels:")
print(y[:5])

# Vectorize text data
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

print(f"Vectorized data shape: {X_vec.shape}")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vec, y)

print("Model training completed.")

# Save the model and vectorizer to disk
joblib.dump(model, "model/rf_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")

print("Model and vectorizer saved to 'rf_model.joblib' and 'vectorizer.joblib'")
