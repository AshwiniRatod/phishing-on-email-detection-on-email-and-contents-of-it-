import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# DATASET LOCATION
# =========================================================

DATA_FOLDER = "data"


DATASETS = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "phishing_email.csv",
    "SpamAssasin.csv"
]


# =========================================================
# CONVERT DIFFERENT DATASET FORMATS INTO TEXT
# =========================================================

def create_text(df):

    # Dataset containing text_combined
    if "text_combined" in df.columns:

        text = df["text_combined"].fillna("").astype(str)

    # Datasets containing subject + body
    elif "subject" in df.columns and "body" in df.columns:

        subject = df["subject"].fillna("").astype(str)
        body = df["body"].fillna("").astype(str)

        text = subject + " " + body

    else:

        raise ValueError(
            "Dataset format not recognized."
        )

    return text


# =========================================================
# LOAD ALL DATASETS
# =========================================================

all_data = []


print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)


for filename in DATASETS:

    path = os.path.join(DATA_FOLDER, filename)

    print(f"\nLoading: {filename}")

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Create temporary text representation
    text = create_text(df)

    # Create temporary dataframe
    temp = pd.DataFrame({
        "text": text,
        "label": df["label"],
        "dataset": filename
    })

    all_data.append(temp)


# =========================================================
# COMBINE IN MEMORY
# =========================================================

data = pd.concat(
    all_data,
    ignore_index=True
)


print("\n" + "=" * 60)
print("COMBINED DATASET")
print("=" * 60)

print("Total emails:", len(data))

print("\nClass distribution:")
print(data["label"].value_counts())

print("\nDataset distribution:")
print(data["dataset"].value_counts())


# =========================================================
# REMOVE ONLY EMPTY TEXT FROM MEMORY
# =========================================================

data = data[
    data["text"].str.strip() != ""
]


# =========================================================
# INPUT AND TARGET
# =========================================================

X = data["text"]

y = data["label"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# TF-IDF
# =========================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print(
    "TF-IDF training shape:",
    X_train_tfidf.shape
)


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

print("\nTraining Logistic Regression...")


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


model.fit(
    X_train_tfidf,
    y_train
)


# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(
    X_test_tfidf
)


# =========================================================
# EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Phishing"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "phishing_model.pkl"
)


joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)


print("\nModel saved successfully.")

print("Created:")
print("  phishing_model.pkl")
print("  tfidf_vectorizer.pkl")