# %% [markdown]
# 
# # Resume Category Classifier — TF-IDF + LinearSVC (Notebook)
# 
# This notebook trains a text classifier for resume categories using **TF-IDF + LinearSVC** and saves artifacts **compatible with your `app.py` Streamlit app**:
# 
# - `tfidf.pkl` — fitted `TfidfVectorizer`
# - `clf.pkl` — fitted `LinearSVC` classifier
# 
# > Data path: by default this notebook uses `/mnt/data/data.csv` (already uploaded).  
# > The CSV must have two columns: `Category` (string label) and `Resume` (text).
# 
# Run cells top-to-bottom. When finished, place the generated `tfidf.pkl` and `clf.pkl`
# next to your `app.py` and run your Streamlit app.
# 

# %%

# Imports
import os
import re
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data.csv"   # Change if needed
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Optional plotting (safe to ignore if not installed)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOTS = True
except Exception:
    _HAS_PLOTS = False

print('Using data at:', DATA_PATH)


# %%

# Load data
assert os.path.exists(DATA_PATH), f"Data not found at {DATA_PATH}"
df = pd.read_csv(DATA_PATH)
required_cols = {"Category", "Resume"}
missing = list(required_cols - set(df.columns))
assert not missing, f"CSV missing required columns: {missing}"

df = df[list(required_cols)].dropna()
print(df.shape)
df.head()


# %%

# Optional: Class distribution plots
if _HAS_PLOTS:
    plt.figure(figsize=(12,4))
    order = df["Category"].value_counts().index
    sns.countplot(x="Category", data=df, order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()

df["Category"].value_counts()


# %% [markdown]
# 
# ### Text Cleaning
# 
# The function below fixes a common bug (each `re.sub` must build on the previous result, not the original text).  
# Use the same logic in your `app.py` to ensure compatibility between training and inference.
# 

# %%

CLEAN_PUNCT_KEEP = r"[^0-9a-zA-Z\s\.,\-\+_\/]"

def clean_resume(text: str) -> str:
    # Basic resume cleaner suitable for TF-IDF models.
    if not isinstance(text, str):
        text = str(text)

    t = text
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\bRT\b|\bcc\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"@[A-Za-z0-9_]+", " ", t)
    t = re.sub(r"#[A-Za-z0-9_]+", " ", t)
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(CLEAN_PUNCT_KEEP, " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

# Quick check
print(clean_resume("Check this out: https://example.com @you #hashtag\nNew line\tTabs... OK!"))


# %% [markdown]
# 
# ### Label Mapping (to match your `app.py`)
# 
# Your Streamlit app expects the classifier to output integer IDs which map to category names via a fixed dictionary.
# We will encode labels using that exact mapping so `clf.predict` returns the same integers your app expects.
# 

# %%

# This mapping matches your app.py (id -> category)
id_to_category = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Invert to category -> id
category_to_id = {v: k for k, v in id_to_category.items()}

# Verify that all dataset labels are in the mapping
labels_in_data = set(df["Category"].unique())
labels_in_mapping = set(category_to_id.keys())
missing_in_mapping = labels_in_data - labels_in_mapping
extra_in_mapping = labels_in_mapping - labels_in_data

print("Categories in data:", len(labels_in_data))
print("Missing in mapping (should be empty):", missing_in_mapping)
if missing_in_mapping:
    print("Some labels in your data are not in app.py mapping. Consider updating app.py mapping.")

# Encode y using the fixed mapping
y = df["Category"].map(category_to_id)
assert not y.isna().any(), "Found labels not present in mapping. Please update mapping or data."
X = df["Resume"].astype(str).apply(clean_resume)


# %%

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
len(X_train), len(X_test)


# %%

# TF-IDF vectorizer
tfidf = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
)

Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# LinearSVC classifier
clf = LinearSVC(random_state=RANDOM_STATE)
clf.fit(Xtr, y_train)

# Evaluate
pred = clf.predict(Xte)
acc = (pred == y_test.values).mean()
print(f"Hold-out accuracy: {acc:.4f}")

# Detailed report (sorted by id order)
from sklearn.metrics import classification_report
ids_sorted = sorted(id_to_category.keys())
target_names = [id_to_category[i] for i in ids_sorted]
print("\nClassification report:\n")
print(classification_report(y_test, pred, labels=ids_sorted, target_names=target_names, zero_division=0))


# %%
# ✅ Fixed: 5-fold cross validation without mutating your trained tfidf/clf
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

do_cv = True
if do_cv:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(random_state=RANDOM_STATE)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=skf)
    print(f"5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")


# %%
# Save artifacts (portable)
import os, json, pickle, shutil
from pathlib import Path

ARTIFACT_DIR = Path.cwd() / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

to_save = {
    ARTIFACT_DIR / "tfidf.pkl": tfidf,
    ARTIFACT_DIR / "clf.pkl": clf,
    ARTIFACT_DIR / "label_mapping.json": id_to_category,
}

for path, obj in to_save.items():
    if path.suffix == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

print(f"Saved artifacts to: {ARTIFACT_DIR.resolve()}")

# Optional: mirror into /mnt/data only if it exists and is writable
MNT = Path("/mnt/data")
if MNT.is_dir() and os.access(MNT, os.W_OK):
    for path in to_save.keys():
        shutil.copy2(path, MNT / path.name)
    print("Also mirrored to /mnt/data")


# %%

# Quick sanity prediction
sample_text = "Experienced Python developer with Flask, REST APIs, Pandas and ML model deployment on AWS."
cleaned = clean_resume(sample_text)
vec = tfidf.transform([cleaned])
pred_id = int(clf.predict(vec)[0])
print("Predicted ID:", pred_id, "| Category:", id_to_category.get(pred_id, "Unknown"))


# %% [markdown]
# 
# ### Drop-in fix for `clean_resume` in your `app.py`
# 
# Your current `clean_resume` repeatedly applies `re.sub(..., text)` which discards earlier replacements.
# Use this fixed version (or keep the one defined in this notebook) so the web app's cleaning matches training:
# 
# ```python
# def clean_resume(text):
#     import re
#     t = text if isinstance(text, str) else str(text)
#     t = re.sub(r"http\S+|www\.\S+", " ", t)
#     t = re.sub(r"\bRT\b|\bcc\b", " ", t, flags=re.IGNORECASE)
#     t = re.sub(r"@[A-Za-z0-9_]+", " ", t)
#     t = re.sub(r"#[A-Za-z0-9_]+", " ", t)
#     t = re.sub(r"[\r\n\t]+", " ", t)
#     t = re.sub(r"[^0-9a-zA-Z\s\.,\-\+_\/]", " ", t)
#     t = re.sub(r"\s{2,}", " ", t).strip()
#     return t
# ```
# 

# %%
# In your notebook, with tfidf/clf already in memory:
import io
from pathlib import Path

pdf_path = "E:\\Udemy_AI_course\\ResumeScreeningMLApp\\SAP_Developer_sample_resume.pdf"

try:
    import fitz
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text.append(page.get_text())
    raw = "\n".join(text)
except Exception:
    import PyPDF2
    reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
    raw = "\n".join([(p.extract_text() or "") for p in reader.pages])

cleaned = clean_resume(raw)
vec = tfidf.transform([cleaned])
pred_id = int(clf.predict(vec)[0])
print("Predicted:", pred_id)


# %%
%pip install pymupdf pypdf2


# %%



