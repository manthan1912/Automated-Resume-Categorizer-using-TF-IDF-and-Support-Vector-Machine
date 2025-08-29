# Automated-Resume-Categorizer-using-TF-IDF-and-Support-Vector-Machine

Classify resumes into common job categories using a simple, fast, and reliable NLP pipeline:
**clean text → TF-IDF → LinearSVC**.  
Train the model on your CSV, then run a minimal **Streamlit** app to predict from uploaded resumes (PDF/TXT/DOCX).

---

## 🔧 What’s in this project

- `app1.py` — Minimal Streamlit app (upload a resume → get 1 predicted category)  
  - Robust PDF/DOCX parsing (PyMuPDF → PyPDF2 → pdfminer → text fallback)  
  - Same cleaner as training, guards against low-text/garbage inputs
- `requirements.txt` — All dependencies for training + app
- `artifacts/` — Model files created after training  
  - `tfidf.pkl` (fitted `TfidfVectorizer`)  
  - `clf.pkl` (fitted `LinearSVC`)
- `resume_classifier_notebook.ipynb` — Notebook to train/evaluate and export the two artifacts
- `data.csv` — Your training data (columns: **Category**, **Resume**)

---

## 🚀 Quickstart

```bash
# 1) Create env & install dependencies
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Train the model (open the notebook and run all cells)
# This creates: artifacts/tfidf.pkl and artifacts/clf.pkl

# 3) Run the app
streamlit run app1.py
```

Open the URL shown in the terminal, upload a resume (PDF/TXT/DOCX), and you’ll see one predicted category.

---

## 📦 Data format

`data.csv` must contain:

| Column    | Type   | Notes                         |
|-----------|--------|-------------------------------|
| Category  | string | e.g., `Data Science`, `HR`    |
| Resume    | string | raw resume text               |

The app uses a fixed **ID → label** mapping during training and inference. If you change labels in your data, update both the notebook and `app1.py` mapping accordingly.

Current mapping:

```
0: Advocate
1: Arts
2: Automation Testing
3: Blockchain
4: Business Analyst
5: Civil Engineer
6: Data Science
7: Database
8: DevOps Engineer
9: DotNet Developer
10: ETL Developer
11: Electrical Engineering
12: HR
13: Hadoop
14: Health and fitness
15: Java Developer
16: Mechanical Engineer
17: Network Security Engineer
18: Operations Manager
19: PMO
20: Python Developer
21: SAP Developer
22: Sales
23: Testing
24: Web Designing
```

---

## 🧪 Training & evaluation

Use `resume_classifier_notebook.ipynb`:

- Cleans text (regex-based), vectorizes with TF-IDF (1–2 grams, sublinear tf)
- Trains `LinearSVC` with stratified split
- Optional 5-fold CV (pipeline-based so it doesn’t mutate your saved artifacts)
- Saves `artifacts/tfidf.pkl` and `artifacts/clf.pkl`

> Keep the **same cleaner** in both training and app for consistent results.

---

## 🖥️ Running the app

```bash
streamlit run app1.py
```

- Upload **text-based** PDF, TXT, or DOCX.  
- The app blocks predictions if too little text is extracted (prevents garbage results).

---

## 🧰 Dependencies

See `requirements.txt`. Key ones:

- **ML:** `scikit-learn`, `numpy`, `pandas`
- **App:** `streamlit`
- **Parsing:** `pymupdf` (PyMuPDF), `PyPDF2`, `pdfminer.six`, `python-docx`
- **(Optional) EDA:** `matplotlib`, `seaborn`

Install with:

```bash
pip install -r requirements.txt
```

## 📁 Suggested repo layout

```
.
├── app1.py
├── requirements.txt
├── resume_classifier_notebook.ipynb
├── data.csv
├── artifacts/
│   ├── tfidf.pkl
│   └── clf.pkl
└── sample_resumes/        # optional test PDFs
    ├── Advocate_sample_resume.pdf
    ├── Civil_Engineer_sample_resume.pdf
    ├── SAP_Developer_sample_resume.pdf
    └── Network_Security_Engineer_sample_resume.pdf
```

---
