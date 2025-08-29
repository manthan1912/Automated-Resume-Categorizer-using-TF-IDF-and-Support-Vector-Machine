import os
import io
import re
import pickle
import numpy as np
import streamlit as st

# Load the exact artifacts you trained

ART_DIR = "artifacts"
TFIDF_PATH = os.path.join(ART_DIR, "tfidf.pkl")
CLF_PATH = os.path.join(ART_DIR, "clf.pkl")

for p in [TFIDF_PATH, CLF_PATH]:
    if not os.path.exists(p):
        st.error(f"Missing artifact: {p}. Re-train or fix the path.")
        st.stop()

with open(TFIDF_PATH, "rb") as f:
    tfidf = pickle.load(f)
with open(CLF_PATH, "rb") as f:
    clf = pickle.load(f)

# Category mapping (must match training IDs)

ID_TO_CATEGORY = {
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


# Cleaner (identical to training)
def clean_resume(text: str) -> str:
    t = text if isinstance(text, str) else str(text)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\bRT\b|\bcc\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"@[A-Za-z0-9_]+", " ", t)
    t = re.sub(r"#[A-Za-z0-9_]+", " ", t)
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r"[^0-9a-zA-Z\s\.,\-\+_\/]", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


#  text extraction
def read_uploaded_text(upload):
    """Return (text, parser_used) so we can gate on real parsing for PDFs."""
    name = (upload.name or "").lower()

    # ---- PDF
    if name.endswith(".pdf"):
        # 1) PyMuPDF
        try:
            import fitz  # pip install pymupdf

            with fitz.open(stream=upload.read(), filetype="pdf") as doc:
                text = "\n".join((page.get_text() or "") for page in doc)
            return text, "PyMuPDF"
        except Exception:
            upload.seek(0)
        # 2) PyPDF2
        try:
            import PyPDF2  # pip install pypdf2

            reader = PyPDF2.PdfReader(io.BytesIO(upload.read()))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            if text.strip():
                return text, "PyPDF2"
        except Exception:
            upload.seek(0)
        # 3) pdfminer.six
        try:
            from pdfminer.high_level import extract_text  # pip install pdfminer.six

            text = extract_text(io.BytesIO(upload.read()))
            if text.strip():
                return text, "pdfminer"
        except Exception:
            upload.seek(0)

    # ---- DOCX
    if name.endswith(".docx"):
        try:
            import docx  # pip install python-docx

            doc = docx.Document(io.BytesIO(upload.read()))
            return "\n".join(p.text for p in doc.paragraphs), "python-docx"
        except Exception:
            upload.seek(0)

    # ---- Plain text fallback
    data = upload.read()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(enc), f"raw-decode:{enc}"
        except Exception:
            continue
    return "", "raw-decode:failed"


# Streamlit UI (minimal)
def main():
    st.title("Resume Screening App with ML")
    upload_file = st.file_uploader("Upload your Resume", type=["pdf", "txt", "docx"])

    if upload_file is None:
        return

    raw_text, parser_used = read_uploaded_text(upload_file)
    raw_text = raw_text.strip()

    # For PDFs, block if we didn't actually parse text
    if upload_file.name.lower().endswith(".pdf") and not parser_used.startswith(
        ("PyMuPDF", "PyPDF2", "pdfminer")
    ):
        st.error(
            "Could not parse this PDF as text. Install PyMuPDF/PyPDF2/pdfminer or upload a text-based PDF."
        )
        st.stop()

    if not raw_text:
        st.error("No text could be extracted from the file.")
        st.stop()

    cleaned = clean_resume(raw_text)
    token_count = len(cleaned.split())
    if token_count < 40:
        st.error(
            "Too little usable text extracted. If this is a scanned PDF, run OCR first "
            "(e.g., `ocrmypdf input.pdf output.pdf`) or upload a .txt export."
        )
        st.stop()

    # Guard against artifact mismatch (silent failure mode)
    try:
        n_tfidf = tfidf.transform(["probe"]).shape[1]
        n_clf = getattr(clf, "n_features_in_", None)
        if n_clf is not None and n_tfidf != n_clf:
            st.error(
                f"Feature-size mismatch: tfidf={n_tfidf}, classifier expects {n_clf}. "
                "Use the matching pair of pickles from the same training run."
            )
            st.stop()
    except Exception as e:
        st.error(f"Artifact sanity-check failed: {e}")
        st.stop()

    # Predict (single best)
    X = tfidf.transform([cleaned])
    pred_id = int(clf.predict(X)[0])
    category_name = ID_TO_CATEGORY.get(pred_id, f"Unknown ({pred_id})")

    st.subheader("Prediction")
    st.success(f"Predicted Category: {category_name}")


if __name__ == "__main__":
    main()
