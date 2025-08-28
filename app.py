import streamlit as st
import pickle
import re
import nltk


nltk.download("punkt")
nltk.download("stopwords")

# loading the models
clf = pickle.load(open("clf.pkl", "rb"))
tfidfd = pickle.load(open("tfidf.pkl", "rb"))


def clean_resume(text):
    import re

    t = text if isinstance(text, str) else str(text)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\bRT\b|\bcc\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"@[A-Za-z0-9_]+", " ", t)
    t = re.sub(r"#[A-Za-z0-9_]+", " ", t)
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r"[^0-9a-zA-Z\s\.,\-\+_\/]", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


# Web app
def main():
    st.title("Resume Screening App with ML")
    upload_file = st.file_uploader("Upload your Resume.", type=["pdf", "txt", "docx"])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode("utf-8")
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode("latin-1")

        cleaned_resume = clean_resume(resume_text)
        cleaned_resume_tfid = tfidfd.transform([cleaned_resume])
        pred_id = clf.predict(cleaned_resume_tfid)[0]

        category_mapping = {
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
        category_name = category_mapping.get(pred_id, "Unknown")
        st.write("Predicted Category: ", category_name)


if __name__ == "__main__":
    main()
