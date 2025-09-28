import re
import pandas as pd
from rapidfuzz import fuzz, process
import streamlit as st
import pdfplumber
import docx
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ===== DB SETUP =====
DATABASE_URL = "sqlite:///evaluations.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class JobDescription(Base):
    __tablename__ = 'job_descriptions'
    id = Column(Integer, primary_key=True, index=True)
    role_title = Column(String)
    raw_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    evaluations = relationship("Evaluation", back_populates="job_description")

class Evaluation(Base):
    __tablename__ = 'evaluations'
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey('job_descriptions.id'))
    candidate_location = Column(String)
    filename = Column(String)
    job_skills = Column(Text)
    resume_skills = Column(Text)
    score = Column(Float)
    verdict = Column(String)
    feedback = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    job_description = relationship("JobDescription", back_populates="evaluations")

Base.metadata.create_all(bind=engine)

# ===== SKILLS DATABASE =====
SKILLS_DB = ["python", "java", "c++", "sql", "machine learning", "data analysis",
             "project management", "communication", "excel", "aws", "docker", "tensorflow"]

# ===== UTILS =====

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        texts = [page.extract_text() or "" for page in pdf.pages]
    return '\n'.join(texts)

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill in SKILLS_DB if skill in text_lower]

def extract_jd_sections(text):
    # Simple: first non-empty line is title; rest skills by keyword
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    role_title = lines[0] if lines else "Unknown Role"
    must_have, good_have = [], []
    section = None
    for line in lines[1:]:
        if re.search(r'(must have|required skills)', line, re.I):
            section = 'must_have'
        elif re.search(r'(good to have|preferred skills)', line, re.I):
            section = 'good_have'
        elif not line or re.match(r'^[A-Z]', line):
            section = None
        elif section == 'must_have':
            must_have.extend([s.strip() for s in re.split(r'[,;]', line) if s.strip()])
        elif section == 'good_have':
            good_have.extend([s.strip() for s in re.split(r'[,;]', line) if s.strip()])
    return {'role_title': role_title,
            'must_have_skills': list(set(must_have)),
            'good_have_skills': list(set(good_have))}

def fuzzy_match(resume_skills, job_skills, threshold=80):
    matched = set()
    for rskill in resume_skills:
        res = process.extractOne(rskill, job_skills, scorer=fuzz.token_sort_ratio)
        if res and res[1] >= threshold:
            matched.add(res[0])
    return list(matched)

def save_jd(session, role_title, raw_text):
    jd = JobDescription(role_title=role_title, raw_text=raw_text)
    session.add(jd)
    session.commit()
    session.refresh(jd)
    return jd

def save_evaluation(session, data):
    try:
        ev = Evaluation(**data)
        session.add(ev)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e

def get_jds(session):
    return session.query(JobDescription).order_by(JobDescription.created_at.desc()).all()

def get_evaluations(session, job_id):
    return session.query(Evaluation).filter(Evaluation.job_id == job_id).order_by(Evaluation.score.desc()).all()

# ===== STREAMLIT APP =====

def main():
    st.title("Simple Automated Resume Checker")

st.markdown("""
### How to Use This Application

1. **Upload Job Description:**  
   - Go to the **Job Descriptions** tab.  
   - Upload a Job Description file in PDF or DOCX format.  
   - Review the extracted role title and skills, then save the Job Description.

2. **Check Resumes Against Job Description:**  
   - Switch to the **Resume Checker** tab.  
   - Select a previously saved Job Description from the dropdown.  
   - Upload one or more candidate resumes in PDF or DOCX format.  
   - Click the **Evaluate** button to see the suitability score and feedback.

3. **View Shortlisted Candidates:**  
   - Visit the **Shortlisted Resumes** tab.  
   - Select a Job Description to view evaluated candidates with scores and feedback.  
   - You can download the shortlisted candidates’ data as a CSV file.

---

> **Note:**  
> This app uses simple keyword matching and fuzzy matching techniques to evaluate resumes relative to job descriptions. It's recommended to upload clear job descriptions and resumes for best results.
""")
    st.set_page_config(page_title="Simple Resume Checker", layout="wide")
    st.title("Simple Automated Resume Checker")

    session = SessionLocal()

    menu = ["Job Descriptions", "Resume Checker", "Shortlisted Resumes"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Job Descriptions":
        st.header("Upload Job Description")
        jd_file = st.file_uploader("Upload JD (PDF/DOCX)", type=["pdf", "docx"])
        if jd_file:
            text = extract_text_from_pdf(jd_file) if jd_file.type == "application/pdf" else extract_text_from_docx(jd_file)
            jd_info = extract_jd_sections(text)
            st.write(f"Role Title: {jd_info['role_title']}")
            if st.button("Save JD"):
                jd = save_jd(session, jd_info['role_title'], text)
                st.success(f"Saved Job Description with ID {jd.id}")

    elif choice == "Resume Checker":
        st.header("Resume Evaluation")
        jds = get_jds(session)
        jd_map = {jd.role_title: jd.id for jd in jds}
        if not jds:
            st.warning("No job descriptions found. Please upload first.")
            return
        selected_jd = st.selectbox("Select Job Description", list(jd_map.keys()))
        jd_id = jd_map[selected_jd]
        uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", accept_multiple_files=True, type=["pdf", "docx"])
        if uploaded_files and st.button("Evaluate"):
            jd_record = session.query(JobDescription).filter(JobDescription.id == jd_id).first()
            jd_info = extract_jd_sections(jd_record.raw_text)
            job_skills = jd_info['must_have_skills'] + jd_info['good_have_skills']
            for f in uploaded_files:
                resume_text = extract_text_from_pdf(f) if f.type == "application/pdf" else extract_text_from_docx(f)
                resume_skills = extract_skills(resume_text)
                matched = fuzzy_match(resume_skills, job_skills)
                score = int(100 * len(matched) / len(job_skills) if job_skills else 0)
                verdict = "High" if score > 75 else "Medium" if score > 50 else "Low"
                feedback = f"Matched skills: {', '.join(matched)}"
                data = {
                    "timestamp": datetime.now(),
                    "job_id": jd_id,
                    "candidate_location": "Unknown",
                    "filename": f.name,
                    "job_skills": ", ".join(job_skills),
                    "resume_skills": ", ".join(resume_skills),
                    "score": score,
                    "verdict": verdict,
                    "feedback": feedback
                }
                save_evaluation(session, data)
                st.write(f"File: {f.name} | Score: {score} | Verdict: {verdict}")
                st.write(f"Feedback: {feedback}")

    elif choice == "Shortlisted Resumes":
        st.header("Shortlisted Resumes")
        jds = get_jds(session)
        jd_map = {jd.role_title: jd.id for jd in jds}
        if not jds:
            st.warning("No job descriptions found.")
            return
        selected_jd = st.selectbox("Select Job Description", list(jd_map.keys()))
        jd_id = jd_map[selected_jd]
        evaluations = get_evaluations(session, jd_id)
        if not evaluations:
            st.info("No evaluations found")
            return
        df = pd.DataFrame([{
            "Timestamp": ev.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "File": ev.filename,
            "Score": ev.score,
            "Verdict": ev.verdict,
            "Feedback": ev.feedback,
        } for ev in evaluations])
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name=f"shortlisted_{selected_jd}.csv")

if __name__ == "__main__":
    main()
