import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from rapidfuzz import fuzz, process
import streamlit as st
import pdfplumber
import docx
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship


# Database setup
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


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
feedback_generator = pipeline('text-generation', model='gpt2')


SKILLS_DB = ["python", "java", "c++", "sql", "machine learning", "data analysis",
             "project management", "communication", "excel", "aws", "docker", "tensorflow"]


def extract_text_from_pdf_with_header_footer_removal(pdf_file):
    pages_text = []
    headers = set()
    footers = set()
    all_pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            page_lines = page_text.split('\n')
            all_pages.append(page_lines)
            if len(page_lines) > 0:
                headers.add(page_lines[0])
                footers.add(page_lines[-1])
    header_candidates = [h for h in headers if sum(h == page[0] for page in all_pages if page) > 1]
    footer_candidates = [f for f in footers if sum(f == page[-1] for page in all_pages if page) > 1]
    clean_text = []
    for lines in all_pages:
        if lines and lines[0] in header_candidates:
            lines = lines[1:]
        if lines and lines[-1] in footer_candidates:
            lines = lines[:-1]
        clean_text.extend(lines)
    return '\n'.join(clean_text)


def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)


def extract_skills_rule_based(text):
    text_lower = text.lower()
    skills_found = [skill for skill in SKILLS_DB if skill in text_lower]
    return skills_found


def extract_jd_sections(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    role_title = lines[0] if lines else "Unknown Role"
    must_have_skills = []
    good_have_skills = []

    must_have_pattern = re.compile(r'(must[\s-]?have skills|required skills)', re.I)
    good_have_pattern = re.compile(r'(good[\s-]?to[\s-]?have skills|preferred skills)', re.I)

    current_section = None
    for line in lines[1:]:
        if must_have_pattern.search(line):
            current_section = 'must_have'
            continue
        elif good_have_pattern.search(line):
            current_section = 'good_have'
            continue
        elif re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*:$', line):
            current_section = None
            continue

        if current_section == 'must_have':
            must_have_skills.extend([s.strip() for s in re.split(r'[,;•-]', line) if s.strip()])
        elif current_section == 'good_have':
            good_have_skills.extend([s.strip() for s in re.split(r'[,;•-]', line) if s.strip()])

    must_have_skills = list(set(must_have_skills))
    good_have_skills = list(set(good_have_skills))

    return {
        'role_title': role_title,
        'must_have_skills': must_have_skills,
        'good_have_skills': good_have_skills
    }


def extract_location_from_resume(text):
    lines = text.lower().split('\n')
    location = None
    keywords = ['location', 'address', 'city', 'place']
    for line in lines[:20]:
        for kw in keywords:
            if kw in line:
                loc = re.split(r':|,', line)
                if len(loc) > 1:
                    location = loc[-1].strip().title()
                    return location
    return "Unknown"


def compute_semantic_similarity(text1, text2):
    emb1 = embedding_model.encode(text1, convert_to_tensor=True)
    emb2 = embedding_model.encode(text2, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(emb1, emb2).item()
    return cosine_score


def fuzzy_skill_match(resume_skills, jd_skills, threshold=80):
    matched_skills = set()
    for r_skill in resume_skills:
        result = process.extractOne(r_skill, jd_skills, scorer=fuzz.token_sort_ratio)
        if result is None:
            continue
        match, score, _ = result
        if score >= threshold:
            matched_skills.add(match)
    return list(matched_skills)


def generate_dynamic_feedback(job_desc, resume_text, missing_elements):
    prompt = f"""You are an expert career advisor. Given this job description: "{job_desc}" and this candidate resume summary: "{resume_text}", the candidate is missing the following important skills or projects: {missing_elements}. Provide constructive, personalized feedback on how the candidate can improve their resume to better fit the job."""
    response = feedback_generator(prompt, max_length=200, num_return_sequences=1)
    feedback = response[0]['generated_text'].replace(prompt, '').strip()
    return feedback


def save_jd_to_db(role_title, raw_text):
    session = SessionLocal()
    jd = JobDescription(role_title=role_title, raw_text=raw_text)
    session.add(jd)
    session.commit()
    session.refresh(jd)
    session.close()
    return jd


def get_all_jds():
    session = SessionLocal()
    jds = session.query(JobDescription).order_by(JobDescription.created_at.desc()).all()
    session.close()
    return jds


def save_evaluation_db(data):
    session = SessionLocal()
    try:
        evaluation = Evaluation(
            timestamp=datetime.strptime(data['timestamp'], "%Y-%m-%d %H:%M:%S"),
            job_id=data['job_id'],
            candidate_location=data['candidate_location'],
            filename=data['filename'],
            job_skills=data['job_skills'],
            resume_skills=data['resume_skills'],
            score=data['score'],
            verdict=data['verdict'],
            feedback=data['feedback']
        )
        session.add(evaluation)
        session.commit()
    finally:
        session.close()


def get_evaluations_by_jd(job_id):
    session = SessionLocal()
    evaluations = session.query(Evaluation).filter(Evaluation.job_id == job_id).order_by(Evaluation.score.desc()).all()
    session.close()
    return evaluations


def main():
    st.set_page_config(page_title="Placement Team Dashboard", layout="wide")
    st.title("Placement Team Interface")

    menu = ["Job Descriptions", "Resume Checker", "Shortlisted Resumes"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Job Descriptions":
        st.header("Manage Job Descriptions")

        st.info("""
        **How to use this application:**

        1. Use the **Job Descriptions** tab to upload and manage job descriptions.

        2. Switch to **Resume Checker** tab to upload resumes and evaluate candidates against stored job descriptions.

        3. Go to **Shortlisted Resumes** to filter and view resume evaluations per job description.

        For assistance or issues, please contact your system administrator.
        """)

        jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=['pdf', 'docx'])
        if jd_file:
            if jd_file.type == 'application/pdf':
                jd_text = extract_text_from_pdf_with_header_footer_removal(jd_file)
            elif jd_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                jd_text = extract_text_from_docx(jd_file)
            else:
                st.error("Unsupported JD file type.")
                jd_text = None

            if jd_text:
                jd_info = extract_jd_sections(jd_text)
                st.markdown(f"**Role Title:** {jd_info['role_title']}")
                if st.button("Save Job Description"):
                    jd = save_jd_to_db(jd_info['role_title'], jd_text)
                    st.success(f"Job Description saved with ID: {jd.id}")

        st.subheader("Existing Job Descriptions")
        jds = get_all_jds()
        for jd in jds:
            st.write(f"ID: {jd.id} | Role: {jd.role_title} | Uploaded on: {jd.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    elif choice == "Resume Checker":
        jds = get_all_jds()
        jd_options = {jd.role_title: jd.id for jd in jds}
        if not jd_options:
            st.warning("No job descriptions available. Please upload a JD first.")
            return
        selected_role = st.selectbox("Select Job Description", list(jd_options.keys()))
        selected_jd_id = jd_options[selected_role]

        uploaded_files = st.file_uploader(f"Upload Resumes to evaluate for {selected_role}", type=['pdf', 'docx'], accept_multiple_files=True)
        if uploaded_files and st.button("Evaluate Resumes"):
            progress_bar = st.progress(0)
            total = len(uploaded_files)
            for idx, uploaded_file in enumerate(uploaded_files):
                if uploaded_file.type == 'application/pdf':
                    resume_text = extract_text_from_pdf_with_header_footer_removal(uploaded_file)
                elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    resume_text = extract_text_from_docx(uploaded_file)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
                    continue

                resume_skills = extract_skills_rule_based(resume_text)
                session = SessionLocal()
                jd_record = session.query(JobDescription).filter(JobDescription.id == selected_jd_id).first()
                jd_text = jd_record.raw_text if jd_record else ""
                session.close()

                jd_info = extract_jd_sections(jd_text)
                job_skills = jd_info['must_have_skills'] + jd_info['good_have_skills']

                matched_skills = fuzzy_skill_match(resume_skills, job_skills)
                missing_skills = set(job_skills) - set(matched_skills)

                hard_match_score = len(matched_skills) / max(len(job_skills), 1)
                soft_match_score = compute_semantic_similarity(jd_text, resume_text)
                relevance_score = int(0.4 * hard_match_score * 100 + 0.6 * soft_match_score * 100)

                missing_str = ", ".join(missing_skills) if missing_skills else "None"
                feedback = generate_dynamic_feedback(jd_text, resume_text, missing_str)

                candidate_location = extract_location_from_resume(resume_text)

                if relevance_score > 75:
                    verdict = "High Suitability"
                elif relevance_score > 50:
                    verdict = "Medium Suitability"
                else:
                    verdict = "Low Suitability"

                data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "job_id": selected_jd_id,
                    "candidate_location": candidate_location,
                    "filename": uploaded_file.name,
                    "job_skills": ", ".join(job_skills),
                    "resume_skills": ", ".join(resume_skills),
                    "score": relevance_score,
                    "verdict": verdict,
                    "feedback": feedback
                }
                save_evaluation_db(data)

                st.markdown(f"**{uploaded_file.name}** - Score: {relevance_score} - Verdict: {verdict}")
                if missing_skills:
                    st.warning(f"Missing Skills: {missing_str}")
                st.write(f"Feedback: {feedback}")
                progress_bar.progress((idx + 1) / total)
            progress_bar.empty()

    elif choice == "Shortlisted Resumes":
        jds = get_all_jds()
        jd_options = {jd.role_title: jd.id for jd in jds}
        if not jd_options:
            st.warning("No job descriptions available. Please upload a JD first.")
            return
        selected_role = st.selectbox("Select Job Description to view Shortlist", list(jd_options.keys()))
        selected_jd_id = jd_options[selected_role]

        evaluations = get_evaluations_by_jd(selected_jd_id)
        if not evaluations:
            st.info("No resumes evaluated for this job description yet.")
            return

        data = [{
            "Timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "File": e.filename,
            "Location": e.candidate_location,
            "Score": e.score,
            "Verdict": e.verdict,
            "Feedback": e.feedback
        } for e in evaluations]

        df = pd.DataFrame(data)

        # Filtering options
        st.subheader("Filter Shortlisted Resumes")
        min_score, max_score = st.slider("Score Range", 0, 100, (0, 100))
        verdict_filter = st.multiselect(
            "Filter by Verdict",
            options=df['Verdict'].unique(),
            default=df['Verdict'].unique()
        )
        filtered_df = df[
            (df['Score'] >= min_score) &
            (df['Score'] <= max_score) &
            (df['Verdict'].isin(verdict_filter))
        ]

        st.dataframe(filtered_df)

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Shortlisted Resumes CSV",
            data=csv_data,
            file_name=f'shortlisted_resumes_{selected_role}.csv',
            mime='text/csv'
        )


if __name__ == "__main__":
    main()
