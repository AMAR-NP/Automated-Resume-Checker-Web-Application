# Automated-Resume-Checker-Web-Application
Placement Team Resume Evaluation Dashboard
This is a Streamlit web application designed for placement teams to manage job descriptions (JDs), evaluate candidate resumes against these JDs using AI-powered relevance scoring, and view shortlisted candidates efficiently.

Features
Job Description Management: Upload, view, and manage multiple job descriptions.

Resume Relevance Checker: Upload candidate resumes and evaluate their relevance to a selected job description. Uses hybrid scoring combining fuzzy keyword matching and semantic similarity.

Shortlisted Resumes Dashboard: View, filter, and export evaluated candidate resumes per job description, with detailed feedback and skill-gap analysis.

Persistent Storage: Uses SQLite with SQLAlchemy ORM for reliable data storage of JDs and evaluations.

AI-Powered Feedback: Generates personalized improvement suggestions using a local GPT-2 language model.

Candidate Location Extraction: Automatically attempts to extract candidate location from resumes for better filtering.

Installation
Clone the repository or download the source code.

Install the required Python packages:

bash
pip install streamlit sqlalchemy spacy pandas transformers sentence-transformers rapidfuzz pdfplumber python-docx torch
python -m spacy download en_core_web_sm
Running the Application
Execute the following command in your terminal:

bash
streamlit run app.py
This will launch the application accessible in your web browser at http://localhost:8501.

Usage
Job Descriptions Tab: Upload new job descriptions (PDF/DOCX), parse, and store them. View existing ones with metadata.

Resume Checker Tab: Select a job description, upload multiple resumes, and run evaluations. View scores, missing skills, and personalized feedback.

Shortlisted Resumes Tab: Select a job description to view evaluated candidates. Use filters by score and verdict. Download evaluation data as CSV.

Project Structure
app.py: Main Streamlit application integrating UI and backend logic.

database.py: SQLAlchemy models and database initialization.

requirements.txt: (optional) List of dependencies for easy installation.

Customization
Enhance resume parsing and location extraction based on your candidate resume formats.

Replace or upgrade LLM feedback module with more powerful models or external APIs as needed.

Add user authentication for multi-user access control.

Deploy on a cloud platform for broader accessibility.

Troubleshooting
Ensure all dependencies are installed correctly.

SpaCy model must be downloaded (en_core_web_sm).

The SQLite database evaluations.db is created automatically in the app folder.

Verify Python version compatibility (recommend Python 3.8+).
