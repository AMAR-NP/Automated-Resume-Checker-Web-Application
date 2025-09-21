from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

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

def init_db():
    Base.metadata.create_all(bind=engine)
