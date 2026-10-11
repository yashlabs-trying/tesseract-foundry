from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean, JSON, Enum as SAEnum
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool
from ..config import settings
import enum

Base = declarative_base()

class JobStatus(enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"

class JobRecord(Base):
    __tablename__ = "jobs"
    job_id = Column(String, primary_key=True)
    job_type = Column(String, nullable=False)
    status = Column(SAEnum(JobStatus), default=JobStatus.pending)
    params = Column(JSON, default=dict)
    result_ref = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

class RecommendationRecord(Base):
    __tablename__ = "recommendations"
    recommendation_id = Column(String, primary_key=True)
    recommended_candidate_id = Column(String)
    feasible_candidates = Column(JSON, default=list)
    rejected_candidates = Column(JSON, default=list)
    approval_status = Column(String, default="pending_review")
    report_data = Column(JSON, nullable=True)
    created_at = Column(DateTime)

def get_engine():
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(
        settings.database_url,
        connect_args=connect_args,
        poolclass=StaticPool if settings.database_url.startswith("sqlite") else None,
    )

def get_session():
    engine = get_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
