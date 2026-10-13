from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean, JSON, Enum as SAEnum
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator
from ..config import settings
import enum
import json
from datetime import date, datetime
from decimal import Decimal

Base = declarative_base()


def _json_default(obj):
    """JSON encoder for values that are not natively serializable.

    ``Decimal`` is emitted as a string to preserve exact precision for financial
    values (never silently converted to float). Datetimes/enums are also handled.
    """
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, enum.Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class SafeJSON(TypeDecorator):
    """A JSON column that tolerates Decimal/datetime/enum values on write.

    The stored value is normalized to plain JSON types (Decimal -> string) so
    SQLAlchemy's JSON serializer never raises, while precision is preserved.
    """

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.loads(json.dumps(value, default=_json_default))

    def process_result_value(self, value, dialect):
        return value



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
    params = Column(SafeJSON, default=dict)
    result_ref = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

class RecommendationRecord(Base):
    __tablename__ = "recommendations"
    recommendation_id = Column(String, primary_key=True)
    recommended_candidate_id = Column(String)
    feasible_candidates = Column(SafeJSON, default=list)
    rejected_candidates = Column(SafeJSON, default=list)
    approval_status = Column(String, default="pending_review")
    report_data = Column(SafeJSON, nullable=True)
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
