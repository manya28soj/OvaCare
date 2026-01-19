from datetime import datetime
from typing import Generator

from sqlalchemy import Column, Integer, String, Date, DateTime, Boolean, ForeignKey, Float, Text, create_engine, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("username", name="uq_users_username"),
        UniqueConstraint("email", name="uq_users_email"),
    )

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class UserVerification(Base):
    __tablename__ = "user_verification"

    user_id = Column(Integer, primary_key=True, index=True)
    verified = Column(Boolean, default=False, nullable=False)
    code_hash = Column(String(255), nullable=True)
    expires_at = Column(DateTime, nullable=True)
    last_sent_at = Column(DateTime, nullable=True)


class UserPasswordReset(Base):
    __tablename__ = "user_password_reset"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    code_hash = Column(String(255), nullable=True)
    expires_at = Column(DateTime, nullable=True)
    last_sent_at = Column(DateTime, nullable=True)


class PendingSignup(Base):
    __tablename__ = "pending_signup"
    __table_args__ = (
        UniqueConstraint("username", name="uq_pending_username"),
        UniqueConstraint("email", name="uq_pending_email"),
    )

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    code_hash = Column(String(255), nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class FitnessLog(Base):
    __tablename__ = "fitness_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "day", name="uq_fitness_user_day"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    day = Column(Date, nullable=False, index=True)

    # General wellbeing
    sleep_hours = Column(Float, nullable=True)
    water_liters = Column(Float, nullable=True)
    exercise_minutes = Column(Integer, nullable=True)
    steps = Column(Integer, nullable=True)
    calories_in = Column(Integer, nullable=True)
    calories_out = Column(Integer, nullable=True)
    weight_kg = Column(Float, nullable=True)
    stress_level = Column(Integer, nullable=True) 
    mood = Column(String(32), nullable=True)  

    # Period/PCOS related
    cramps = Column(Boolean, default=False, nullable=False)
    flow_level = Column(String(16), nullable=True) 
    pain_level = Column(Integer, nullable=True)  
    acne = Column(Boolean, default=False, nullable=False)
    bloating = Column(Boolean, default=False, nullable=False)
    headaches = Column(Boolean, default=False, nullable=False)
    cravings = Column(Boolean, default=False, nullable=False)
    fatigue = Column(Boolean, default=False, nullable=False)

    # Free-form
    notes = Column(Text, nullable=True)
    medications_json = Column(Text, nullable=True)  
    symptoms_json = Column(Text, nullable=True)     

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    city = Column(String(80), nullable=False, index=True)
    specialty = Column(String(120), nullable=False, default="Gynecologist")
    whatsapp_number = Column(String(32), nullable=True)  # E.164 format
    email = Column(String(160), nullable=True)


class ConsultRequest(Base):
    __tablename__ = "consult_requests"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True, index=True)
    patient_name = Column(String(120), nullable=False)
    patient_age = Column(Integer, nullable=True)
    patient_email = Column(String(160), nullable=True)
    patient_phone = Column(String(40), nullable=True)
    city = Column(String(80), nullable=False)
    inputs_json = Column(Text, nullable=False)
    prediction_json = Column(Text, nullable=False)
    pdf_path = Column(String(255), nullable=False)
    status = Column(String(40), nullable=False, default="queued")  # queued|sent|failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    doctor = relationship("Doctor")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
