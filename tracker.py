import json
from datetime import date, datetime, timedelta
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db import get_db, FitnessLog, User as DbUser
from router.auth import get_current_user

tracker_router = APIRouter(prefix="/tracker", tags=["Tracker"])


class FitnessLogIn(BaseModel):
    day: date = Field(..., description="Log date (YYYY-MM-DD)")
    sleep_hours: Optional[float] = None
    water_liters: Optional[float] = None
    exercise_minutes: Optional[int] = None
    steps: Optional[int] = None
    calories_in: Optional[int] = None
    calories_out: Optional[int] = None
    weight_kg: Optional[float] = None
    stress_level: Optional[int] = Field(None, ge=1, le=10)
    mood: Optional[str] = None

    cramps: Optional[bool] = None
    flow_level: Optional[str] = Field(None, description="none|light|moderate|heavy")
    pain_level: Optional[int] = Field(None, ge=0, le=10)
    acne: Optional[bool] = None
    bloating: Optional[bool] = None
    headaches: Optional[bool] = None
    cravings: Optional[bool] = None
    fatigue: Optional[bool] = None
    notes: Optional[str] = None
    medications: Optional[List[str]] = None
    symptoms: Optional[List[str]] = None


class FitnessLogOut(BaseModel):
    id: int
    day: date
    sleep_hours: Optional[float]
    water_liters: Optional[float]
    exercise_minutes: Optional[int]
    steps: Optional[int]
    calories_in: Optional[int]
    calories_out: Optional[int]
    weight_kg: Optional[float]
    stress_level: Optional[int]
    mood: Optional[str]
    cramps: bool
    flow_level: Optional[str]
    pain_level: Optional[int]
    acne: bool
    bloating: bool
    headaches: bool
    cravings: bool
    fatigue: bool
    notes: Optional[str]
    medications: List[str] = []
    symptoms: List[str] = []

    class Config:
        from_attributes = True


class FitnessStats(BaseModel):
    start: date
    end: date
    days: int
    avg_sleep_hours: Optional[float]
    avg_water_liters: Optional[float]
    avg_exercise_minutes: Optional[float]
    avg_pain_level: Optional[float]
    avg_stress_level: Optional[float]
    days_with_cramps: int
    days_with_acne: int
    days_with_bloating: int



def _db_user(db: Session, username: str) -> DbUser:
    user = db.query(DbUser).filter(DbUser.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid user")
    return user


def _to_out(log: FitnessLog) -> FitnessLogOut:
    meds = []
    sym = []
    try:
        if log.medications_json:
            meds = json.loads(log.medications_json)
    except Exception:
        meds = []
    try:
        if log.symptoms_json:
            sym = json.loads(log.symptoms_json)
    except Exception:
        sym = []
    return FitnessLogOut(
        id=log.id,
        day=log.day,
        sleep_hours=log.sleep_hours,
        water_liters=log.water_liters,
        exercise_minutes=log.exercise_minutes,
        steps=log.steps,
        calories_in=log.calories_in,
        calories_out=log.calories_out,
        weight_kg=log.weight_kg,
        stress_level=log.stress_level,
        mood=log.mood,
        cramps=bool(log.cramps),
        flow_level=log.flow_level,
        pain_level=log.pain_level,
        acne=bool(log.acne),
        bloating=bool(log.bloating),
        headaches=bool(log.headaches),
        cravings=bool(log.cravings),
        fatigue=bool(log.fatigue),
        notes=log.notes,
        medications=meds,
        symptoms=sym,
    )


@tracker_router.post("/logs", response_model=FitnessLogOut)
def upsert_log(payload: FitnessLogIn, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    user = _db_user(db, current_user["username"])  # fetch ORM user
    # Find existing
    log = (
        db.query(FitnessLog)
        .filter(FitnessLog.user_id == user.id, FitnessLog.day == payload.day)
        .first()
    )
    creating = False
    if not log:
        log = FitnessLog(user_id=user.id, day=payload.day, cramps=False, acne=False, bloating=False, headaches=False, cravings=False, fatigue=False)
        creating = True

    # Update fields if provided
    for field in (
        "sleep_hours","water_liters","exercise_minutes","steps","calories_in","calories_out","weight_kg",
        "stress_level","mood","cramps","flow_level","pain_level","acne","bloating","headaches","cravings","fatigue","notes"
    ):
        val = getattr(payload, field)
        if val is not None:
            setattr(log, field, val)

    if payload.medications is not None:
        log.medications_json = json.dumps(payload.medications)
    if payload.symptoms is not None:
        log.symptoms_json = json.dumps(payload.symptoms)

    log.updated_at = datetime.utcnow()

    db.add(log)
    db.commit()
    db.refresh(log)
    return _to_out(log)


@tracker_router.get("/logs", response_model=List[FitnessLogOut])
def list_logs(
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _db_user(db, current_user["username"])  # fetch ORM user
    if end is None:
        end = date.today()
    if start is None:
        start = end - timedelta(days=30)

    logs = (
        db.query(FitnessLog)
        .filter(
            FitnessLog.user_id == user.id,
            FitnessLog.day >= start,
            FitnessLog.day <= end,
        )
        .order_by(FitnessLog.day.desc())
        .all()
    )
    return [_to_out(l) for l in logs]


@tracker_router.delete("/logs/{log_id}")
def delete_log(log_id: int, current_user: Dict[str, Any] = Depends(get_current_user), db: Session = Depends(get_db)):
    user = _db_user(db, current_user["username"])  # fetch ORM user
    log = db.query(FitnessLog).filter(FitnessLog.id == log_id, FitnessLog.user_id == user.id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    db.delete(log)
    db.commit()
    return {"message": "Deleted"}


@tracker_router.get("/stats", response_model=FitnessStats)
def stats(
    window_days: int = Query(30, ge=1, le=365),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _db_user(db, current_user["username"])  # fetch ORM user
    end = date.today()
    start = end - timedelta(days=window_days)

    logs = (
        db.query(FitnessLog)
        .filter(
            FitnessLog.user_id == user.id,
            FitnessLog.day >= start,
            FitnessLog.day <= end,
        )
        .all()
    )
    def _avg(values: List[Optional[float]]) -> Optional[float]:
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 2)

    return FitnessStats(
        start=start,
        end=end,
        days=len(logs),
        avg_sleep_hours=_avg([l.sleep_hours for l in logs]),
        avg_water_liters=_avg([l.water_liters for l in logs]),
        avg_exercise_minutes=_avg([float(l.exercise_minutes) if l.exercise_minutes is not None else None for l in logs]),
        avg_pain_level=_avg([float(l.pain_level) if l.pain_level is not None else None for l in logs]),
        avg_stress_level=_avg([float(l.stress_level) if l.stress_level is not None else None for l in logs]),
        days_with_cramps=sum(1 for l in logs if l.cramps),
        days_with_acne=sum(1 for l in logs if l.acne),
        days_with_bloating=sum(1 for l in logs if l.bloating),
    )
