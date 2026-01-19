import os
import json
import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks
from fastapi import UploadFile
from fastapi import status

from db import get_db, Doctor, ConsultRequest
from router.auth import get_current_user

# Optional dependencies
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
except Exception:
    A4 = None
import smtplib
from email.message import EmailMessage

MEDIA_DIR = os.environ.get("MEDIA_DIR", os.path.join(os.getcwd(), "media"))
CONSULT_DIR = os.path.join(MEDIA_DIR, "consults")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@example.com")

router = APIRouter(prefix="/doctor", tags=["doctor"])


def _ensure_dirs():
    os.makedirs(CONSULT_DIR, exist_ok=True)


def _slug(text: Optional[str]) -> str:
    if not text:
        return "user"
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "user"


def _generate_pdf(patient: dict, doctor: Optional[dict], inputs: dict, prediction: dict, filename_hint: Optional[str] = None) -> str:
    """Create a simple PDF on disk and return absolute path."""
    _ensure_dirs()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    hint = filename_hint or patient.get("name") or "user"
    filename = f"consult_{_slug(hint)}_{ts}.pdf"
    fpath = os.path.join(CONSULT_DIR, filename)

    if A4 is None:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("OVACARE – Consultation Summary\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
            f.write("Patient\n")
            for k in ("name","age","email","phone","city"):
                f.write(f" - {k}: {patient.get(k,'')}\n")
            if doctor:
                f.write("\nDoctor (Recommended)\n")
                f.write(f" - {doctor.get('name','')} – {doctor.get('specialty','')} ({doctor.get('city','')})\n")
            f.write("\nPrediction\n")
            f.write(f" - risk: {prediction.get('risk')}\n")
            if 'explanation' in prediction:
                f.write(f" - explanation: {str(prediction.get('explanation'))[:600]}\n")
            f.write("\nInputs\n")
            for k, v in inputs.items():
                f.write(f" - {k}: {v}\n")
        return fpath

    width, height = A4
    c = canvas.Canvas(fpath, pagesize=A4)
    x, y = 20*mm, height - 20*mm

    def line(txt: str, dy: float = 7*mm, bold: bool = False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if bold else 10)
        c.drawString(x, y, txt)
        y -= dy

    line("OVACARE – Consultation Summary", bold=True)
    line(f"Generated: {datetime.utcnow().isoformat()}Z")
    line("")

    line("Patient", bold=True)
    line(f"Name: {patient.get('name','')}")
    line(f"Age: {patient.get('age','')}")
    line(f"Email: {patient.get('email','')}")
    line(f"Phone: {patient.get('phone','')}")
    line(f"City: {patient.get('city','')}")
    line("")

    if doctor:
        line("Doctor (Recommended)", bold=True)
        line(f"Name: {doctor.get('name','')}  •  City: {doctor.get('city','')}  •  Specialty: {doctor.get('specialty','')}")
        if doctor.get('whatsapp_number'):
            line(f"WhatsApp: {doctor.get('whatsapp_number')}")
        line("")

    line("Prediction", bold=True)
    line(f"Risk: {prediction.get('risk')}")
    if 'explanation' in prediction:
        line(f"Explanation: {str(prediction.get('explanation'))[:560]}")
    line("")

    line("Submitted Inputs", bold=True)
    for k, v in list(inputs.items())[:36]:
        line(f"- {k}: {v}")

    c.showPage()
    c.save()
    return fpath

def _send_email(to_emails, subject: str, body: str, attachment_path: Optional[str] = None) -> dict:
    if not (SMTP_HOST and (SMTP_USER or SMTP_FROM)):
        return {"ok": False, "reason": "SMTP not configured"}
    try:
        if isinstance(to_emails, str):
            recipients = [to_emails]
        else:
            recipients = [e for e in (to_emails or []) if e]
        if not recipients:
            return {"ok": False, "reason": "No recipients"}
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = recipients[0]
        if len(recipients) > 1:
            msg["Bcc"] = ", ".join(recipients[1:])
        msg.set_content(body)
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                data = f.read()
            msg.add_attachment(data, maintype="application", subtype="pdf", filename=os.path.basename(attachment_path))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            if SMTP_USER and SMTP_PASS:
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def _is_placeholder_email(email: Optional[str]) -> bool:
    if not email or "@" not in email:
        return True
    domain = email.split("@")[-1].lower()
    return domain in {"example.com", "example.org", "test.com", "test.local"}


@router.get("/cities")
def get_cities(db: Session = Depends(get_db)):
    if db.query(Doctor).count() == 0:
        db.add_all([
            Doctor(name="Dr. Lakshay Jain", city="Delhi", specialty="Gynecologist", email="lakshayj539@example.com"),
            Doctor(name="Dr. Neha Kapoor", city="Mumbai", specialty="Endocrinologist", email="neha.kapoor@example.com"),
            Doctor(name="Dr. Priya Rao", city="Bengaluru", specialty="Gynecologist", email="priya.rao@example.com"),
            Doctor(name="Dr. Sana Khan", city="Hyderabad", specialty="Reproductive Medicine", email="sana.khan@example.com"),
        ])
        db.commit()
    rows = db.query(Doctor.city).distinct().all()
    return {"cities": sorted({r[0] for r in rows})}


@router.get("/recommend")
def recommend(city: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    docs = db.query(Doctor).filter(Doctor.city == city).limit(5).all()
    return {"doctors": [
        {"id": d.id, "name": d.name, "city": d.city, "specialty": d.specialty}
        for d in docs
    ]}


@router.post("/set-email")
async def set_doctor_email(req: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Admin-lite helper: set or update a doctor's email by id or by unique name/city.
    Payload examples:
      {"doctor_id": 1, "email": "name@example.com"}
      {"name": "Dr. Lakshay Jain", "city": "Delhi", "email": "lakshayj539@gmail.com"}
    """
    payload = await req.json()
    email = (payload.get("email") or "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email is required")

    doctor = None
    if payload.get("doctor_id"):
        doctor = db.query(Doctor).get(int(payload["doctor_id"]))
        if not doctor:
            raise HTTPException(status_code=404, detail="Doctor not found")
    else:
        name = (payload.get("name") or "").strip()
        city = (payload.get("city") or "").strip()
        q = db.query(Doctor).filter(Doctor.name == name)
        if city:
            q = q.filter(Doctor.city == city)
        matches = q.all()
        if not matches:
            raise HTTPException(status_code=404, detail="Doctor not found by name/city")
        if len(matches) > 1:
            raise HTTPException(status_code=400, detail="Multiple matches; specify doctor_id or include city")
        doctor = matches[0]

    doctor.email = email
    db.add(doctor)
    db.commit()
    return {"id": doctor.id, "name": doctor.name, "city": doctor.city, "email": doctor.email}

@router.post("/consult")
async def consult(req: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    payload = await req.json()
    patient = payload.get("patient") or {}
    inputs = payload.get("inputs") or {}
    prediction = payload.get("prediction") or {}
    doctor_id = payload.get("doctor_id")

    if not (patient.get("name") and patient.get("city") and inputs and prediction):
        raise HTTPException(status_code=400, detail="Missing required fields")

    doctor = db.query(Doctor).get(doctor_id) if doctor_id else None

    uname = None
    try:
        uname = getattr(user, 'username', None)
    except Exception:
        try:
            uname = (user or {}).get('username')
        except Exception:
            uname = None
    name_hint = patient.get('name') or uname
    pdf_path = _generate_pdf(patient, vars(doctor) if doctor else {}, inputs, prediction, filename_hint=name_hint)

    cr = ConsultRequest(
        user_id=getattr(user, 'id', None) if user else None,
        doctor_id=doctor.id if doctor else None,
        patient_name=patient.get("name"),
        patient_age=patient.get("age"),
        patient_email=patient.get("email"),
        patient_phone=patient.get("phone"),
        city=patient.get("city"),
        inputs_json=json.dumps(inputs),
        prediction_json=json.dumps(prediction),
        pdf_path=pdf_path,
        status="queued",
    )
    db.add(cr)
    db.commit()
    db.refresh(cr)

    send_result = None
    sent_via = "email"
    recipients = []
    if doctor and getattr(doctor, 'email', None) and not _is_placeholder_email(doctor.email):
        recipients.append(doctor.email)
    if patient.get('email') and not _is_placeholder_email(patient['email']):
        if patient['email'] not in recipients:
            recipients.append(patient['email'])

    if recipients:
        subject = f"OVACARE consult: {patient.get('name')} ({patient.get('city')})"
        pdf_filename = os.path.basename(pdf_path)
        body = (
            f"Hello! You have a consult request from {patient.get('name')}\n\n"
            f"Consult Summary PDF is attached\n"
            f"{pdf_filename}"
        )
        send_result = _send_email(recipients, subject, body, attachment_path=pdf_path)
        cr.status = "sent" if send_result.get("ok") else "failed"
        db.add(cr)
        db.commit()
    else:
        cr.status = "queued"  
        db.add(cr)
        db.commit()

    return JSONResponse({
        "id": cr.id,
        "status": cr.status,
        "sentVia": sent_via,
    "recipients": recipients,
        "pdf": os.path.basename(pdf_path),
        "doctor": {"id": doctor.id, "name": doctor.name} if doctor else None,
        "sendResult": send_result,
    }, status_code=status.HTTP_201_CREATED)
