import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from router.auth import auth_router, get_current_user 
from router.tracker import tracker_router
from router.doctor import router as doctor_router
from db import init_db
from fastapi import Depends
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_PATH = BASE_DIR / "frontend.html"

MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
MODEL_PATH = MODELS_DIR / "model.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

NUM_COLS = [
    "Age", "Height(cm)", "Weight(kg)", "BMI", "Cycle Length(days)", "Marriage Status (Yrs)",
    "No. of Abortions", "Exercise (days/week)", "Alcohol (drinks/week)", "Sleep (hours/day)",
    "Stress Level (1-10)", "Water Intake (liters/day)", "Fast Food (meals/week)", "Coffee/Tea (cups/day)"
]

app = FastAPI(title="OVACARE Gateway", version="1.0.0")

app.include_router(auth_router)
app.include_router(tracker_router)
app.include_router(doctor_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
encoders: Optional[Dict[str, Any]] = None
feature_names: Optional[List[str]] = None
scaler = None


def _load(path: Path):
    return joblib.load(path.as_posix())


@app.on_event("startup")
def load_artifacts():
    global model, encoders, feature_names, scaler
    # Ensure DB tables exist
    init_db()
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing model at {MODEL_PATH}")
    model = _load(MODEL_PATH)

    if ENCODERS_PATH.exists():
        enc = _load(ENCODERS_PATH)
        if not isinstance(enc, dict):
            raise RuntimeError("encoders.pkl must be a dict of LabelEncoders keyed by column.")
        encoders = enc
    else:
        encoders = {}

    if FEATURE_NAMES_PATH.exists():
        fn = _load(FEATURE_NAMES_PATH)
        feature_names = list(fn) if fn is not None else None
    else:
        feature_names = None

    if SCALER_PATH.exists():
        scaler = _load(SCALER_PATH)

    media_dir = os.environ.get("MEDIA_DIR", os.path.join(os.getcwd(), "media"))
    os.makedirs(media_dir, exist_ok=True)
    try:
        app.mount("/media", StaticFiles(directory=media_dir), name="media")
    except Exception:\
        pass


def _ensure_bmi(row: Dict[str, Any]) -> Dict[str, Any]:
    if "BMI" not in row and ("Height(cm)" in row and "Weight(kg)" in row):
        try:
            h_cm = float(row["Height(cm)"])
            w_kg = float(row["Weight(kg)"])
            if h_cm > 0:
                row["BMI"] = round(w_kg / ((h_cm / 100.0) ** 2), 1)
        except Exception:
            pass
    return row


def _order_and_fill(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[feature_names]
        used = list(feature_names)
    else:
        used = df.columns.tolist()
    df = df.replace({None: np.nan}).fillna(0)
    return df, used


def _safe_label_transform(le, series: pd.Series) -> pd.Series:
    classes = getattr(le, "classes_", [])
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    return series.astype(str).map(mapping).fillna(-1).astype(int)


def _apply_encoders(df: pd.DataFrame) -> pd.DataFrame:
    if not encoders:
        return df
    df2 = df.copy()
    for col, le in encoders.items():
        if col in df2.columns:
            df2[col] = _safe_label_transform(le, df2[col])
    return df2


def _apply_scaler(df: pd.DataFrame) -> pd.DataFrame:
    if scaler is None:
        return df
    df2 = df.copy()
    present_num = [c for c in NUM_COLS if c in df2.columns]
    if present_num:
        df2[present_num] = scaler.transform(df2[present_num])
    return df2


def _predict_core(df: pd.DataFrame) -> Tuple[List[int], Optional[List[float]]]:
    preds = model.predict(df).tolist()
    probs: Optional[List[float]] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            probs = proba[:, 1].tolist()
        else:
            probs = np.squeeze(proba).tolist()
    return preds, probs


def _coerce_numeric_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for c in NUM_COLS:
        if c in out:
            try:
                out[c] = float(out[c])
            except Exception:
                pass
    return out


def _feature_importance_map(top_k: int = 8) -> Dict[str, float]:
    imp: Dict[str, float] = {}
    if hasattr(model, "feature_importances_") and feature_names:
        vals = np.array(getattr(model, "feature_importances_"))
        names = np.array(feature_names)
        idx = np.argsort(vals)[::-1][:top_k]
        top_vals = vals[idx]
        total = top_vals.sum() or 1.0
        for n, v in zip(names[idx], top_vals):
            imp[str(n)] = round(float(v / total * 100.0), 1)
    return imp

def clean_ai_response(text):
    lines = text.split('\n')
    filtered = []
    for line in lines:
        if not re.search(r'(let me|wait|maybe|i need to|i\'ll proceed|now, considering|now, let me|starting with|next,|for each category|overall,|here’s a breakdown)', line, re.IGNORECASE):
            filtered.append(line)
    return '\n'.join(filtered)

@app.get("/")
def index():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="frontend.html not found")
    return FileResponse(FRONTEND_PATH)


from fastapi import Depends
from router.auth import get_current_user

@app.post("/predict")
async def predict_handler(request: Request, current_user: dict = Depends(get_current_user)):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object with feature fields")

        row = _coerce_numeric_fields(_ensure_bmi(payload))
        df = pd.DataFrame([row])
        df, _ = _order_and_fill(df)
        df = _apply_encoders(df)
        df = _apply_scaler(df)

        preds, probs = _predict_core(df)
        proba = float(probs[0]) if probs else float(preds[0])
        risk_pct = round(proba * 100.0, 1)

        if risk_pct < 30:
            explanation = "Low risk: maintain healthy habits and regular checkups."
        elif risk_pct < 60:
            explanation = "Moderate risk: consider consulting a healthcare professional."
        else:
            explanation = "High risk: please consult a healthcare professional for assessment."

        return JSONResponse({
            "risk": risk_pct,
            "explanation": explanation,
            "featureImportance": _feature_importance_map(top_k=8),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3-32b")
SYSTEM_PROMPT = (
    "You are an expert assistant specialized in PCOS (Polycystic Ovary Syndrome). "
    "Only answer questions related to PCOS, its symptoms, diagnosis, management, "
    "lifestyle, and women's hormonal health. If a user asks a question unrelated to PCOS, "
    "politely respond that you are only able to assist with PCOS-related topics. "
    "Do NOT include chain-of-thought, hidden reasoning, or <think> tags in your reply; "
    "respond with the final answer only."
)


@app.post("/chat")
async def chat_handler(request: Request , current_user: dict = Depends(get_current_user)):
    try:
        data = await request.json()
        message = (data or {}).get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        # Read the API key from environment at request time to avoid stale values
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return JSONResponse({
                "response": (
                    "Chat backend is not configured. Set GROQ_API_KEY to enable responses, "
                    "or replace /chat with your own provider."
                )
            })

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            reply = _strip_think(reply)
            return JSONResponse({"response": reply})
        else:
            return JSONResponse({"response": f"Backend error: {resp.text}"}, status_code=500)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

def _strip_think(text: str) -> str:
    """Remove chain-of-thought blocks such as <think>...</think> and common variants."""
    if not text:
        return text
    text = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", text)
    text = re.sub(r"(?is)```\s*(thought|thinking|reasoning)[\s\S]*?```", "", text)
    text = re.sub(r"(?is)^(?:\s*(thoughts?|reasoning)\s*:\s*.*?)(?:\n\s*\n|$)", "", text)
    return text.strip()


@app.post("/ai-suggest")
async def ai_suggest(request: Request , current_user: dict = Depends(get_current_user)):
    try:
        data = await request.json()
        prompt = data.get("prompt")
        print("Prompt received:", prompt) 
        groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            return JSONResponse(status_code=500, content={"error": "GROQ_API_KEY not configured"})
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(groq_api_url, json=payload, headers=headers, timeout=30)
        print("Groq response status:", response.status_code)  
        print("Groq response body:", response.text)  
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        suggestion = clean_ai_response(_strip_think(raw))
        return {"suggestion": suggestion}
    except Exception as e:
        print("Error in /ai-suggest:", e) 
        return JSONResponse(status_code=500, content={"error": str(e)})