import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from db import get_db, User as DbUser, UserVerification, UserPasswordReset, PendingSignup
import smtplib
from email.message import EmailMessage
import hashlib
import secrets
 
logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "placeholder_secret_key") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  


"""
Password hashing
- Prefer pbkdf2_sha256 for new hashes, but also support verifying legacy bcrypt_sha256/bcrypt hashes.
"""
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256", "bcrypt_sha256", "bcrypt"],
    deprecated="auto",
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login") 

def verify_password(plain_password:str, hashed_password:str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    # Remove unnecessary truncation, hash full password
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
   
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Note: Users are persisted in SQLite via SQLAlchemy (see db.py)


class UserOut(BaseModel):
    username: str
    email: EmailStr

class UserSignup(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, example="patient_pcos", description="username for signup")
    email: EmailStr = Field(..., example="user@example.com", description="Valid email address")
    password: str = Field(..., min_length=8, max_length=128, example="SecurePassword123", description="Strong password (8-128 chars)")

class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, example="patient_pcos", description="Your registered username")
    password: str = Field(..., min_length=8, max_length=128, example="SecurePassword123", description="Your password")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

auth_router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _gen_otp(length: int = 6) -> str:
    # 6-digit numeric code
    return ''.join(secrets.choice('0123456789') for _ in range(length))


def _utcnow() -> datetime:
    """Return naive UTC datetime for consistent storage and comparison."""
    return datetime.utcnow()


def _as_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize any datetime to naive UTC for safe comparisons and math."""
    if dt is None:
        return None
    if dt.tzinfo is not None and dt.utcoffset() is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _send_email(to_email: str, subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    port = int(os.getenv("SMTP_PORT", "587"))
    from_email = os.getenv("SMTP_FROM", user or "no-reply@example.com")

    if not host or not user or not password:
        # Fallback: log only; useful for development without SMTP
        logging.getLogger(__name__).warning("SMTP not configured; email contents below.\nTo: %s\nSubject: %s\n%s", to_email, subject, body)
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
    return True

class MessageResponse(BaseModel):
    message: str


@auth_router.post("/signup", response_model=MessageResponse, status_code=201)
def signup(user: UserSignup, db: Session = Depends(get_db)):
    """Begin signup by creating a pending record and emailing an OTP.
    The actual user row is only created after successful verification.
    """
    # Block if already a registered user
    existing_user = db.query(DbUser).filter((DbUser.username == user.username) | (DbUser.email == user.email)).first()
    if existing_user:
        if existing_user.username == user.username:
            raise HTTPException(status_code=400, detail="Username already registered")
        else:
            raise HTTPException(status_code=400, detail="Email already registered")

    # Block if another pending signup is using same username or email (avoid confusion)
    pending = db.query(PendingSignup).filter((PendingSignup.username == user.username) | (PendingSignup.email == user.email)).first()
    if pending and (pending.username != user.username or pending.email != user.email):
        # Someone else is attempting with either same username or same email
        if pending.username == user.username:
            raise HTTPException(status_code=400, detail="Username already in use (pending verification)")
        if pending.email == user.email:
            raise HTTPException(status_code=400, detail="Email already in use (pending verification)")

    # Create or update pending signup
    hashed_password = get_password_hash(user.password)
    code = _gen_otp()
    now = _utcnow()
    if not pending:
        pending = PendingSignup(
            username=user.username,
            email=user.email,
            password_hash=hashed_password,
            code_hash=_hash_code(code),
            expires_at=now + timedelta(minutes=10),
            last_sent_at=now,
        )
        db.add(pending)
    else:
        # Refresh password and code
        pending.password_hash = hashed_password
        pending.code_hash = _hash_code(code)
        pending.expires_at = now + timedelta(minutes=10)
        pending.last_sent_at = now
        db.add(pending)
    db.commit()

    subject = "OVACARE: Verify your email"
    body = (
        f"Hello {user.username},\n\n"
        f"Your verification code is: {code}\nThis code will expire in 10 minutes.\n\n"
        f"If you did not sign up, please ignore this email."
    )
    sent = _send_email(user.email, subject, body)
    if not sent:
        logger.warning("SMTP not configured; sent OTP via logs for %s: %s", user.email, code)

    logger.info(f"✅ Pending signup created for {user.username} ({user.email}); verification code issued")
    return MessageResponse(message="Verification code sent. Check your email to complete signup.")



@auth_router.post("/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    username = form_data.username
    password = form_data.password

    user = db.query(DbUser).filter(DbUser.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # Enforce email verification only for legacy accounts that used UserVerification.
    # For the new flow, users are created only after verification, so no record exists -> allow login.
    ver = db.query(UserVerification).filter(UserVerification.user_id == user.id).first()
    if ver and not ver.verified:
        raise HTTPException(status_code=403, detail="Email not verified. Please check your email for the verification code.")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    logger.info(f"🔓 User logged in: {username}")
    return {"access_token": access_token, "token_type": "bearer"}


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = db.query(DbUser).filter(DbUser.username == token_data.username).first()
    if not user:
        raise credentials_exception
    return {"username": user.username, "email": user.email}


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=8, max_length=128, description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")


@auth_router.post("/change-password")
def change_password(payload: ChangePasswordRequest, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    """Change the authenticated user's password.
    Validates current password, requires new password to differ, then updates the hash.
    """
    user = db.query(DbUser).filter(DbUser.username == current_user["username"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(payload.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect current password")

    if payload.old_password == payload.new_password:
        raise HTTPException(status_code=400, detail="New password must be different from current password")

    # Optional: add additional complexity checks here if desired

    user.password_hash = get_password_hash(payload.new_password)
    db.add(user)
    db.commit()
    return {"message": "Password updated successfully"}


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=4, max_length=10)


@auth_router.post("/verify-email")
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    """Verify email for either pending signups (new flow) or legacy unverified users (old flow)."""
    now = _utcnow()
    # New flow: pending signup exists
    pending = db.query(PendingSignup).filter(PendingSignup.email == payload.email).first()
    if pending:
        exp = _as_naive_utc(pending.expires_at)
        if exp and now > exp:
            raise HTTPException(status_code=400, detail="Verification code expired. Please resend a new code.")
        if _hash_code(payload.code) != pending.code_hash:
            raise HTTPException(status_code=400, detail="Invalid code")

        # Ensure no user was created concurrently
        conflict = db.query(DbUser).filter((DbUser.username == pending.username) | (DbUser.email == pending.email)).first()
        if conflict:
            raise HTTPException(status_code=400, detail="Account already exists. Try signing in.")

        # Create user and clear pending
        user = DbUser(username=pending.username, email=pending.email, password_hash=pending.password_hash)
        db.add(user)
        db.commit()

        # Remove pending record
        db.delete(pending)
        db.commit()
        return {"message": "Email verified successfully"}

    # Legacy flow fallback
    user = db.query(DbUser).filter(DbUser.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    ver = db.query(UserVerification).filter(UserVerification.user_id == user.id).first()
    if not ver:
        raise HTTPException(status_code=400, detail="No verification request found. Please sign up again.")
    if ver.verified:
        return {"message": "Email already verified"}
    if ver.expires_at:
        exp = _as_naive_utc(ver.expires_at)
        if now > exp:
            raise HTTPException(status_code=400, detail="Verification code expired. Please resend a new code.")
    if _hash_code(payload.code) != ver.code_hash:
        raise HTTPException(status_code=400, detail="Invalid code")

    ver.verified = True
    ver.code_hash = None
    ver.expires_at = None
    db.add(ver)
    db.commit()
    return {"message": "Email verified successfully"}


class ResendCodeRequest(BaseModel):
    email: EmailStr


@auth_router.post("/resend-code")
def resend_code(payload: ResendCodeRequest, db: Session = Depends(get_db)):
    now = _utcnow()
    # First try new flow (pending signup)
    pending = db.query(PendingSignup).filter(PendingSignup.email == payload.email).first()
    if pending:
        last = _as_naive_utc(pending.last_sent_at)
        if last and (now - last).total_seconds() < 60:
            raise HTTPException(status_code=429, detail="Please wait before requesting a new code")
        code = _gen_otp()
        pending.code_hash = _hash_code(code)
        pending.expires_at = now + timedelta(minutes=10)
        pending.last_sent_at = now
        db.add(pending)
        db.commit()

        subject = "OVACARE: Your verification code"
        body = f"Your verification code is: {code}\nThis code will expire in 10 minutes."
        sent = _send_email(pending.email, subject, body)
        if not sent:
            logger.warning("SMTP not configured; resent OTP via logs for %s: %s", pending.email, code)
        return {"message": "Verification code sent"}

    # Fallback to legacy flow
    user = db.query(DbUser).filter(DbUser.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    ver = db.query(UserVerification).filter(UserVerification.user_id == user.id).first()
    if not ver:
        ver = UserVerification(user_id=user.id, verified=False)

    last = _as_naive_utc(ver.last_sent_at)
    if last and (now - last).total_seconds() < 60:
        raise HTTPException(status_code=429, detail="Please wait before requesting a new code")

    code = _gen_otp()
    ver.code_hash = _hash_code(code)
    ver.expires_at = now + timedelta(minutes=10)
    ver.last_sent_at = now
    db.add(ver)
    db.commit()

    subject = "OVACARE: Your verification code"
    body = f"Your verification code is: {code}\nThis code will expire in 10 minutes."
    sent = _send_email(user.email, subject, body)
    if not sent:
        logger.warning("SMTP not configured; resent OTP via logs for %s: %s", user.email, code)

    return {"message": "Verification code sent"}


# Password reset flow
class ResetRequest(BaseModel):
    email: EmailStr


class ResetConfirm(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=4, max_length=10)
    new_password: str = Field(..., min_length=8, max_length=128)


@auth_router.post("/reset-request")
def reset_request(payload: ResetRequest, db: Session = Depends(get_db)):
    user = db.query(DbUser).filter(DbUser.email == payload.email).first()
    # Respond 200 regardless to avoid account enumeration
    if not user:
        return {"message": "If an account exists, a reset code has been sent"}

    # Find existing non-used request
    reset = (
        db.query(UserPasswordReset)
        .filter(UserPasswordReset.user_id == user.id, UserPasswordReset.used == False)
        .order_by(UserPasswordReset.id.desc())
        .first()
    )

    now = _utcnow()
    # Simple rate limit: 60s between sends per user
    last_sent = _as_naive_utc(reset.last_sent_at) if reset else None
    if reset and last_sent and (now - last_sent).total_seconds() < 60:
        raise HTTPException(status_code=429, detail="Please wait before requesting another reset code")

    code = _gen_otp()
    if not reset:
        reset = UserPasswordReset(user_id=user.id)
    reset.used = False
    reset.code_hash = _hash_code(code)
    reset.expires_at = now + timedelta(minutes=10)
    reset.last_sent_at = now
    db.add(reset)
    db.commit()

    subject = "OVACARE: Password reset code"
    body = (
        f"Hello {user.username},\n\n"
        f"Your password reset code is: {code}\n"
        f"This code will expire in 10 minutes. If you did not request this, you can ignore this email."
    )
    sent = _send_email(user.email, subject, body)
    if not sent:
        logger.warning("SMTP not configured; reset OTP via logs for %s: %s", user.email, code)

    return {"message": "If an account exists, a reset code has been sent"}


@auth_router.post("/reset-confirm")
def reset_confirm(payload: ResetConfirm, db: Session = Depends(get_db)):
    user = db.query(DbUser).filter(DbUser.email == payload.email).first()
    if not user:
        # Avoid enumeration: generic error
        raise HTTPException(status_code=400, detail="Invalid code or request")

    reset = (
        db.query(UserPasswordReset)
        .filter(UserPasswordReset.user_id == user.id, UserPasswordReset.used == False)
        .order_by(UserPasswordReset.id.desc())
        .first()
    )
    if not reset:
        raise HTTPException(status_code=400, detail="Invalid code or request")

    now = _utcnow()
    exp = _as_naive_utc(reset.expires_at)
    if exp and now > exp:
        raise HTTPException(status_code=400, detail="Reset code expired. Please request a new code.")

    if _hash_code(payload.code) != reset.code_hash:
        raise HTTPException(status_code=400, detail="Invalid code or request")

    # Update password
    user.password_hash = get_password_hash(payload.new_password)
    db.add(user)
    # Mark reset used
    reset.used = True
    reset.code_hash = None
    reset.expires_at = None
    db.add(reset)
    db.commit()

    return {"message": "Password has been reset successfully"}