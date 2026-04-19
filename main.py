from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import models
from database import engine, get_db

# ── Create tables ─────────────────────────────────────────
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SpendWise AI - FastAPI Backend",
    description="Backend API for SpendWise AI mobile and web app",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── JWT Config ────────────────────────────────────────────
SECRET_KEY = "spendwise-secret-key-2026"
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 1 day

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ── Pydantic Schemas ──────────────────────────────────────
class RegisterRequest(BaseModel):
    username:     str
    email:        str
    password:     str
    first_name:   Optional[str] = ""
    income_type:  Optional[str] = "other"
    income_cycle: Optional[str] = "monthly"

class ProfileUpdate(BaseModel):
    income_amount: Optional[float] = None
    savings_goal:  Optional[float] = None
    income_type:   Optional[str]   = None 
    income_cycle:  Optional[str]   = None


class ExpenseCreate(BaseModel):
    amount:       float
    category_key: str
    description:  Optional[str] = ""
    timestamp:    str


class ExpenseUpdate(BaseModel):
    amount:      Optional[float] = None
    description: Optional[str]  = None


class SavingsGoalCreate(BaseModel):
    name:           str
    target_amount:  float
    current_amount: Optional[float] = 0
    deadline:       Optional[str]   = None

class SavingsGoalUpdate(BaseModel):
    current_amount: Optional[float] = None
    name:           Optional[str]   = None