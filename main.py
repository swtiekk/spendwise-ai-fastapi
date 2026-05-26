from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, date
from collections import defaultdict
from jose import JWTError, jwt
from passlib.context import CryptContext
import models
from database import engine, get_db
from ml.predict import predict_cluster, predict_risk, predict_sustainability, predict_smart_purchase

# ── Constants ─────────────────────────────────────────────
RISK_NORMALIZE = {
    "safe":    "low",
    "caution": "medium",
    "risky":   "high",
    "low":     "low",
    "medium":  "medium",
    "high":    "high",
}

# ── Create tables ─────────────────────────────────────────
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SpendWise AI - FastAPI Backend",
    description="Backend API for SpendWise AI mobile and web app",
    version="1.0.0",
    redirect_slashes=False,
)

# ── CORS ──────────────────────────────────────────────────
# ✅ CORRECT
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://spendwise-ai-admin-indol.vercel.app",
        "https://spendwise-ai-fastapi-zic2.onrender.com",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── JWT Config ────────────────────────────────────────────
SECRET_KEY = "spendwise-secret-key-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 1 day
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ── Pydantic Schemas ──────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    first_name: Optional[str] = ""
    income_type: Optional[str] = "other"
    income_cycle: Optional[str] = "monthly"

class LoginRequest(BaseModel):
    username: str
    password: str

class ExpenseCreate(BaseModel):
    amount: float
    category_key: str
    description: Optional[str] = ""
    timestamp: str

class ExpenseUpdate(BaseModel):
    amount: Optional[float] = None
    description: Optional[str] = None

class SavingsGoalCreate(BaseModel):
    name: str
    target_amount: float
    current_amount: Optional[float] = 0
    deadline: Optional[str] = None

class SavingsGoalUpdate(BaseModel):
    current_amount: Optional[float] = None
    name: Optional[str] = None

class ProfileUpdate(BaseModel):
    income_amount: Optional[float] = None
    savings_goal: Optional[float] = None
    income_type: Optional[str] = None
    income_cycle: Optional[str] = None
    next_income_date: Optional[str] = None
    first_name: Optional[str] = None

class SmartPurchaseRequest(BaseModel):
    amount: float
    category: str
    description: Optional[str] = ""

# ── Helper functions ──────────────────────────────────────
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def get_user_financial_summary(user, expenses):
    total_expenses = sum(e.amount for e in expenses)
    income = user.income_amount or 0
    balance = income - total_expenses
    daily_burn = round(total_expenses / 30, 2) if total_expenses > 0 else 0
    spending_ratio = (total_expenses / income) if income > 0 else 0

    if user.next_income_date:
        today = date.today()
        days_remaining = max(0, (user.next_income_date - today).days)
    else:
        days_remaining = int(balance / daily_burn) if daily_burn > 0 else 30

    category_breakdown = {}
    for e in expenses:
        key = e.category.key if e.category else "other"
        category_breakdown[key] = category_breakdown.get(key, 0) + e.amount

    return {
        "total_expenses": total_expenses,
        "income": income,
        "balance": balance,
        "daily_burn": daily_burn,
        "spending_ratio": spending_ratio,
        "days_remaining": days_remaining,
        "category_breakdown": category_breakdown,
    }

def seed_categories(db: Session):
    if db.query(models.Category).count() == 0:
        categories = [
            models.Category(key='food', label='Food & Dining', icon='🍔', color='#F59E0B'),
            models.Category(key='transport', label='Transport', icon='🚗', color='#6366F1'),
            models.Category(key='shopping', label='Shopping', icon='🛍️', color='#EC4899'),
            models.Category(key='utilities', label='Utilities', icon='💡', color='#2DD4BF'),
            models.Category(key='health', label='Health', icon='💊', color='#10B981'),
            models.Category(key='entertainment', label='Entertainment', icon='🎮', color='#8B5CF6'),
            models.Category(key='savings', label='Savings', icon='💰', color='#1A2B47'),
            models.Category(key='education', label='Education', icon='📚', color='#3B82F6'),
            models.Category(key='other', label='Other', icon='📦', color='#94A3B8'),
        ]
        db.add_all(categories)
        db.commit()

def require_admin(current_user: models.User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# ── Startup ───────────────────────────────────────────────
def seed_admin(db: Session):
    existing = db.query(models.User).filter(
        models.User.username == "admin"
    ).first()
    if not existing:
        admin = models.User(
            username        = "admin",
            email           = "admin",
            first_name      = "Admin",
            hashed_password = hash_password("admin"),
            is_admin        = True,
        )
        db.add(admin)
        db.commit()
        print("Admin user seeded.")

@app.on_event("startup")
def startup():
    db = next(get_db())
    seed_categories(db)
    seed_admin(db)      

# ── Root ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"system": "SpendWise AI", "version": "1.0.0", "status": "running"}

# ── AUTH ──────────────────────────────────────────────────
@app.post("/auth/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(models.User).filter(models.User.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(
        username=request.username,
        email=request.email,
        first_name=request.first_name or "",
        hashed_password=hash_password(request.password),
        income_type=request.income_type or "other",
        income_cycle=request.income_cycle or "monthly",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id":           user.id,
            "username":     user.username,
            "email":        user.email,
            "name":         user.first_name or user.username,
            "income_type":  user.income_type,
            "income_cycle": user.income_cycle,
        },
    }


@app.post("/auth/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == request.username).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id":               user.id,
            "username":         user.username,
            "email":            user.email,
            "name":             user.first_name or user.username,
            "income_type":      user.income_type,
            "income_cycle":     user.income_cycle,
            "income_amount":    user.income_amount,
            "next_income_date": str(user.next_income_date) if user.next_income_date else None,
        },
    }


@app.post("/auth/logout")
def logout(current_user: models.User = Depends(get_current_user)):
    return {"message": "Logged out successfully"}


@app.get("/auth/me")
def get_me(current_user: models.User = Depends(get_current_user)):
    return {
        "id":               current_user.id,
        "username":         current_user.username,
        "email":            current_user.email,
        "name":             current_user.first_name or current_user.username,
        "income_type":      current_user.income_type,
        "income_cycle":     current_user.income_cycle,
        "income_amount":    current_user.income_amount,
        "next_income_date": str(current_user.next_income_date) if current_user.next_income_date else None,
        "is_admin":         current_user.is_admin,
    }

# ── PROFILE ───────────────────────────────────────────────
@app.get("/profile")
def get_profile(current_user: models.User = Depends(get_current_user)):
    return {
        "id":               current_user.id,
        "username":         current_user.username,
        "email":            current_user.email,
        "name":             current_user.first_name or current_user.username,
        "income_type":      current_user.income_type,
        "income_cycle":     current_user.income_cycle,
        "income_amount":    current_user.income_amount,
        "next_income_date": str(current_user.next_income_date) if current_user.next_income_date else None,
        "savings_goal":     current_user.savings_goal,
    }


@app.patch("/profile")
def update_profile(
    data: ProfileUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if data.income_amount    is not None: current_user.income_amount  = data.income_amount
    if data.savings_goal     is not None: current_user.savings_goal   = data.savings_goal
    if data.income_type      is not None: current_user.income_type    = data.income_type
    if data.income_cycle     is not None: current_user.income_cycle   = data.income_cycle
    if data.first_name       is not None: current_user.first_name     = data.first_name
    if data.next_income_date is not None:
        try:
            current_user.next_income_date = date.fromisoformat(data.next_income_date)
        except ValueError:
            pass

    db.commit()
    db.refresh(current_user)
    return {
        "id":               current_user.id,
        "income_type":      current_user.income_type,
        "income_cycle":     current_user.income_cycle,
        "income_amount":    current_user.income_amount,
        "next_income_date": str(current_user.next_income_date) if current_user.next_income_date else None,
        "savings_goal":     current_user.savings_goal,
        "first_name":       current_user.first_name,
    }

# ── EXPENSES ──────────────────────────────────────────────
@app.get("/expenses/stats")
def get_expense_stats(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    expenses = db.query(models.Expense).filter(models.Expense.user_id == current_user.id).all()
    summary = get_user_financial_summary(current_user, expenses)
    return {
        "total_expenses":      summary["total_expenses"],
        "total_income":        summary["income"],
        "balance":             summary["balance"],
        "average_daily_spend": summary["daily_burn"],
        "days_remaining":      summary["days_remaining"],
        "category_breakdown":  summary["category_breakdown"],
        "savings_goal":        current_user.savings_goal or 0,
    }


@app.get("/expenses")
def get_expenses(
    page: int = 1,
    page_size: int = 20,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * page_size
    expenses = (
        db.query(models.Expense)
        .filter(models.Expense.user_id == current_user.id)
        .order_by(models.Expense.created_at.desc())
        .offset(offset).limit(page_size).all()
    )
    return [
        {
            "id":           e.id,
            "amount":       e.amount,
            "description":  e.description,
            "timestamp":    e.timestamp,
            "created_at":   str(e.created_at),
            "category_key": e.category.key if e.category else "other",
            "category":     e.category.key if e.category else "other",
        }
        for e in expenses
    ]


@app.post("/expenses")
def create_expense(
    data: ExpenseCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    category = db.query(models.Category).filter(models.Category.key == data.category_key).first()
    if not category:
        category = db.query(models.Category).filter(models.Category.key == "other").first()
    expense = models.Expense(
        amount=data.amount,
        category_id=category.id if category else None,
        description=data.description,
        timestamp=data.timestamp,
        user_id=current_user.id,
    )
    db.add(expense)
    db.commit()
    db.refresh(expense)
    return {
        "id":           expense.id,
        "amount":       expense.amount,
        "description":  expense.description,
        "timestamp":    expense.timestamp,
        "created_at":   str(expense.created_at),
        "category_key": category.key if category else "other",
        "category":     category.key if category else "other",
    }


@app.patch("/expenses/{expense_id}")
def update_expense(
    expense_id: int,
    data: ExpenseUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    expense = db.query(models.Expense).filter(
        models.Expense.id == expense_id,
        models.Expense.user_id == current_user.id
    ).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    if data.amount      is not None: expense.amount      = data.amount
    if data.description is not None: expense.description = data.description
    expense.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(expense)
    return {
        "id":          expense.id,
        "amount":      expense.amount,
        "description": expense.description,
        "timestamp":   expense.timestamp,
        "category":    expense.category.key if expense.category else "other",
    }


@app.delete("/expenses/{expense_id}")
def delete_expense(
    expense_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    expense = db.query(models.Expense).filter(
        models.Expense.id == expense_id,
        models.Expense.user_id == current_user.id
    ).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    db.delete(expense)
    db.commit()
    return {"message": "Deleted"}


# ── SMART PURCHASE (Full ML Version) ─────────────────────
@app.post("/smart-purchase")
def smart_purchase(
    data: SmartPurchaseRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    expenses = db.query(models.Expense).filter(models.Expense.user_id == current_user.id).all()
    summary = get_user_financial_summary(current_user, expenses)

    category_breakdown = summary["category_breakdown"]
    top_category_amount = max(category_breakdown.values()) if category_breakdown else 0
    num_transactions = len(expenses)
    savings_goal_progress = (
        (current_user.savings_goal or 0) / (current_user.income_amount or 1)
    ) if current_user.income_amount else 0

    result = predict_smart_purchase(
        purchase_amount=data.amount,
        purchase_category=data.category,
        balance=summary["balance"],
        daily_burn_rate=summary["daily_burn"],
        days_remaining=summary["days_remaining"],
        spending_ratio=summary["spending_ratio"],
        savings_goal_progress=savings_goal_progress,
        income_amount=summary["income"],
        total_expenses=summary["total_expenses"],
        top_category_amount=top_category_amount,
        num_transactions=num_transactions,
    )

    return {
        "decision":                       result["decision"],
        "reason":                         result["reasoning"],
        "balance_before":                 result["current_balance"],
        "balance_after":                  result["remaining_after"],
        "estimated_days_until_shortfall": result.get("projected_days_left"),
        "suggestions":                    result["suggestions"],
        "risk_score":                     result["risk_score"],
        "cluster_tip":                    result.get("cluster_tip", ""),
    }


# ── INSIGHTS / ML ─────────────────────────────────────────
@app.get("/insights")
def get_insights(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    expenses = db.query(models.Expense).filter(models.Expense.user_id == current_user.id).all()
    summary = get_user_financial_summary(current_user, expenses)

    insight = db.query(models.MLInsight).filter(models.MLInsight.user_id == current_user.id).first()
    if not insight:
        insight = models.MLInsight(user_id=current_user.id)
        db.add(insight)
        db.commit()
        db.refresh(insight)

    if expenses:
        try:
            category_breakdown = summary["category_breakdown"]
            top_category_amount = max(category_breakdown.values()) if category_breakdown else 0
            num_transactions = len(expenses)
            savings_goal_progress = (
                (current_user.savings_goal or 0) / (current_user.income_amount or 1)
            ) if current_user.income_amount else 0

            cluster_result = predict_cluster(
                income_amount=summary["income"],
                total_expenses=summary["total_expenses"],
                spending_ratio=summary["spending_ratio"],
                top_category_amount=top_category_amount,
                num_transactions=num_transactions,
            )
            risk_result = predict_risk(
                balance=summary["balance"],
                daily_burn_rate=summary["daily_burn"],
                days_remaining=summary["days_remaining"],
                spending_ratio=summary["spending_ratio"],
                savings_goal_progress=savings_goal_progress,
            )
            sustainability_result = predict_sustainability(
                balance=summary["balance"],
                daily_burn_rate=summary["daily_burn"],
                days_remaining=summary["days_remaining"],
                spending_ratio=summary["spending_ratio"],
            )

            raw_risk = risk_result.get("risk_level", "safe")
            insight.risk_level = RISK_NORMALIZE.get(raw_risk.lower(), "low")
            insight.user_cluster = cluster_result.get("cluster", "Balanced")
            insight.cluster_description = cluster_result.get("description", "")
            insight.daily_burn_rate = summary["daily_burn"]
            insight.days_remaining = summary["days_remaining"]
            insight.model_accuracy = risk_result.get("accuracy", 94.2)
            insight.prediction = sustainability_result.get("sustainability", "on_track")
            insight.last_updated = datetime.utcnow()

            db.commit()
            db.refresh(insight)
        except Exception as e:
            print(f"ML prediction error: {e}")

    return {
        "user_cluster":        insight.user_cluster,
        "cluster_description": insight.cluster_description,
        "daily_burn_rate":     insight.daily_burn_rate,
        "days_remaining":      insight.days_remaining,
        "risk_level":          insight.risk_level,
        "model_accuracy":      insight.model_accuracy,
        "prediction":          insight.prediction,
        "last_updated":        insight.last_updated,
        "financial_summary":   summary,
    }

# ── ADMIN ROUTES ──────────────────────────────────────────
@app.get("/admin/reports")
def admin_reports(admin: models.User = Depends(require_admin), db: Session = Depends(get_db)):
    expenses = db.query(models.Expense).all()
    monthly = {}
    for e in expenses:
        month = str(e.created_at)[:7]
        if month not in monthly:
            monthly[month] = {"month": month, "users": set(), "totalSpend": 0, "alerts": 0, "savings": 0, "categories": {}}
        monthly[month]["users"].add(e.user_id)
        monthly[month]["totalSpend"] += e.amount
        cat = e.category.label if e.category else "Other"
        monthly[month]["categories"][cat] = monthly[month]["categories"].get(cat, 0) + e.amount

    alerts = db.query(models.Alert).all()
    for a in alerts:
        month = str(a.created_at)[:7]
        if month in monthly:
            monthly[month]["alerts"] += 1

    goals = db.query(models.SavingsGoal).all()
    for g in goals:
        month = str(g.created_at)[:7]
        if month in monthly:
            monthly[month]["savings"] += g.current_amount

    result = []
    for month, data in sorted(monthly.items()):
        user_count = len(data["users"])
        total = data["totalSpend"]
        top_cat = max(data["categories"], key=data["categories"].get) if data["categories"] else "N/A"
        result.append({
            "month": month,
            "users": user_count,
            "totalSpend": f"₱{total:,.2f}",
            "avgSpend": f"₱{(total / user_count):,.2f}" if user_count > 0 else "₱0.00",
            "alerts": data["alerts"],
            "savings": f"₱{data['savings']:,.2f}",
            "topCategory": top_cat,
        })
    return result


@app.get("/admin/dashboard")
def admin_dashboard(admin: models.User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    expenses = db.query(models.Expense).all()
    alerts = db.query(models.Alert).all()
    goals = db.query(models.SavingsGoal).all()

    total_spend = sum(e.amount for e in expenses)
    total_users = len(users)

    monthly = defaultdict(float)
    for e in expenses:
        month = str(e.created_at)[:7]
        monthly[month] += e.amount

    spending_trend = [
        {"month": m, "total": round(t, 2)}
        for m, t in sorted(monthly.items())[-6:]
    ]

    insights = db.query(models.MLInsight).all()
    cluster_distribution = defaultdict(int)
    for i in insights:
        label = i.user_cluster or "Unknown"
        cluster_distribution[label] += 1

    return {
        "total_users": total_users,
        "total_expenses": round(total_spend, 2),
        "total_alerts": len(alerts),
        "total_savings": round(sum(g.current_amount for g in goals), 2),
        "spending_trend": spending_trend,
        "cluster_distribution": dict(cluster_distribution),
    }


@app.get("/admin/ml-insights")
def admin_ml_insights(admin: models.User = Depends(require_admin), db: Session = Depends(get_db)):
    expenses = db.query(models.Expense).all()
    users = db.query(models.User).all()
    insights = db.query(models.MLInsight).all()

    total_users = len(users)
    total_expenses = sum(e.amount for e in expenses)
    total_alerts = db.query(models.Alert).count()
    avg_income = sum(u.income_amount or 0 for u in users) / total_users if total_users else 0

    category_totals = {}
    for e in expenses:
        if e.category:
            key = e.category.key
            if key not in category_totals:
                category_totals[key] = {
                    "key": key,
                    "label": e.category.label,
                    "color": e.category.color,
                    "total": 0,
                    "count": 0,
                }
            category_totals[key]["total"] += e.amount
            category_totals[key]["count"] += 1

    category_data = sorted(category_totals.values(), key=lambda x: x["total"], reverse=True)

    monthly = defaultdict(float)
    for e in expenses:
        month = str(e.created_at)[:7]
        monthly[month] += e.amount

    sorted_months = sorted(monthly.items())[-6:]
    prediction_data = []
    for month, actual in sorted_months:
        prediction_data.append({
            "month": month,
            "actual": round(actual, 2),
            "predicted": round(actual * 0.95, 2),
        })

    if sorted_months:
        last_month_str = sorted_months[-1][0]
        last_dt = datetime.strptime(last_month_str, "%Y-%m")
        next_dt = (last_dt.replace(day=1) + timedelta(days=32)).replace(day=1)
        avg_spend = total_expenses / total_users if total_users else 0
        prediction_data.append({
            "month": next_dt.strftime("%Y-%m"),
            "actual": None,
            "predicted": round(avg_spend * 1.05, 2),
        })

    high_risk = [i for i in insights if i.risk_level == "high"]
    top_flagged = []
    for i in high_risk[:5]:
        user = next((u for u in users if u.id == i.user_id), None)
        if user:
            top_flagged.append({
                "name": f"User #{user.id}",
                "avatar": "👤",
                "reason": f"Cluster: {i.user_cluster or 'Unknown'} · Burn rate: ₱{i.daily_burn_rate or 0:,.0f}/day",
                "risk": min(99, int((i.daily_burn_rate or 0) / max(user.income_amount or 1, 1) * 100)),
            })

    return {
        "total_users": total_users,
        "total_expenses": round(total_expenses, 2),
        "avg_income": round(avg_income, 2),
        "flagged_users": len(high_risk),
        "category_data": category_data,
        "prediction_data": prediction_data,
        "top_flagged": top_flagged,
    }


@app.get("/admin/clusters")
def admin_clusters(admin: models.User = Depends(require_admin), db: Session = Depends(get_db)):
    insights = db.query(models.MLInsight).all()
    cluster_counts = defaultdict(int)
    for i in insights:
        cluster_counts[i.user_cluster or "Unknown"] += 1

    color_map = {
        "Savers":    "#2DD4BF",
        "Balanced":  "#6366F1",
        "Impulsive": "#F59E0B",
        "At-Risk":   "#ef4444",
        "Unknown":   "#94A3B8",
    }
    desc_map = {
        "Savers":    "Consistently spend below income with strong savings habit",
        "Balanced":  "Healthy balance between spending and saving",
        "Impulsive": "Large portion of income spent, watch unplanned purchases",
        "At-Risk":   "Spending exceeds or nearly exceeds income, immediate review needed",
        "Unknown":   "Insufficient data to classify",
    }

    total = sum(cluster_counts.values()) or 1
    return [
        {
            "label": label,
            "value": count,
            "percentage": round((count / total) * 100, 1),
            "color": color_map.get(label, "#94A3B8"),
            "desc": desc_map.get(label, ""),
        }
        for label, count in sorted(cluster_counts.items(), key=lambda x: -x[1])
    ]


@app.get("/admin/users")
def admin_users(admin: models.User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    result = []
    for u in users:
        insight = db.query(models.MLInsight).filter(models.MLInsight.user_id == u.id).first()
        result.append({
            "id": u.id,
            "name": u.first_name or u.username,
            "email": u.email,
            "income_type": u.income_type,
            "income_cycle": u.income_cycle,
            "date_joined": str(u.created_at),
            "cluster": insight.user_cluster if insight else "Unknown",
            "risk_level": insight.risk_level if insight else "low",
        })
    return result


@app.get("/admin/users-detail")
def admin_users_detail(admin: models.User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    result = []
    for u in users:
        insight = db.query(models.MLInsight).filter(models.MLInsight.user_id == u.id).first()
        expenses = db.query(models.Expense).filter(models.Expense.user_id == u.id).all()
        total_spent = sum(e.amount for e in expenses)

        result.append({
            "id": u.id,
            "name": u.first_name or u.username,
            "email": u.email,
            "income": f"₱{u.income_amount:,.2f}" if u.income_amount else "₱0.00",
            "spent": f"₱{total_spent:,.2f}",
            "income_type": u.income_type,
            "income_cycle": u.income_cycle,
            "date_joined": str(u.created_at),
            "cluster": insight.user_cluster if insight else "Unknown",
            "risk_level": insight.risk_level if insight else "low",
            "spendingScore": round(
                (1 - min(total_spent, u.income_amount or 1) / max(u.income_amount or 1, 1)) * 100,
                1
            ) if u.income_amount else 50,
            "transactions": len(expenses),
            "avatar": "👤",
            "status": insight.risk_level if insight else "low",
        })
    return result

@app.get("/admin/users/{user_id}/expenses")
def admin_user_expenses(
    user_id: int,
    admin: models.User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    expenses = db.query(models.Expense).filter(
        models.Expense.user_id == user_id
    ).order_by(models.Expense.created_at.asc()).all()

    # Group by day of month
    daily = {}
    for e in expenses:
        day = e.created_at.day
        if day not in daily:
            daily[day] = []
        daily[day].append({
            "id":       str(e.id),
            "name":     e.description or "Expense",
            "category": e.category.label if e.category else "Other",
            "icon":     e.category.icon  if e.category else "📦",
            "price":    e.amount,
        })

    return {
        "user_id":        user_id,
        "income":         user.income_amount or 0,
        "daily_expenses": daily,
    }