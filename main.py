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
    version="1.0.0",
    redirect_slashes=False,
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
    username:   str
    email:      str
    password:   str
    first_name: Optional[str] = ""
    income_type:  Optional[str] = "other"
    income_cycle: Optional[str] = "monthly"

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

class ProfileUpdate(BaseModel):
    income_amount: Optional[float] = None
    savings_goal:  Optional[float] = None
    income_type:   Optional[str]   = None
    income_cycle:  Optional[str]   = None

class SmartPurchaseRequest(BaseModel):
    amount:      float
    category:    str
    description: Optional[str] = ""

# ── Helper functions ──────────────────────────────────────
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_token(data: dict):
    to_encode = data.copy()
    expire    = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db:    Session = Depends(get_db)
):
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def seed_categories(db: Session):
    if db.query(models.Category).count() == 0:
        categories = [
            models.Category(key='food',          label='Food & Dining',  icon='🍔', color='#F59E0B'),
            models.Category(key='transport',     label='Transport',       icon='🚗', color='#6366F1'),
            models.Category(key='shopping',      label='Shopping',        icon='🛍️', color='#EC4899'),
            models.Category(key='utilities',     label='Utilities',       icon='💡', color='#2DD4BF'),
            models.Category(key='health',        label='Health',          icon='💊', color='#10B981'),
            models.Category(key='entertainment', label='Entertainment',   icon='🎮', color='#8B5CF6'),
            models.Category(key='savings',       label='Savings',         icon='💰', color='#1A2B47'),
            models.Category(key='education',     label='Education',       icon='📚', color='#3B82F6'),
            models.Category(key='other',         label='Other',           icon='📦', color='#94A3B8'),
        ]
        db.add_all(categories)
        db.commit()

# ── Startup ───────────────────────────────────────────────
@app.on_event("startup")
def startup():
    db = next(get_db())
    seed_categories(db)

# ── Root ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "system":  "SpendWise AI",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "http://127.0.0.1:8000/docs",
    }

# ── AUTH ──────────────────────────────────────────────────
@app.post("/auth/register", status_code=201)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    if db.query(models.User).filter(models.User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = models.User(
        username        = data.username,
        email           = data.email,
        first_name      = data.first_name,
        hashed_password = hash_password(data.password),
        income_type     = data.income_type,
        income_cycle    = data.income_cycle,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    insight = models.MLInsight(user_id=user.id)
    db.add(insight)
    db.commit()

    return {
        "id":         user.id,
        "username":   user.username,
        "email":      user.email,
        "first_name": user.first_name,
    }

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/login")
def login(
    data: LoginRequest,
    db: Session = Depends(get_db)
):
    # Check by username first, then by email
    user = db.query(models.User).filter(
        models.User.username == data.username
    ).first()

    if not user:
        user = db.query(models.User).filter(
            models.User.email == data.username
        ).first()

    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":         user.id,
            "username":   user.username,
            "email":      user.email,
            "first_name": user.first_name,
            "is_admin":   user.is_admin,
        }
    }

@app.get("/auth/me")
def me(current_user: models.User = Depends(get_current_user)):
    return {
        "id":            current_user.id,
        "username":      current_user.username,
        "email":         current_user.email,
        "first_name":    current_user.first_name,
        "income_type":   current_user.income_type,
        "income_cycle":  current_user.income_cycle,
        "income_amount": current_user.income_amount,
        "savings_goal":  current_user.savings_goal,
        "is_admin":      current_user.is_admin,  # ← ADD THIS
    }

# ── PROFILE ───────────────────────────────────────────────

@app.get("/profile")
def get_profile(
    current_user: models.User = Depends(get_current_user)
):
    """
    This allows the mobile app to GET the user's data 
    to display it on the profile screen.
    """
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "income_amount": current_user.income_amount,
        "savings_goal": current_user.savings_goal,
        "income_type": current_user.income_type,
        "income_cycle": current_user.income_cycle,
    }

@app.patch("/profile")
def update_profile(
    data:         ProfileUpdate,
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.id == current_user.id).first()
    if data.income_amount is not None: user.income_amount = data.income_amount
    if data.savings_goal  is not None: user.savings_goal  = data.savings_goal
    if data.income_type   is not None: user.income_type   = data.income_type
    if data.income_cycle  is not None: user.income_cycle  = data.income_cycle
    db.commit()
    db.refresh(user)
    return {
        "income_amount": user.income_amount,
        "savings_goal":  user.savings_goal,
        "income_type":   user.income_type,
        "income_cycle":  user.income_cycle,
    }

# ── CATEGORIES ────────────────────────────────────────────
@app.get("/categories")
def get_categories(db: Session = Depends(get_db)):
    return db.query(models.Category).all()

# ── EXPENSES ──────────────────────────────────────────────
@app.get("/expenses")
def get_expenses(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    expenses = db.query(models.Expense).filter(
        models.Expense.user_id == current_user.id
    ).order_by(models.Expense.created_at.desc()).all()

    return [{
        "id":          e.id,
        "amount":      e.amount,
        "description": e.description,
        "timestamp":   e.timestamp,
        "created_at":  e.created_at,
        "updated_at":  e.updated_at,
        "category": {
            "key":   e.category.key   if e.category else None,
            "label": e.category.label if e.category else None,
            "icon":  e.category.icon  if e.category else None,
            "color": e.category.color if e.category else None,
        }
    } for e in expenses]

@app.post("/expenses", status_code=201)
def create_expense(
    data:         ExpenseCreate,
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    category = db.query(models.Category).filter(
        models.Category.key == data.category_key
    ).first()
    if not category:
        raise HTTPException(status_code=400, detail=f"Category '{data.category_key}' not found")

    expense = models.Expense(
        amount      = data.amount,
        description = data.description,
        timestamp   = data.timestamp,
        user_id     = current_user.id,
        category_id = category.id,
    )
    db.add(expense)
    db.commit()
    db.refresh(expense)
    return {
        "id":          expense.id,
        "amount":      expense.amount,
        "description": expense.description,
        "timestamp":   expense.timestamp,
        "category":    {"key": category.key, "label": category.label, "icon": category.icon, "color": category.color},
        "message":     "Expense created successfully"
    }

@app.get("/expenses/stats")
def get_expense_stats(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    expenses = db.query(models.Expense).filter(
        models.Expense.user_id == current_user.id
    ).all()

    total_expenses = sum(e.amount for e in expenses)
    income         = current_user.income_amount or 0
    balance        = income - total_expenses

    breakdown = {}
    for e in expenses:
        key = e.category.key if e.category else "other"
        breakdown[key] = breakdown.get(key, 0) + e.amount

    return {
        "total_expenses":      total_expenses,
        "total_income":        income,
        "balance":             balance,
        "average_daily_spend": round(total_expenses / 30, 2),
        "days_remaining":      14,
        "category_breakdown":  breakdown,
    }

@app.patch("/expenses/{expense_id}")
def update_expense(
    expense_id:   int,
    data:         ExpenseUpdate,
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    expense = db.query(models.Expense).filter(
        models.Expense.id      == expense_id,
        models.Expense.user_id == current_user.id
    ).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")

    if data.amount      is not None: expense.amount      = data.amount
    if data.description is not None: expense.description = data.description
    expense.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(expense)
    return {"id": expense.id, "amount": expense.amount, "description": expense.description, "message": "Updated successfully"}

@app.delete("/expenses/{expense_id}", status_code=204)
def delete_expense(
    expense_id:   int,
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    expense = db.query(models.Expense).filter(
        models.Expense.id      == expense_id,
        models.Expense.user_id == current_user.id
    ).first()
    if not expense:
        raise HTTPException(status_code=404, detail="Expense not found")
    db.delete(expense)
    db.commit()

# ── DASHBOARD ─────────────────────────────────────────────
@app.get("/dashboard")
def dashboard(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    expenses = db.query(models.Expense).filter(
        models.Expense.user_id == current_user.id
    ).all()

    total_expenses = sum(e.amount for e in expenses)
    income         = current_user.income_amount or 0
    balance        = income - total_expenses

    breakdown = {}
    for e in expenses:
        key = e.category.key if e.category else "other"
        breakdown[key] = breakdown.get(key, 0) + e.amount

    return {
        "balance":             balance,
        "total_expenses":      total_expenses,
        "total_income":        income,
        "average_daily_spend": round(total_expenses / 30, 2),
        "savings_goal":        current_user.savings_goal or 0,
        "category_breakdown":  breakdown,
    }

# ── SAVINGS GOALS ─────────────────────────────────────────
@app.get("/savings-goals")
def get_savings_goals(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    return db.query(models.SavingsGoal).filter(
        models.SavingsGoal.user_id == current_user.id
    ).all()

@app.post("/savings-goals", status_code=201)
def create_savings_goal(
    data:         SavingsGoalCreate,
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    goal = models.SavingsGoal(
        name           = data.name,
        target_amount  = data.target_amount,
        current_amount = data.current_amount,
        deadline       = data.deadline,
        user_id        = current_user.id,
    )
    db.add(goal)
    db.commit()
    db.refresh(goal)
    return goal

@app.patch("/savings-goals/{goal_id}")
def update_savings_goal(
    goal_id:      int,
    data:         SavingsGoalUpdate,
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    goal = db.query(models.SavingsGoal).filter(
        models.SavingsGoal.id      == goal_id,
        models.SavingsGoal.user_id == current_user.id
    ).first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    if data.current_amount is not None: goal.current_amount = data.current_amount
    if data.name           is not None: goal.name           = data.name
    db.commit()
    db.refresh(goal)
    return goal

@app.delete("/savings-goals/{goal_id}", status_code=204)
def delete_savings_goal(
    goal_id:      int,
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    goal = db.query(models.SavingsGoal).filter(
        models.SavingsGoal.id      == goal_id,
        models.SavingsGoal.user_id == current_user.id
    ).first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    db.delete(goal)
    db.commit()

# ── ALERTS ────────────────────────────────────────────────
@app.get("/alerts")
def get_alerts(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    return db.query(models.Alert).filter(
        models.Alert.user_id == current_user.id
    ).order_by(models.Alert.created_at.desc()).all()

# ── INSIGHTS ──────────────────────────────────────────────
@app.get("/insights")
def get_insights(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    insight = db.query(models.MLInsight).filter(
        models.MLInsight.user_id == current_user.id
    ).first()
    if not insight:
        raise HTTPException(status_code=404, detail="No insights found")
    return {
        "user_cluster":        insight.user_cluster,
        "cluster_description": insight.cluster_description,
        "daily_burn_rate":     insight.daily_burn_rate,
        "days_remaining":      insight.days_remaining,
        "risk_level":          insight.risk_level,
        "model_accuracy":      insight.model_accuracy,
        "prediction":          insight.prediction,
        "last_updated":        insight.last_updated,
    }

# ── SMART PURCHASE ────────────────────────────────────────
@app.post("/smart-purchase")
def smart_purchase(
    data:         SmartPurchaseRequest,
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    expenses = db.query(models.Expense).filter(
        models.Expense.user_id == current_user.id
    ).all()
    total_expenses    = sum(e.amount for e in expenses)
    income            = current_user.income_amount or 0
    balance           = income - total_expenses
    safe_threshold    = balance * 0.10
    caution_threshold = balance * 0.25
    amount            = data.amount

    if balance <= 0:
        decision, risk_score = "risky", 100
        reasoning   = f"Your balance is ₱{balance:,.2f}. Any purchase is not recommended."
        suggestions = ["You have no remaining budget.", "Wait for your next income cycle."]
    elif amount <= safe_threshold:
        decision   = "safe"
        risk_score = int((amount / safe_threshold) * 30) if safe_threshold > 0 else 0
        reasoning  = f"₱{amount:,.2f} is within your safe range based on balance of ₱{balance:,.2f}."
        suggestions = ["You can proceed.", "Log it immediately after buying."]
    elif amount <= caution_threshold:
        decision   = "caution"
        risk_score = int(30 + ((amount - safe_threshold) / (caution_threshold - safe_threshold)) * 40)
        reasoning  = f"₱{amount:,.2f} is manageable but uses a significant portion of your ₱{balance:,.2f} balance."
        suggestions = ["Only proceed if priority.", "Look for a lower-cost alternative."]
    else:
        decision   = "risky"
        risk_score = min(100, int(70 + ((amount - caution_threshold) / max(caution_threshold, 1)) * 30))
        reasoning  = f"₱{amount:,.2f} exceeds 25% of your balance of ₱{balance:,.2f}."
        suggestions = ["Defer until next pay cycle.", "Review spending breakdown first."]

    return {
        "decision":          decision,
        "risk_score":        risk_score,
        "reasoning":         reasoning,
        "suggestions":       suggestions,
        "current_balance":   balance,
        "remaining_budget":  balance - amount,
        "safe_threshold":    safe_threshold,
        "caution_threshold": caution_threshold,
    }

# ── ADMIN ─────────────────────────────────────────────────
@app.get("/admin/users")
def admin_users(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    users = db.query(models.User).all()
    return [{
        "id":           u.id,
        "username":     u.username,
        "email":        u.email,
        "name":         u.first_name,
        "income_type":  u.income_type,
        "income_cycle": u.income_cycle,
        "date_joined":  u.created_at,
    } for u in users]

@app.get("/admin/dashboard")
def admin_dashboard(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    total_users    = db.query(models.User).count()
    total_expenses = sum(e.amount for e in db.query(models.Expense).all())
    return {
        "total_users":    total_users,
        "total_expenses": total_expenses,
    }

@app.get("/admin/ml-insights")
def admin_ml_insights(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    users    = db.query(models.User).all()
    expenses = db.query(models.Expense).all()

    total_users    = len(users)
    total_expenses = sum(e.amount for e in expenses)
    avg_income     = sum(u.income_amount or 0 for u in users) / max(total_users, 1)

    # Category breakdown across all users
    category_breakdown = {}
    for e in expenses:
        key   = e.category.key   if e.category else "other"
        label = e.category.label if e.category else "Other"
        if key not in category_breakdown:
            category_breakdown[key] = {"key": key, "label": label, "total": 0, "count": 0}
        category_breakdown[key]["total"] += e.amount
        category_breakdown[key]["count"] += 1

    category_data = sorted(category_breakdown.values(), key=lambda x: x["total"], reverse=True)

    # Per-user stats for prediction data
    prediction_data = []
    for u in users:
        user_expenses = [e.amount for e in expenses if e.user_id == u.id]
        total_spent   = sum(user_expenses)
        income        = u.income_amount or 0
        prediction_data.append({
            "user":      u.username,
            "actual":    round(total_spent, 2),
            "predicted": round(income * 0.75, 2),  # placeholder prediction
            "income":    income,
        })

    # Top flagged users (spent > 80% of income)
    top_flagged = []
    for u in users:
        user_expenses = sum(e.amount for e in expenses if e.user_id == u.id)
        income        = u.income_amount or 1
        ratio         = user_expenses / income if income > 0 else 0
        if ratio > 0.5:
            top_flagged.append({
                "user":    u.username,
                "amount":  round(user_expenses, 2),
                "income":  income,
                "ratio":   round(ratio * 100, 1),
                "risk":    "high" if ratio > 0.9 else "medium" if ratio > 0.7 else "low",
            })
    top_flagged = sorted(top_flagged, key=lambda x: x["ratio"], reverse=True)[:10]

    # ML metrics summary
    ml_metrics = {
        "total_users":     total_users,
        "total_expenses":  round(total_expenses, 2),
        "avg_income":      round(avg_income, 2),
        "flagged_users":   len(top_flagged),
        "category_data":   category_data,
        "prediction_data": prediction_data,
        "top_flagged":     top_flagged,
    }

    return ml_metrics


@app.get("/admin/users-detail")
def admin_users_detail(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    users    = db.query(models.User).all()
    expenses = db.query(models.Expense).all()

    result = []
    for u in users:
        user_expenses   = [e for e in expenses if e.user_id == u.id]
        total_spent     = sum(e.amount for e in user_expenses)
        income          = u.income_amount or 0
        spending_ratio  = (total_spent / income * 100) if income > 0 else 0

        result.append({
            "id":            u.id,
            "username":      u.username,
            "email":         u.email,
            "name":          u.first_name,
            "income":        income,
            "income_type":   u.income_type,
            "income_cycle":  u.income_cycle,
            "spent":         round(total_spent, 2),
            "transactions":  len(user_expenses),
            "spendingScore": round(spending_ratio, 1),
            "date_joined":   u.created_at,
        })

    return result

@app.get("/admin/clusters")
def admin_clusters(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    users    = db.query(models.User).all()
    expenses = db.query(models.Expense).all()

    savers    = 0
    balanced  = 0
    impulsive = 0
    at_risk   = 0

    for u in users:
        income       = u.income_amount or 0
        user_spent   = sum(e.amount for e in expenses if e.user_id == u.id)
        if income <= 0:
            balanced += 1
            continue
        ratio = user_spent / income
        if ratio < 0.4:
            savers    += 1
        elif ratio < 0.7:
            balanced  += 1
        elif ratio < 0.9:
            impulsive += 1
        else:
            at_risk   += 1

    return [
        { "label": "Savers",    "value": savers,    "color": "#2DD4BF", "desc": "Consistently under budget, high savings rate" },
        { "label": "Balanced",  "value": balanced,  "color": "#6366F1", "desc": "Moderate spending, occasional alerts" },
        { "label": "Impulsive", "value": impulsive, "color": "#F59E0B", "desc": "Frequent unplanned purchases detected" },
        { "label": "At-Risk",   "value": at_risk,   "color": "#ef4444", "desc": "Exceeding budget, multiple alerts flagged" },
    ]

@app.get("/admin/reports")
def admin_reports(
    current_user: models.User = Depends(get_current_user),
    db:           Session     = Depends(get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    expenses = db.query(models.Expense).all()
    users    = db.query(models.User).all()

    # Group expenses by month
    monthly = {}
    for e in expenses:
        try:
            dt    = datetime.fromisoformat(str(e.timestamp))
            month = dt.strftime("%Y-%m")
        except:
            month = "Unknown"

        if month not in monthly:
            monthly[month] = {
                "total":      0,
                "user_ids":   set(),
                "categories": {},
                "alerts":     0,
            }
        monthly[month]["total"]    += e.amount
        monthly[month]["user_ids"].add(e.user_id)

        cat_label = e.category.label if e.category else "Other"
        monthly[month]["categories"][cat_label] = (
            monthly[month]["categories"].get(cat_label, 0) + e.amount
        )

    # Build per-user income map
    income_map = { u.id: (u.income_amount or 0) for u in users }

    result = []
    for month in sorted(monthly.keys()):
        data         = monthly[month]
        total        = data["total"]
        active_users = len(data["user_ids"])
        avg_spend    = round(total / active_users, 2) if active_users > 0 else 0

        # Savings = sum of income for active users minus their total spend
        total_income  = sum(income_map.get(uid, 0) for uid in data["user_ids"])
        savings       = max(0, total_income - total)

        # Top category
        top_category = max(data["categories"], key=data["categories"].get) if data["categories"] else "N/A"

        # Alerts = users who spent more than 80% of income that month
        alerts = 0
        for uid in data["user_ids"]:
            user_month_spend = sum(
                e.amount for e in expenses
                if e.user_id == uid and
                datetime.fromisoformat(str(e.timestamp)).strftime("%Y-%m") == month
            )
            income = income_map.get(uid, 0)
            if income > 0 and user_month_spend / income > 0.8:
                alerts += 1

        result.append({
            "month":       month,
            "users":       active_users,
            "totalSpend":  f"₱{total:,.2f}",
            "avgSpend":    f"₱{avg_spend:,.2f}",
            "alerts":      alerts,
            "savings":     f"₱{savings:,.2f}",
            "topCategory": top_category,
        })

    return result