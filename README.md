# SpendWise AI – FastAPI Backend

## Project Description

SpendWise AI is an AI-powered personal finance management system. The FastAPI backend is the core engine powering both the mobile app and the admin web dashboard. It handles user authentication, expense tracking, savings goals, smart alerts, and machine learning–based financial insights.

## Features

- **User Authentication** – Register and login with JWT bearer tokens (24-hour expiry), passwords hashed with bcrypt
- **Expense Management** – Create, read, update, and delete expenses across 9 categories: Food & Dining, Transport, Shopping, Utilities, Health, Entertainment, Savings, Education, Other
- **Savings Goals** – Named goals with target amount, current progress, and optional deadline
- **Financial Alerts** – Auto-generated alerts for overspending and risk events
- **Profile Management** – Update income amount, income type (salary/freelance/other), income cycle, savings goal, and next income date
- **AI / ML Insights** – `/insights` endpoint returns:
  - User cluster: Saver / Balanced / Spender / Impulsive
  - Risk level: low / medium / high
  - Sustainability prediction: on_track / at_risk / critical
  - Daily burn rate and days-remaining estimate
- **Smart Purchase Advisor** – Evaluates a proposed purchase against current balance and spending patterns
- **Admin Routes** – `/admin/reports`, `/admin/users`, `/admin/ml-insights` (admin-only, JWT-guarded)

## Technology Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI |
| Server | Uvicorn |
| Database | SQLite via SQLAlchemy ORM |
| Authentication | python-jose (JWT), passlib + bcrypt |
| File Uploads | python-multipart |
| Machine Learning | scikit-learn, pandas, numpy |
| Deployment | Render (`web: uvicorn main:app --host 0.0.0.0 --port $PORT`) |



## System Architecture

```
spendwise-ai-fastapi/
├── main.py         – All API routes, JWT middleware, CORS, business logic
├── models.py       – SQLAlchemy ORM models (User, Expense, Category, SavingsGoal, Alert, MLInsight)
├── database.py     – DB engine and session factory
└── ml/
    ├── train.py    – Trains and serializes the three ML models
    ├── predict.py  – predict_cluster(), predict_risk(), predict_sustainability()
    └── *.pkl       – Pre-trained model artifacts
```

**Request flow:** `Client → CORS Middleware → JWT Auth → Route Handler → SQLAlchemy / ML → JSON Response`

## API Endpoints

> All protected endpoints require `Authorization: Bearer <token>` in the request header.

### General

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | ❌ | Health check – returns system name, version, and status |

---

### Auth

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/register` | ❌ | Register a new user |
| `POST` | `/auth/login` | ❌ | Login and receive a JWT token |
| `GET` | `/auth/me` | ✅ | Get the current authenticated user's info |

**POST** `/auth/register`
```json
{
  "username": "john",
  "email": "john@email.com",
  "password": "secret",
  "first_name": "John",
  "income_type": "salary",
  "income_cycle": "monthly"
}
```

**POST** `/auth/login`
```json
{
  "username": "john",
  "password": "secret"
}
```

---

### Profile

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/profile` | ✅ | Get the current user's profile |
| `PATCH` | `/profile` | ✅ | Update income, savings goal, income type/cycle, next income date |

**PATCH** `/profile`
```json
{
  "income_amount": 30000,
  "savings_goal": 5000,
  "income_type": "salary",
  "income_cycle": "monthly",
  "next_income_date": "2026-06-01",
  "first_name": "John"
}
```

---

### Expenses

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/expenses` | ✅ | Get paginated list of expenses (`?page=1&page_size=20`) |
| `GET` | `/expenses/stats` | ✅ | Get spending stats (total, by category, balance, daily burn) |
| `POST` | `/expenses` | ✅ | Create a new expense |
| `PATCH` | `/expenses/{expense_id}` | ✅ | Update amount or description of an expense |
| `DELETE` | `/expenses/{expense_id}` | ✅ | Delete an expense |

**POST** `/expenses`
```json
{
  "amount": 150.00,
  "category_key": "food",
  "description": "Lunch at Jollibee",
  "timestamp": "2026-05-26T12:00:00"
}
```

---

### Savings Goals

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/savings-goals` | ✅ | List all savings goals with progress % |
| `POST` | `/savings-goals` | ✅ | Create a new savings goal |
| `PATCH` | `/savings-goals/{goal_id}` | ✅ | Update current amount or name |
| `DELETE` | `/savings-goals/{goal_id}` | ✅ | Delete a savings goal |

**POST** `/savings-goals`
```json
{
  "name": "Emergency Fund",
  "target_amount": 10000,
  "current_amount": 2000,
  "deadline": "2026-12-31"
}
```

---

### Alerts

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/alerts` | ✅ | Get all alerts for the current user |
| `PATCH` | `/alerts/{alert_id}/read` | ✅ | Mark an alert as read |

---

### Categories

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/categories` | ❌ | List all expense categories (key, label, icon, color) |

---

### ML / Insights

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/insights` | ✅ | Get ML predictions: cluster, risk level, sustainability, burn rate |
| `POST` | `/smart-purchase` | ✅ | Evaluate whether a proposed purchase is advisable |

**POST** `/smart-purchase`
```json
{
  "amount": 2500.00,
  "category": "shopping",
  "description": "New shoes"
}
```

---

### Admin

> All admin endpoints require the user to have `is_admin = True`.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/admin/reports` | ✅ 🔒 | Monthly report: users, total spend, avg spend, alerts, savings, top category |
| `GET` | `/admin/ml-insights` | ✅ 🔒 | Aggregated ML data: category totals, user count, total expenses, alert count |
| `GET` | `/admin/users` | ✅ 🔒 | Full user list with cluster and risk level from ML insights |

---

## Installation & Setup

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/swtiekk/spendwise-ai-fastapi.git
cd spendwise-ai-fastapi

python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://127.0.0.1:8000/docs`

## Deployment Link
https://spendwise-ai-fastapi-zic2.onrender.com/

## Team Members and Roles

Sotie Katrina Golez  
Florie Jayne Soler  
Trisha Araquil  
Steve Drylle Sarino  

## Known Limitations

- SQLite is used for both development and deployment; not suitable for high-concurrency production workloads
- ML models are trained on synthetic data and may not generalize to all user profiles
- `SECRET_KEY` is hardcoded in `main.py` and should be moved to environment variables for production
- CORS is set to allow all origins (`*`) and should be restricted in production

## Screenshots
![ Screenshots ](docs/Screenshot 2026-05-26 145805.png)
![Screenshots](Screenshot 2026-05-26 145733.png) ![alt text](<Screenshot 2026-05-26 144422.png>) ![alt text](<Screenshot 2026-05-26 144436.png>) ![alt text](<Screenshot 2026-05-26 145638.png>) ![alt text](<Screenshot 2026-05-26 145711.png>)