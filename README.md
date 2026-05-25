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
- **Admin Routes** – `/admin/dashboard`, `/admin/reports`, `/admin/users`, `/admin/clusters`, `/admin/ml-insights` (admin-only, JWT-guarded)

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
main.py         – All API routes, JWT middleware, CORS, business logic
models.py       – SQLAlchemy ORM models (User, Expense, Category, SavingsGoal, Alert, MLInsight)
database.py     – DB engine and session factory
ml/train.py     – Trains and serializes the three ML models
ml/predict.py   – predict_cluster(), predict_risk(), predict_sustainability()
ml/*.pkl        – Pre-trained model artifacts

Request flow: `Client → CORS Middleware → JWT Auth → Route Handler → SQLAlchemy / ML → JSON Response`

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

