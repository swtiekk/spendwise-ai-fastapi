"""
SpendWise AI — ML Prediction Functions
Loaded by FastAPI on startup.
Never run this directly — import it in main.py instead.
"""

import os
import pickle

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTER_MODEL_PATH = os.path.join(BASE_DIR, "cluster_model.pkl")
RISK_MODEL_PATH    = os.path.join(BASE_DIR, "risk_model.pkl")
SUSTAIN_MODEL_PATH = os.path.join(BASE_DIR, "sustainability_model.pkl")

# ── Load models once when FastAPI starts ───────────────────
with open(CLUSTER_MODEL_PATH, "rb") as f:
    cluster_bundle = pickle.load(f)

with open(RISK_MODEL_PATH, "rb") as f:
    risk_bundle = pickle.load(f)

with open(SUSTAIN_MODEL_PATH, "rb") as f:
    sustain_bundle = pickle.load(f)

# ── Cluster descriptions ───────────────────────────────────
CLUSTER_DESCRIPTIONS = {
    "Savers":    "You consistently spend below your income. You have a strong savings habit and low financial risk.",
    "Balanced":  "You maintain a healthy balance between spending and saving. Moderate financial risk.",
    "Impulsive": "You tend to spend a large portion of your income. Watch out for unplanned purchases.",
    "At-Risk":   "Your spending exceeds or nearly exceeds your income. Immediate budget review is recommended.",
}

# ── Sustainability messages ────────────────────────────────
SUSTAIN_MESSAGES = {
    "on_track": "Your funds are on track to last until the end of your income cycle.",
    "at_risk":  "Your funds may run out near the end of your cycle. Consider reducing daily spending.",
    "critical": "At your current burn rate, your funds may not last until your next income. Act now.",
}

# ──────────────────────────────────────────────────────────
# FUNCTION 1 — Predict spending behavior cluster
# ──────────────────────────────────────────────────────────
def predict_cluster(
    income_amount: float,
    total_expenses: float,
    spending_ratio: float,
    top_category_amount: float,
    num_transactions: int,
) -> dict:
    """
    Returns the user's spending behavior type.
    Output: Savers / Balanced / Impulsive / At-Risk
    """
    model     = cluster_bundle["model"]
    scaler    = cluster_bundle["scaler"]
    label_map = cluster_bundle["label_map"]

    features = [[
        income_amount,
        total_expenses,
        spending_ratio,
        top_category_amount,
        num_transactions,
    ]]

    scaled       = scaler.transform(features)
    cluster_id   = model.predict(scaled)[0]
    cluster_name = label_map[cluster_id]

    return {
        "cluster":     cluster_name,
        "description": CLUSTER_DESCRIPTIONS.get(cluster_name, "Unknown cluster"),
    }


# ──────────────────────────────────────────────────────────
# FUNCTION 2 — Predict risk level
# ──────────────────────────────────────────────────────────
def predict_risk(
    balance: float,
    daily_burn_rate: float,
    days_remaining: int,
    spending_ratio: float,
    savings_goal_progress: float,
) -> dict:
    """
    Returns the user's current financial risk level.
    Output: safe / caution / risky
    """
    model   = risk_bundle["model"]
    encoder = risk_bundle["encoder"]

    features = [[
        balance,
        daily_burn_rate,
        days_remaining,
        spending_ratio,
        savings_goal_progress,
    ]]

    encoded_pred = model.predict(features)[0]
    risk_label   = encoder.inverse_transform([encoded_pred])[0]

    # Get confidence probabilities
    proba      = model.predict_proba(features)[0]
    confidence = round(float(max(proba)) * 100, 2)

    return {
        "risk_level": risk_label,
        "confidence": confidence,
        "accuracy":   risk_bundle.get("accuracy", 85.0),
    }


# ──────────────────────────────────────────────────────────
# FUNCTION 3 — Predict sustainability (FIXED)
# ──────────────────────────────────────────────────────────
def predict_sustainability(
    balance: float,
    daily_burn_rate: float,
    days_remaining: int,
    spending_ratio: float,
) -> dict:
    """
    Returns whether the user's money will last their income cycle.
    Output: on_track / at_risk / critical
    """
    model   = sustain_bundle["model"]
    encoder = sustain_bundle["encoder"]

    # Calculate projected days (for response only)
    projected_days_left = round(balance / daily_burn_rate, 2) if daily_burn_rate > 0 else 30.0

    # IMPORTANT: Model was trained with 4 features only (no projected_days_left)
    features = [[
        balance,
        daily_burn_rate,
        days_remaining,
        spending_ratio,
    ]]

    encoded_pred   = model.predict(features)[0]
    sustain_label  = encoder.inverse_transform([encoded_pred])[0]

    proba      = model.predict_proba(features)[0]
    confidence = round(float(max(proba)) * 100, 2)

    return {
        "sustainability":      sustain_label,
        "projected_days_left": projected_days_left,
        "message":             SUSTAIN_MESSAGES.get(sustain_label, ""),
        "confidence":          confidence,
        "accuracy":            sustain_bundle.get("accuracy", 88.0),
    }


# ──────────────────────────────────────────────────────────
# FUNCTION 4 — Smart Purchase Adviser
# ──────────────────────────────────────────────────────────
def predict_smart_purchase(
    purchase_amount: float,
    purchase_category: str,
    balance: float,
    daily_burn_rate: float,
    days_remaining: int,
    spending_ratio: float,
    savings_goal_progress: float,
    income_amount: float,
    total_expenses: float,
    top_category_amount: float,
    num_transactions: int,
) -> dict:
    """
    Advises the user whether they should make a purchase.
    Uses all 3 models + rule-based logic.
    """

    # Get context from models
    cluster_result = predict_cluster(
        income_amount, total_expenses, spending_ratio,
        top_category_amount, num_transactions
    )
    sustain_result = predict_sustainability(
        balance, daily_burn_rate, days_remaining, spending_ratio
    )
    risk_result = predict_risk(
        balance, daily_burn_rate, days_remaining,
        spending_ratio, savings_goal_progress
    )

    cluster     = cluster_result["cluster"]
    sustain     = sustain_result["sustainability"]
    risk        = risk_result["risk_level"]
    proj_days   = sustain_result["projected_days_left"]

    # Rule-based decision
    safe_threshold    = balance * 0.10
    caution_threshold = balance * 0.25

    if balance <= 0:
        decision   = "risky"
        risk_score = 100
        reasoning  = f"Your balance is ₱{balance:,.2f}. Any purchase is not recommended."
        suggestions = ["You have no remaining budget.", "Wait for your next income cycle."]

    elif sustain == "critical" and purchase_amount > safe_threshold:
        decision   = "risky"
        risk_score = 90
        reasoning  = f"At your current burn rate, your funds may last only {proj_days:.0f} more days. A ₱{purchase_amount:,.2f} purchase is not advisable right now."
        suggestions = ["Defer this purchase until your next income cycle.", "Focus on essential expenses only."]

    elif purchase_amount <= safe_threshold:
        decision   = "safe"
        risk_score = int((purchase_amount / safe_threshold) * 30) if safe_threshold > 0 else 0
        reasoning  = f"₱{purchase_amount:,.2f} is within your safe spending range."
        suggestions = ["You can proceed with this purchase.", "Remember to log it in your expenses."]

    elif purchase_amount <= caution_threshold:
        decision   = "caution"
        risk_score = int(30 + ((purchase_amount - safe_threshold) / max(caution_threshold - safe_threshold, 1)) * 40)
        reasoning  = f"₱{purchase_amount:,.2f} is manageable but uses a significant portion of your balance."
        suggestions = ["Only proceed if this is a priority.", "Look for a lower-cost alternative."]

    else:
        decision   = "risky"
        risk_score = min(100, int(70 + ((purchase_amount - caution_threshold) / max(caution_threshold, 1)) * 30))
        reasoning  = f"₱{purchase_amount:,.2f} exceeds 25% of your ₱{balance:,.2f} balance."
        suggestions = ["Defer until your next pay cycle.", "Review your spending breakdown first."]

    cluster_tips = {
        "Savers":    "Keep up your great saving habits — only buy if truly necessary.",
        "Balanced":  "Stay balanced — make sure this fits your monthly plan.",
        "Impulsive": "Be mindful — impulsive purchases are your biggest risk.",
        "At-Risk":   "You are already at risk — avoid any non-essential spending.",
    }

    return {
        "decision":            decision,
        "risk_score":          risk_score,
        "reasoning":           reasoning,
        "suggestions":         suggestions,
        "cluster":             cluster,
        "sustainability":      sustain,
        "risk_level":          risk,
        "projected_days_left": proj_days,
        "current_balance":     balance,
        "remaining_after":     round(balance - purchase_amount, 2),
        "safe_threshold":      round(safe_threshold, 2),
        "caution_threshold":   round(caution_threshold, 2),
        "cluster_tip":         cluster_tips.get(cluster, ""),
    }