from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True)
    email           = Column(String, unique=True, index=True)
    first_name      = Column(String, default="")
    hashed_password = Column(String)
    income_type     = Column(String, default="salary")
    income_cycle    = Column(String, default="monthly")
    income_amount   = Column(Float, default=0)
    savings_goal    = Column(Float, default=0)
    is_admin        = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

    expenses      = relationship("Expense", back_populates="owner")
    alerts        = relationship("Alert", back_populates="owner")
    savings_goals = relationship("SavingsGoal", back_populates="owner")


<<<<<<< HEAD
=======
class Category(Base):
    __tablename__ = "categories"

    id    = Column(Integer, primary_key=True, index=True)
    key   = Column(String, unique=True, index=True)
    label = Column(String)
    icon  = Column(String)
    color = Column(String)

    expenses = relationship("Expense", back_populates="category")


class Expense(Base):
    __tablename__ = "expenses"

    id          = Column(Integer, primary_key=True, index=True)
    amount      = Column(Float)
    description = Column(String, default="")
    timestamp   = Column(String)
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow)
    user_id     = Column(Integer, ForeignKey("users.id"))
    category_id = Column(Integer, ForeignKey("categories.id"))

    owner    = relationship("User", back_populates="expenses")
    category = relationship("Category", back_populates="expenses")

class SavingsGoal(Base):
    __tablename__ = "savings_goals"

    id             = Column(Integer, primary_key=True, index=True)
    name           = Column(String)
    target_amount  = Column(Float)
    current_amount = Column(Float, default=0)
    deadline       = Column(String, nullable=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    user_id        = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="savings_goals")


class Alert(Base):
    __tablename__ = "alerts"

    id         = Column(Integer, primary_key=True, index=True)
    type       = Column(String)
    title      = Column(String)
    message    = Column(Text)
    is_read    = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id    = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="alerts")
>>>>>>> b21368601a5db83370bcfe5d2fa04c35dba4d27d
