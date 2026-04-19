from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    _tablename_ = "users"

    id             = Column(Integer, primary_key=True, index=True)
    username       = Column(String, unique=True, index=True)
    email          = Column(String, unique=True, index=True)
    first_name     = Column(String, default="")
    hashed_password = Column(String)
    income_type    = Column(String, default="salary")
    income_cycle   = Column(String, default="monthly")
    income_amount  = Column(Float, default=0)
    savings_goal   = Column(Float, default=0)
    is_admin       = Column(Boolean, default=False)
    created_at     = Column(DateTime, default=datetime.utcnow)

    expenses       = relationship("Expense", back_populates="owner")
    alerts         = relationship("Alert", back_populates="owner")
    savings_goals  = relationship("SavingsGoal", back_populates="owner")


