from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    cash = Column(Float, default=0.0)
    currency = Column(String, default="USD")

    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")

class Holding(Base):
    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False)
    alias = Column(String, nullable=True)
    quantity = Column(Float, nullable=False)
    type = Column(String, nullable=False) # 'ETF', 'DIRECT', 'MUTUALFUND', 'CRYPTOCURRENCY', 'Stock'
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), default=1)
    average_cost = Column(Float, default=0.0)
    purchase_date = Column(String, nullable=True)
    account = Column(String, default="Main Account")

    portfolio = relationship("Portfolio", back_populates="holdings")
