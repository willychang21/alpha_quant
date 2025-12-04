from sqlalchemy import Column, Integer, String, Float, Date, Boolean, ForeignKey, BigInteger
from sqlalchemy.orm import relationship
from app.core.database import Base

class Security(Base):
    __tablename__ = "securities"

    sid = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    exchange = Column(String, nullable=True)
    type = Column(String, nullable=True) # Equity, ETF
    active = Column(Boolean, default=True)

    market_data = relationship("MarketDataDaily", back_populates="security")

class MarketDataDaily(Base):
    __tablename__ = "market_data_daily"

    id = Column(Integer, primary_key=True, index=True)
    sid = Column(Integer, ForeignKey("securities.sid"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(BigInteger)

    security = relationship("Security", back_populates="market_data")

class Fundamentals(Base):
    __tablename__ = "fundamentals"

    id = Column(Integer, primary_key=True, index=True)
    sid = Column(Integer, ForeignKey("securities.sid"), nullable=False)
    date = Column(Date, nullable=False, index=True) # Filing date or Period End Date
    metric = Column(String, nullable=False) # e.g., 'revenue', 'ebitda', 'net_income'
    value = Column(Float)
    period = Column(String) # '12M', 'Q'

    security = relationship("Security", back_populates="fundamentals")

class ModelSignals(Base):
    __tablename__ = "model_signals"

    id = Column(Integer, primary_key=True, index=True)
    sid = Column(Integer, ForeignKey("securities.sid"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    model_name = Column(String, nullable=False) # 'ranking_v1'
    score = Column(Float)
    rank = Column(Integer)
    metadata_json = Column(String) # Store factor breakdown

    security = relationship("Security", back_populates="signals")

class PortfolioTargets(Base):
    __tablename__ = "portfolio_targets"

    id = Column(Integer, primary_key=True, index=True)
    sid = Column(Integer, ForeignKey("securities.sid"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    model_name = Column(String, nullable=False) # 'mvo_sharpe'
    weight = Column(Float, nullable=False)
    
    security = relationship("Security", back_populates="targets")

Security.fundamentals = relationship("Fundamentals", back_populates="security")
Security.signals = relationship("ModelSignals", back_populates="security")
Security.targets = relationship("PortfolioTargets", back_populates="security")
