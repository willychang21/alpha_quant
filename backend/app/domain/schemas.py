from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class HoldingBase(BaseModel):
    ticker: str
    alias: Optional[str] = None
    quantity: float
    type: str
    average_cost: Optional[float] = 0.0
    purchase_date: Optional[str] = None
    account: Optional[str] = "Main Account"

class HoldingCreate(HoldingBase):
    pass

class Holding(HoldingBase):
    id: int
    portfolio_id: int

    class Config:
        from_attributes = True

class PortfolioBase(BaseModel):
    name: str
    cash: Optional[float] = 0.0
    currency: Optional[str] = "USD"

class PortfolioCreate(PortfolioBase):
    pass

class Portfolio(PortfolioBase):
    id: int
    holdings: List[Holding] = []

    class Config:
        from_attributes = True

# Config Schemas (for the specific JSON structure used by frontend)
class ETFConfig(BaseModel):
    ticker: str
    quantity: float
    averageCost: Optional[float] = 0
    purchaseDate: Optional[str] = ""
    account: Optional[str] = "Main Account"

class DirectConfig(BaseModel):
    ticker: str
    alias: Optional[str] = None
    quantity: float
    averageCost: Optional[float] = 0
    purchaseDate: Optional[str] = ""
    account: Optional[str] = "Main Account"

class PortfolioConfig(BaseModel):
    etfs: List[ETFConfig]
    direct: List[DirectConfig]
    cash: float
    currency: str

# Search Result Schema
class SearchResult(BaseModel):
    symbol: str
    shortname: str
    exchange: str
    typeDisp: str

# Asset Details Schema
class AssetHolding(BaseModel):
    ticker: str
    name: str
    percent: float

class AssetDetails(BaseModel):
    ticker: str
    type: str
    holdings: List[AssetHolding]

# Portfolio Data Response (Complex structure)
class MatrixHolding(BaseModel):
    ticker: str
    name: str
    percentOfPortfolio: float
    fundsHolding: int
    priceChangePercent1D: float
    pl1D: float
    flag: str
    account: str
    purchaseDate: str
    quantity: float
    averageCost: float
    lastPrice: float
    marketValue: float
    plExclFX: Any
    plFromFX: Any
    pl: Any
    plPercent: Any
    totalReturnPercent: Any
    etfs: Dict[str, float]
    direct: float
    price: float
    changePercent: float
    sector: Optional[str] = "Unknown"
    industry: Optional[str] = "Unknown"
    country: Optional[str] = "Unknown"
    type: Optional[str] = "Unknown"

class ProfitLossHolding(BaseModel):
    ticker: str
    name: str
    priceChangePercent1D: float
    pl1D: float
    percentOfPortfolio: float
    flag: str
    account: str
    purchaseDate: str
    quantity: float
    averageCost: float
    lastPrice: float
    marketValue: float
    plExclFX: Any
    plFromFX: Any
    pl: Any
    plPercent: Any
    totalReturnPercent: Any
    type: str
    sector: str
    industry: str
    country: str

class ETFHeader(BaseModel):
    id: str
    label: str
    weight: float

class PortfolioSummary(BaseModel):
    totalValue: float
    dayChange: float
    dayChangePercent: float
    holdingsCount: int

class PortfolioData(BaseModel):
    lastUpdated: str
    summary: PortfolioSummary
    matrixHoldings: List[MatrixHolding]
    profitLossHoldings: List[ProfitLossHolding]
    etfHeaders: List[ETFHeader]

# Backtest Schemas
class BacktestRequest(BaseModel):
    allocation: Dict[str, float]
    initialAmount: float = 10000
    monthlyAmount: float = 0

class BacktestPoint(BaseModel):
    date: str
    value: float
    invested: Optional[float] = None

class BacktestMetrics(BaseModel):
    pass # Add metrics if needed

class BacktestResponse(BaseModel):
    lumpSum: List[BacktestPoint]
    dca: List[BacktestPoint]
    monthlyReturns: List[Dict[str, Any]] # year, returns[], total
    metrics: Dict[str, Any]

# Analysis Schemas
class AnalysisRequest(BaseModel):
    monthlyAmount: float
    selectedAssets: List[Dict[str, Any]] # id, type, allocation, label
    dynamicCompositions: Optional[Dict[str, Dict[str, float]]] = {}
    stockNames: Optional[Dict[str, str]] = {}

class AnalysisResponse(BaseModel):
    matrixHoldings: List[MatrixHolding]
    etfHeaders: List[ETFHeader]

# Fundamental Analysis Schemas
class FundamentalData(BaseModel):
    ticker: str
    name: str
    sector: Optional[str] = "Unknown"
    industry: Optional[str] = "Unknown"
    currency: Optional[str] = "USD"
    price: float
    
    # Valuation
    peRatio: Optional[float] = None
    forwardPE: Optional[float] = None
    pegRatio: Optional[float] = None
    pbRatio: Optional[float] = None
    dividendYield: Optional[float] = None
    
    # Profitability
    roe: Optional[float] = None
    profitMargin: Optional[float] = None
    eps: Optional[float] = None
    revenueGrowth: Optional[float] = None
    
    # Financial Health
    debtToEquity: Optional[float] = None
    currentRatio: Optional[float] = None
    freeCashflow: Optional[float] = None
    fcfYield: Optional[float] = None # FCF / Market Cap
    
    # Valuation Extended
    evToEbitda: Optional[float] = None
    
    # Analyst
    targetMeanPrice: Optional[float] = None
    recommendationKey: Optional[str] = None

class ValueAnalysis(BaseModel):
    grahamNumber: Optional[float] = None
    isUndervalued: bool = False
    details: Dict[str, Any] = {}

class FundamentalResponse(BaseModel):
    data: FundamentalData
    analysis: ValueAnalysis

# Advanced Valuation Schemas
class DCFInput(BaseModel):
    revenue: float
    ebitda: float
    netIncome: float
    fcf: float
    totalDebt: float
    totalCash: float
    netDebt: float
    sharesOutstanding: float
    beta: float
    riskFreeRate: float
    marketRiskPremium: float

class DCFOutput(BaseModel):
    wacc: float
    growthRate: float
    terminalGrowthRate: float
    projectedFCF: List[float]
    projectedDiscountedFCF: Optional[List[float]] = None
    terminalValue: float
    terminalValueExitMultiple: Optional[float] = None
    presentValueSum: float
    equityValue: float
    equityValueExitMultiple: Optional[float] = None
    sharePrice: float
    sharePriceGordon: Optional[float] = None
    sharePriceExitMultiple: Optional[float] = None
    upside: float

class SensitivityRow(BaseModel):
    wacc: float
    prices: List[float] # Prices for different growth rates

class SensitivityAnalysis(BaseModel):
    growthRates: List[float]
    rows: List[SensitivityRow]

class WACCDetails(BaseModel):
    riskFreeRate: float
    betaRaw: float
    betaAdjusted: float
    marketRiskPremium: float
    costOfEquity: float
    costOfDebt: float
    taxRate: float
    afterTaxCostOfDebt: float
    equityWeight: float
    debtWeight: float
    wacc: float

class MultiplesAnalysis(BaseModel):
    eps: float
    peScenarios: Dict[str, float] # e.g. {"15x": 150.0, "20x": 200.0}
    ebitda: float
    evEbitdaScenarios: Dict[str, float] # e.g. {"10x": 120.0}
    currentPE: Optional[float] = None
    currentEvEbitda: Optional[float] = None

class SpecialAnalysis(BaseModel):
    title: str
    description: str
    models: List[Dict[str, Any]] # e.g. [{"name": "AI CapEx Model", "value": 150.0, "details": "..."}]
    blendedValue: float

class DDMOutput(BaseModel):
    dividendYield: float
    dividendGrowthRate: float
    costOfEquity: float
    fairValue: float
    modelType: str

class REITOutput(BaseModel):
    ffo: float
    ffoPerShare: float
    priceToFFO: float
    fairValue: float
    sectorAveragePFFO: float = 15.0 # Default benchmark

class MarketRiskIndicator(BaseModel):
    name: str
    value: float
    score: float # 0-100
    description: str
    signal: str
    history: List[float] = [] # Last 30 points for sparkline

class MarketRiskResponse(BaseModel):
    timestamp: str
    indicators: Dict[str, MarketRiskIndicator]
    score: float
    rating: str
    modelType: str = "FFO Multiples"

class QuantScore(BaseModel):
    quality: float # 0-100
    value: float # 0-100
    growth: float # 0-100
    momentum: float # 0-100
    total: float # 0-100
    details: Dict[str, str] # Explanation for each score

class ResidualIncomeOutput(BaseModel):
    bookValuePerShare: float
    roe: float
    costOfEquity: float
    fairValue: float
    modelType: str = "Residual Income"

class RiskMetrics(BaseModel):
    beta: float
    volatility: float # Annualized
    sharpeRatio: float
    maxDrawdown: float # 1Y Max Drawdown

class ValuationResult(BaseModel):
    ticker: str
    name: Optional[str] = "Unknown"
    sector: Optional[str] = "Unknown"
    industry: Optional[str] = "Unknown"
    price: Optional[float] = 0.0
    currency: Optional[str] = "USD"
    
    # Fundamental Ratios
    profitMargin: Optional[float] = None
    roe: Optional[float] = None
    dividendYield: Optional[float] = None
    
    # Financial Health
    debtToEquity: Optional[float] = None
    currentRatio: Optional[float] = None
    freeCashflow: Optional[float] = None
    operatingCashflow: Optional[float] = None
    
    # Valuation Extended
    forwardPE: Optional[float] = None
    pegRatio: Optional[float] = None
    priceToBook: Optional[float] = None
    bookValue: Optional[float] = None
    evToEbitda: Optional[float] = None
    fcfYield: Optional[float] = None
    
    # Analyst
    targetMeanPrice: Optional[float] = None
    recommendationKey: Optional[str] = None
    
    inputs: DCFInput
    dcf: Optional[DCFOutput] = None
    ddm: Optional[DDMOutput] = None
    reit: Optional[REITOutput] = None
    rim: Optional[ResidualIncomeOutput] = None # Residual Income Model
    quant: Optional[QuantScore] = None # Quant Factor Scores
    risk: Optional[RiskMetrics] = None # Risk Analysis
    waccDetails: Optional[WACCDetails] = None
    multiples: Optional[MultiplesAnalysis] = None
    specialAnalysis: Optional[SpecialAnalysis] = None
    sensitivity: Optional[SensitivityAnalysis] = None
    rating: str
    fairValueRange: List[float]

class AnalyzeRequest(BaseModel):
    ticker: str

class PortfolioRequest(BaseModel):
    holdings: List[Dict[str, Any]]
