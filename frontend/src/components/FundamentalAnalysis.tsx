import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useParams } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  TrendingUp, 
  DollarSign, 
  Activity, 
  CheckCircle2,
  Sparkles,
  Search,
  Scale,
  AlertCircle
} from "lucide-react";
import api from '../api/axios';
import { API_ENDPOINTS } from '../api/endpoints';

// ... (interfaces remain the same)
interface FundamentalData {
  ticker: string;
  name: string;
  sector: string;
  industry: string;
  currency: string;
  price: number;
  peRatio?: number;
  forwardPE?: number;
  pegRatio?: number;
  pbRatio?: number;
  dividendYield?: number;
  roe?: number;
  profitMargin?: number;
  eps?: number;
  revenueGrowth?: number;
  debtToEquity?: number;
  currentRatio?: number;
  freeCashflow?: number;
  fcfYield?: number;
  evToEbitda?: number;
  targetMeanPrice?: number;
  recommendationKey?: string;
  staticPE?: number;
}

interface ValueAnalysis {
  grahamNumber?: number;
  isUndervalued: boolean;
  details: Record<string, string>;
}

// Unified Interface matching Backend ValuationResult
interface ValuationResult {
  ticker: string;
  name?: string;
  sector?: string;
  industry?: string;
  price?: number;
  currency?: string;
  
  profitMargin?: number;
  roe?: number;
  dividendYield?: number;
  
  debtToEquity?: number;
  currentRatio?: number;
  freeCashflow?: number;
  operatingCashflow?: number;
  
  forwardPE?: number;
  staticPE?: number;
  pegRatio?: number;
  priceToBook?: number;
  bookValue?: number;
  evToEbitda?: number;
  fcfYield?: number;
  
  targetMeanPrice?: number;
  recommendationKey?: string;
  
  inputs: {
    revenue: number;
    ebitda: number;
    netIncome: number;
    fcf: number;
    netDebt: number;
    sharesOutstanding: number;
    beta: number;
    riskFreeRate: number;
    marketRiskPremium: number;
  };
  dcf?: {
    wacc: number;
    growthRate: number;
    terminalGrowthRate: number;
    projectedFCF: number[];
    projectedDiscountedFCF?: number[];
    terminalValue: number;
    terminalValueExitMultiple?: number;
    presentValueSum: number;
    equityValue: number;
    equityValueExitMultiple?: number;
    sharePrice: number;
    sharePriceGordon?: number;
    sharePriceExitMultiple?: number;
    upside: number;
  };
  multiples?: {
    eps: number;
    peScenarios: Record<string, number>;
    ebitda: number;
    evEbitdaScenarios: Record<string, number>;
    currentPE?: number;
    currentEvEbitda?: number;
  };
  specialAnalysis?: {
    title: string;
    description: string;
    models: { name: string; value: number; details: string }[];
    blendedValue: number;
  };
  ddm?: {
    dividendYield: number;
    dividendGrowthRate: number;
    costOfEquity: number;
    fairValue: number;
    modelType: string;
  };
  reit?: {
    ffo: number;
    ffoPerShare: number;
    priceToFFO: number;
    fairValue: number;
    sectorAveragePFFO: number;
    modelType: string;
  };
  rim?: {
    bookValuePerShare: number;
    roe: number;
    costOfEquity: number;
    fairValue: number;
    modelType: string;
  };
  quant?: {
    quality: number;
    value: number;
    growth: number;
    momentum: number;
    total: number;
    details: Record<string, string>;
  };
  risk?: {
    beta: number;
    volatility: number;
    sharpeRatio: number;
    maxDrawdown: number;
  };
  sensitivity?: {
    growthRates: number[];
    rows: { wacc: number; prices: number[] }[];
  };
  rating: string;
  fairValueRange: number[];
}

interface FundamentalResponse {
  data: FundamentalData;
  analysis: ValueAnalysis;
}

const FundamentalAnalysis: React.FC = () => {
  const { ticker: paramTicker } = useParams<{ ticker: string }>();
  const [ticker, setTicker] = useState(paramTicker || "");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FundamentalResponse | null>(null);
  const [valuation, setValuation] = useState<ValuationResult | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (paramTicker) {
      setTicker(paramTicker);
      handleSearch(undefined, paramTicker);
    }
  }, [paramTicker]);

  const handleSearch = async (e?: React.FormEvent, overrideTicker?: string) => {
    if (e) e.preventDefault();
    const searchTicker = overrideTicker || ticker;
    if (!searchTicker) return;

    setLoading(true);
    setError("");
    setResult(null);
    setValuation(null);

    try {
      const res = await api.get(API_ENDPOINTS.FUNDAMENTAL(searchTicker));
      const backendData = res.data as ValuationResult;
      
      // Map Backend Data to Frontend Structure
      // Note: Backend 'ValuationResult' currently lacks some fundamental fields like name, sector, price directly at top level
      // We might need to fetch them or assume they are in inputs/multiples
      // Wait, the backend schema for ValuationResult DOES NOT have 'name', 'sector', 'price'.
      // It has 'ticker', 'inputs', 'dcf', etc.
      // We need to update backend schema to include these or fetch them separately.
      // BUT, checking schemas.py, ValuationResult ONLY has ticker.
      // HOWEVER, the code in valuation.py returns `valuation_service.analyze_stock(ticker)`.
      // Let's check `backend/app/engines/valuation/core.py` or `service.py` to see what `analyze_stock` returns.
      // It likely returns a dict that matches ValuationResult.
      
      // CRITICAL: The backend ValuationResult schema is missing basic info (name, price, sector).
      // We should update the backend schema to include these, or the frontend will break.
      // Let's assume for now we map what we can, but we really need that info.
      
      // Temporary mapping to avoid crash, but data will be missing
      const mappedResult: FundamentalResponse = {
          data: {
              ticker: backendData.ticker,
              name: backendData.name || backendData.ticker,
              sector: backendData.sector || "Unknown",
              industry: backendData.industry || "Unknown",
              currency: backendData.currency || "USD",
              price: backendData.price || 0,
              peRatio: backendData.multiples?.currentPE,
              forwardPE: backendData.forwardPE,
              pegRatio: backendData.pegRatio,
              pbRatio: backendData.priceToBook,
              dividendYield: backendData.dividendYield,
              roe: backendData.roe,
              profitMargin: backendData.profitMargin,
              eps: backendData.multiples?.eps,
              revenueGrowth: backendData.dcf?.growthRate,
              debtToEquity: backendData.debtToEquity,
              currentRatio: backendData.currentRatio,
              freeCashflow: backendData.freeCashflow,
              fcfYield: backendData.fcfYield,
              evToEbitda: backendData.evToEbitda,
              targetMeanPrice: backendData.targetMeanPrice,
              recommendationKey: backendData.recommendationKey,
          },
          analysis: {
              grahamNumber: (backendData.multiples?.eps && backendData.bookValue) 
                  ? Math.sqrt(22.5 * backendData.multiples.eps * backendData.bookValue) 
                  : 0,
              isUndervalued: false,
              details: backendData.quant?.details || {}
          }
      };

      setResult(mappedResult);
      setValuation(backendData); // Store full backend result for advanced view
    } catch (err) {
      console.error(err);
      setError("Failed to fetch data. Please check the ticker and try again.");
    } finally {
      setLoading(false);
    }
  };



  const formatNumber = (num?: number, decimals = 2, suffix = "") => {
    if (num === undefined || num === null) return "N/A";
    return num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
  };

  const formatPercent = (num?: number) => {
    if (num === undefined || num === null) return "N/A";
    return (num * 100).toFixed(2) + "%";
  };

  const formatRawPercent = (num?: number) => {
    if (num === undefined || num === null) return "N/A";
    return num.toFixed(2) + "%";
  };

  const getScoreColor = (val: number, threshold: number, inverse = false) => {
    if (val === undefined || val === null) return "text-muted-foreground";
    const good = inverse ? val < threshold : val > threshold;
    return good ? "text-success" : "text-warning"; // Using custom colors if available, or fallback
  };

  return (
    <div className="space-y-6 animate-fade-in max-w-7xl mx-auto">
      {/* Search Section */}
      <Card className="material-card p-6">
        <CardContent className="p-0">
          <form onSubmit={handleSearch} className="flex gap-4 items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground h-5 w-5" />
              <Input 
                placeholder="Enter stock ticker (e.g. AAPL, MSFT, KO)..." 
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                className="pl-10 h-12 text-lg bg-secondary/30 border-none"
              />
            </div>
            <Button type="submit" size="lg" disabled={loading} className="h-12 px-8 rounded-xl font-semibold">
              {loading ? "Analyzing..." : "Analyze"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <div className="p-4 rounded-xl bg-destructive/10 text-destructive flex items-center gap-2">
          <AlertCircle className="h-5 w-5" />
          {error}
        </div>
      )}

      {result && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Header Info */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
              <h2 className="text-3xl font-bold tracking-tight">{result.data.name} <span className="text-muted-foreground text-xl font-medium">({result.data.ticker})</span></h2>
              <div className="flex gap-3 mt-2 text-sm text-muted-foreground">
                <Badge variant="outline" className="bg-secondary/50">{result.data.sector}</Badge>
                <Badge variant="outline" className="bg-secondary/50">{result.data.industry}</Badge>
                <span className="flex items-center gap-1"><DollarSign className="h-3 w-3" /> {result.data.currency}</span>
              </div>
            </div>
            <div className="flex items-center gap-4">
               <div className="text-right">
                  <div className="text-4xl font-bold">${formatNumber(result.data.price)}</div>
                  {result.data.targetMeanPrice && (
                    <div className="text-sm text-muted-foreground mt-1">
                      Target: ${formatNumber(result.data.targetMeanPrice)} 
                      <span className={`ml-2 font-medium ${result.data.price < result.data.targetMeanPrice ? 'text-success' : 'text-destructive'}`}>
                        ({((result.data.targetMeanPrice - result.data.price) / result.data.price * 100).toFixed(1)}%)
                      </span>
                    </div>
                  )}
               </div>
            </div>
          </div>

          {/* Basic View */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Valuation */}
                <Card className="material-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <Scale className="h-5 w-5 text-primary" /> Valuation
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">

                    <div className="flex justify-between items-center group relative">
                      <span className="text-muted-foreground border-b border-dotted border-muted-foreground/50 cursor-help" title="Trailing 12 Months: Uses past 12 months (4 quarters) of real earnings. Most current actual data.">
                        P/E (TTM)
                      </span>
                      <span className={`font-mono font-bold ${getScoreColor(result.data.peRatio || 0, 25, true)}`}>
                        {formatNumber(result.data.peRatio)}
                      </span>
                    </div>

                    {/* P/E Static */}
                    <div className="flex justify-between items-center group relative">
                      <span className="text-muted-foreground border-b border-dotted border-muted-foreground/50 cursor-help" title="Static P/E: Uses last full fiscal year's earnings. Most audited/certain data, but may be outdated.">
                        P/E (Static)
                      </span>
                      <span className="font-mono font-bold">
                        {formatNumber(result.data.staticPE)}
                      </span>
                    </div>

                    {/* Forward P/E */}
                    <div className="flex justify-between items-center group relative">
                      <span className="text-muted-foreground border-b border-dotted border-muted-foreground/50 cursor-help" title="Forward P/E: Uses analyst estimates for next year. Key for growth stocks to see future expectations.">
                        P/E (Forward)
                      </span>
                      <span className={`font-mono font-bold ${getScoreColor(result.data.forwardPE || 0, 25, true)}`}>
                        {formatNumber(result.data.forwardPE)}
                      </span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">PEG Ratio</span>
                      <span className={`font-mono font-bold ${getScoreColor(result.data.pegRatio || 0, 1, true)}`}>
                        {formatNumber(result.data.pegRatio)}
                      </span>
                    </div>

                    {/* P/B Ratio */}
                    <div className="flex justify-between items-center group relative">
                      <span className="text-muted-foreground border-b border-dotted border-muted-foreground/50 cursor-help" title="Price-to-Book: Price paid for $1 of net assets. Relevant for banks/heavy industry, less for tech.">
                        P/B Ratio
                      </span>
                      <span className="font-mono font-bold">{formatNumber(result.data.pbRatio)}</span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">EV/EBITDA</span>
                      <span className="font-mono font-bold">{formatNumber(result.data.evToEbitda)}</span>
                    </div>
                    <div className="flex justify-between items-center pt-2 border-t border-border/50">
                      <span className="text-muted-foreground">Dividend Yield</span>
                      <span className="font-mono font-bold text-success">
                        {formatRawPercent(result.data.dividendYield)}
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {/* Profitability */}
                <Card className="material-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <TrendingUp className="h-5 w-5 text-success" /> Profitability
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">ROE</span>
                      <span className={`font-mono font-bold ${getScoreColor(result.data.roe || 0, 0.15)}`}>
                        {formatPercent(result.data.roe)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">Profit Margin</span>
                      <span className={`font-mono font-bold ${getScoreColor(result.data.profitMargin || 0, 0.10)}`}>
                        {formatPercent(result.data.profitMargin)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">EPS (TTM)</span>
                      <span className="font-mono font-bold">${formatNumber(result.data.eps)}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">Revenue Growth</span>
                      <span className={`font-mono font-bold ${getScoreColor(result.data.revenueGrowth || 0, 0.05)}`}>
                        {formatPercent(result.data.revenueGrowth)}
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {/* Financial Health */}
                <Card className="material-card">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg">
                      <Activity className="h-5 w-5 text-blue-500" /> Health
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">Debt/Equity</span>
                      <span className={`font-mono font-bold ${getScoreColor(result.data.debtToEquity || 0, 100, true)}`}>
                        {formatNumber(result.data.debtToEquity)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">Current Ratio</span>
                      <span className={`font-mono font-bold ${getScoreColor(result.data.currentRatio || 0, 1.5)}`}>
                        {formatNumber(result.data.currentRatio)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">Free Cash Flow</span>
                      <span className="font-mono font-bold truncate max-w-[120px]" title={result.data.freeCashflow?.toString()}>
                        ${result.data.freeCashflow ? (result.data.freeCashflow / 1e9).toFixed(2) + "B" : "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">FCF Yield</span>
                      <span className={`font-mono font-bold ${getScoreColor(result.data.fcfYield || 0, 0.04)}`}>
                        {formatPercent(result.data.fcfYield)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center pt-2 border-t border-border/50">
                      <span className="text-muted-foreground">Recommendation</span>
                      <Badge variant="secondary" className="uppercase font-bold">
                        {result.data.recommendationKey?.replace(/_/g, " ") || "N/A"}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
                
                {/* Value Analysis */}
                <Card className="material-card bg-primary/5 border-primary/20 md:col-span-3">
                    <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <TrendingUp className="h-6 w-6 text-primary" /> Smart Valuation Check
                    </CardTitle>
                    <CardDescription>
                        Automated checks based on modern growth and value investing criteria.
                    </CardDescription>
                    </CardHeader>
                    <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-4">
                        <h3 className="font-semibold text-lg">Intrinsic Value Estimate</h3>
                        <div className="flex items-center justify-between p-4 bg-background/50 rounded-xl border border-border/50">
                            <span className="text-muted-foreground">Graham Number</span>
                            <div className="text-right">
                            <div className="text-2xl font-bold font-mono">${formatNumber(result.analysis.grahamNumber)}</div>
                            <div className="text-xs text-muted-foreground">√(22.5 × EPS × Book Value)</div>
                            </div>
                        </div>
                        
                        <div className="flex items-center gap-3">
                            <div className={`flex-1 h-2 rounded-full ${result.analysis.isUndervalued || (result.data.pegRatio && result.data.pegRatio < 2.0) ? 'bg-success' : 'bg-warning'}`} />
                            <span className={`font-bold ${result.analysis.isUndervalued || (result.data.pegRatio && result.data.pegRatio < 2.0) ? 'text-success' : 'text-warning'}`}>
                            {result.analysis.isUndervalued || (result.data.pegRatio && result.data.pegRatio < 2.0) ? "Fair / Undervalued" : "Premium Valuation"}
                            </span>
                        </div>
                        </div>

                        <div className="space-y-2">
                        <h3 className="font-semibold text-lg">Checklist</h3>
                        <div className="space-y-2">
                            {Object.entries(result.analysis.details).map(([key, msg]) => (
                            <div key={key} className="flex items-center gap-2 text-sm text-success font-medium bg-success/10 p-2 rounded-lg">
                                <CheckCircle2 className="h-4 w-4" />
                                {msg}
                            </div>
                            ))}
                            
                            {/* Forward P/E Check (< 25 for Growth) */}
                            <div className="flex items-center gap-2 text-sm p-2">
                            {result.data.forwardPE && result.data.forwardPE < 25 ? (
                                <CheckCircle2 className="h-4 w-4 text-success" />
                            ) : (
                                <AlertCircle className="h-4 w-4 text-muted-foreground" />
                            )}
                            <span className={result.data.forwardPE && result.data.forwardPE < 25 ? "text-foreground" : "text-muted-foreground"}>
                                Forward P/E &lt; 25 (Growth)
                            </span>
                            </div>

                            {/* PEG Ratio Check (< 2.0) */}
                            <div className="flex items-center gap-2 text-sm p-2">
                            {result.data.pegRatio && result.data.pegRatio < 2.0 ? (
                                <CheckCircle2 className="h-4 w-4 text-success" />
                            ) : (
                                <AlertCircle className="h-4 w-4 text-muted-foreground" />
                            )}
                            <span className={result.data.pegRatio && result.data.pegRatio < 2.0 ? "text-foreground" : "text-muted-foreground"}>
                                PEG Ratio &lt; 2.0
                            </span>
                            </div>

                            {/* ROE Check (> 15%) */}
                            <div className="flex items-center gap-2 text-sm p-2">
                            {result.data.roe && result.data.roe > 0.15 ? (
                                <CheckCircle2 className="h-4 w-4 text-success" />
                            ) : (
                                <AlertCircle className="h-4 w-4 text-muted-foreground" />
                            )}
                            <span className={result.data.roe && result.data.roe > 0.15 ? "text-foreground" : "text-muted-foreground"}>
                                ROE &gt; 15%
                            </span>
                            </div>
                        </div>
                        </div>
                    </div>
                    </CardContent>
                </Card>
              </div>
              
              {/* Advanced Valuation Section */}
              <div className="pt-8 border-t border-border/50">
                  <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                      <Scale className="h-6 w-6 text-primary" /> Advanced Valuation Model
                  </h3>
                  <div className="space-y-6 animate-fade-in">
                 {valuation && (
                    <>
                       <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          {/* Inputs Card */}
                          <Card className="material-card">
                             <CardHeader><CardTitle>Valuation Inputs</CardTitle></CardHeader>
                             <CardContent className="space-y-2 text-sm">
                                <div className="flex justify-between"><span>Revenue</span><span className="font-mono">${(valuation.inputs.revenue/1e9).toFixed(2)}B</span></div>
                                <div className="flex justify-between"><span>EBITDA</span><span className="font-mono">${(valuation.inputs.ebitda/1e9).toFixed(2)}B</span></div>
                                <div className="flex justify-between"><span>FCF</span><span className="font-mono">${(valuation.inputs.fcf/1e9).toFixed(2)}B</span></div>
                                <div className="flex justify-between"><span>Net Debt</span><span className="font-mono">${(valuation.inputs.netDebt/1e9).toFixed(2)}B</span></div>
                                <div className="flex justify-between"><span>Beta</span><span className="font-mono">{valuation.inputs.beta.toFixed(2)}</span></div>
                                {valuation.dcf && (
                                    <>
                                        <div className="flex justify-between"><span>WACC</span><span className="font-mono">{(valuation.dcf.wacc*100).toFixed(2)}%</span></div>
                                        <div className="flex justify-between"><span>Growth Rate</span><span className="font-mono">{(valuation.dcf.growthRate*100).toFixed(2)}%</span></div>
                                    </>
                                )}
                             </CardContent>
                          </Card>

                          {/* Special Analysis Card */}
                          {valuation.specialAnalysis && (
                            <Card className="material-card border-primary/50 bg-primary/5">
                                <CardHeader>
                                    <div className="flex items-center gap-2">
                                        <Sparkles className="h-5 w-5 text-primary" />
                                        <CardTitle>{valuation.specialAnalysis.title}</CardTitle>
                                    </div>
                                    <CardDescription>{valuation.specialAnalysis.description}</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="grid grid-cols-1 gap-4 mb-4">
                                        {valuation.specialAnalysis.models.map((model, idx) => (
                                            <div key={idx} className="p-3 bg-background/50 rounded-lg border border-border">
                                                <div className="flex justify-between items-center mb-1">
                                                    <div className="text-sm font-medium text-muted-foreground">{model.name}</div>
                                                    <div className="text-lg font-bold text-primary">${model.value.toFixed(2)}</div>
                                                </div>
                                                <div className="text-xs text-muted-foreground">{model.details}</div>
                                            </div>
                                        ))}
                                    </div>
                                    <div className="flex items-center justify-between p-3 bg-primary/10 rounded-lg border border-primary/20">
                                        <div className="font-semibold text-primary">Blended Special Valuation</div>
                                        <div className="text-2xl font-bold text-primary">${valuation.specialAnalysis.blendedValue.toFixed(2)}</div>
                                    </div>
                                </CardContent>
                            </Card>
                          )}
                       </div>

                       {/* Valuation Cards (Conditional based on Sector) */}
                       
                       {/* 1. DCF Card (Standard) */}
                       {valuation.dcf && (
                         <Card className="material-card md:col-span-3">
                             <CardHeader>
                                 <div className="flex items-center justify-between">
                                     <div>
                                         <CardTitle>Discounted Cash Flow (DCF)</CardTitle>
                                         <CardDescription>Intrinsic value based on future cash flows</CardDescription>
                                     </div>
                                     <Badge variant={valuation.dcf.upside > 0 ? "default" : "destructive"} className="text-lg px-3 py-1">
                                         {valuation.dcf.upside > 0 ? "+" : ""}{(valuation.dcf.upside * 100).toFixed(2)}% Upside
                                     </Badge>
                                 </div>
                             </CardHeader>
                             <CardContent>
                                 <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                     <div className="space-y-2">
                                         <div className="text-sm text-muted-foreground">Estimated Fair Value (Blended)</div>
                                         <div className="text-2xl font-semibold">${valuation.dcf.sharePrice.toFixed(2)}</div>
                                         <div className="text-xs text-muted-foreground mt-1">
                                             Avg of Gordon Growth (${valuation.dcf.sharePriceGordon?.toFixed(2)}) & Exit Multiple (${valuation.dcf.sharePriceExitMultiple?.toFixed(2)})
                                         </div>
                                     </div>
                                     <div className="space-y-2">
                                         <div className="text-sm text-muted-foreground">Gordon Growth Method</div>
                                         <div className="text-2xl font-semibold">${valuation.dcf.sharePriceGordon?.toFixed(2)}</div>
                                         <div className="text-xs text-muted-foreground">Terminal Growth: {(valuation.dcf.terminalGrowthRate * 100).toFixed(1)}%</div>
                                     </div>
                                     <div className="space-y-2">
                                         <div className="text-sm text-muted-foreground">Exit Multiple Method</div>
                                         <div className="text-2xl font-semibold">${valuation.dcf.sharePriceExitMultiple?.toFixed(2)}</div>
                                         <div className="text-xs text-muted-foreground">Terminal Value (PV): ${(valuation.dcf.terminalValueExitMultiple! / 1e9).toFixed(2)}B</div>
                                     </div>
                                 </div>
                             </CardContent>
                         </Card>
                       )}

                       {/* 2. DDM Card (Financials) */}
                       {valuation.ddm && (
                         <Card className="material-card md:col-span-3 border-blue-500/20 bg-blue-500/5">
                             <CardHeader>
                                 <div className="flex items-center gap-2">
                                     <DollarSign className="h-5 w-5 text-blue-500" />
                                     <div>
                                         <CardTitle>Dividend Discount Model (DDM)</CardTitle>
                                         <CardDescription>Valuation for Financials (Banks/Insurance)</CardDescription>
                                     </div>
                                 </div>
                             </CardHeader>
                             <CardContent>
                                 <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                     <div className="space-y-2">
                                         <div className="text-sm text-muted-foreground">Fair Value (DDM)</div>
                                         <div className="text-4xl font-bold text-blue-500">${valuation.ddm.fairValue.toFixed(2)}</div>
                                         <div className="text-xs text-muted-foreground">Based on Dividends & Excess Returns</div>
                                     </div>
                                     <div className="space-y-2">
                                         <div className="text-sm text-muted-foreground">Key Assumptions</div>
                                         <div className="grid grid-cols-2 gap-2 text-sm">
                                             <div className="text-muted-foreground">Cost of Equity:</div>
                                             <div className="font-mono">{(valuation.ddm.costOfEquity * 100).toFixed(2)}%</div>
                                             <div className="text-muted-foreground">Div. Growth:</div>
                                             <div className="font-mono">{(valuation.ddm.dividendGrowthRate * 100).toFixed(2)}%</div>
                                         </div>
                                     </div>
                                     <div className="space-y-2">
                                         <div className="text-sm text-muted-foreground">Dividend Yield</div>
                                         <div className="text-2xl font-semibold">{(valuation.ddm.dividendYield * 100).toFixed(2)}%</div>
                                         <div className="text-xs text-muted-foreground">Model: {valuation.ddm.modelType}</div>
                                     </div>
                                 </div>
                             </CardContent>
                         </Card>
                       )}

                       {/* 3. REIT Card (Real Estate) */}
                       {valuation.reit && (
                         <Card className="material-card md:col-span-3 border-green-500/20 bg-green-500/5">
                             <CardHeader>
                                 <div className="flex items-center gap-2">
                                     <Activity className="h-5 w-5 text-green-500" />
                                     <div>
                                         <CardTitle>REIT Valuation (FFO Model)</CardTitle>
                                         <CardDescription>Funds From Operations Analysis</CardDescription>
                                     </div>
                                 </div>
                             </CardHeader>
                             <CardContent>
                                 <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                     <div className="space-y-2">
                                         <div className="text-sm text-muted-foreground">Fair Value (P/FFO)</div>
                                         <div className="text-4xl font-bold text-green-500">${valuation.reit.fairValue.toFixed(2)}</div>
                                         <div className="text-xs text-muted-foreground">Target P/FFO: {valuation.reit.sectorAveragePFFO}x</div>
                                     </div>
                                     <div className="space-y-2">
                                         <div className="text-sm text-muted-foreground">FFO Metrics</div>
                                         <div className="grid grid-cols-2 gap-2 text-sm">
                                             <div className="text-muted-foreground">FFO/Share:</div>
                                             <div className="font-mono">${valuation.reit.ffoPerShare.toFixed(2)}</div>
                                             <div className="text-muted-foreground">Price/FFO:</div>
                                             <div className="font-mono">{valuation.reit.priceToFFO.toFixed(2)}x</div>
                                         </div>
                                     </div>
                                     <div className="space-y-2">
                                         <div className="text-sm text-muted-foreground">Total FFO</div>
                                         <div className="text-2xl font-semibold">${(valuation.reit.ffo / 1e9).toFixed(2)}B</div>
                                         <div className="text-xs text-muted-foreground">Net Income + Depreciation</div>
                                     </div>
                                 </div>
                             </CardContent>
                         </Card>
                       )}

                       {/* Multiples Valuation Card */}
                       {valuation.multiples && (
                         <Card className="material-card md:col-span-3">
                             <CardHeader><CardTitle>Relative Valuation (Multiples)</CardTitle></CardHeader>
                             <CardContent>
                                 <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                     {/* P/E Scenarios */}
                                     <div className="space-y-4">
                                         <h4 className="font-semibold text-primary border-b border-border pb-2">Implied Price based on P/E Ratio</h4>
                                         <div className="text-sm text-muted-foreground mb-2">Current EPS: <span className="font-mono text-foreground">${valuation.multiples.eps.toFixed(2)}</span> | Current P/E: <span className="font-mono text-foreground">{valuation.multiples.currentPE?.toFixed(1) || 'N/A'}x</span></div>
                                         <div className="grid grid-cols-4 gap-2 text-center">
                                             {Object.entries(valuation.multiples.peScenarios).map(([mult, price]) => (
                                                 <div key={mult} className="p-3 bg-secondary/30 rounded-lg border border-border/50">
                                                     <div className="text-xs text-muted-foreground mb-1">{mult} P/E</div>
                                                     <div className="font-bold font-mono text-primary">${price.toFixed(2)}</div>
                                                 </div>
                                             ))}
                                         </div>
                                     </div>
                                     
                                     {/* EV/EBITDA Scenarios */}
                                     <div className="space-y-4">
                                         <h4 className="font-semibold text-primary border-b border-border pb-2">Implied Price based on EV/EBITDA</h4>
                                         <div className="text-sm text-muted-foreground space-y-1">
                                             <div>Current EBITDA: <span className="font-mono text-foreground">${(valuation.multiples.ebitda/1e9).toFixed(2)}B</span></div>
                                             <div>Net Debt: <span className="font-mono text-foreground">${(valuation.inputs.netDebt/1e9).toFixed(2)}B</span></div>
                                             <div className="text-xs italic mt-1">Formula: (EBITDA × Multiple - Net Debt) / Shares</div>
                                         </div>
                                         <div className="grid grid-cols-4 gap-2 text-center mt-2">
                                             {Object.entries(valuation.multiples.evEbitdaScenarios).map(([mult, price]) => (
                                                 <div key={mult} className="p-3 bg-secondary/30 rounded-lg border border-border/50">
                                                     <div className="text-xs text-muted-foreground mb-1">{mult}</div>
                                                     <div className="font-bold font-mono text-primary">${price.toFixed(2)}</div>
                                                 </div>
                                             ))}
                                         </div>
                                     </div>
                                 </div>
                             </CardContent>
                         </Card>
                       )}

                       {/* Projection Table Card (Only for DCF) */}
                       {valuation.dcf && (
                           <Card className="material-card md:col-span-3">
                              <CardHeader><CardTitle>Annual Cash Flow Projections</CardTitle></CardHeader>
                              <CardContent className="overflow-x-auto">
                                 <table className="w-full text-sm text-left">
                                    <thead className="text-muted-foreground border-b border-border">
                                       <tr>
                                          <th className="py-2 font-medium">Year</th>
                                          <th className="py-2 font-medium text-right">Projected FCF</th>
                                          <th className="py-2 font-medium text-right">Discount Factor</th>
                                          <th className="py-2 font-medium text-right">Discounted FCF (PV)</th>
                                       </tr>
                                    </thead>
                                    <tbody>
                                       {valuation.dcf.projectedFCF.map((fcf, i) => {
                                          const discountFactor = 1 / Math.pow(1 + valuation.dcf!.wacc, i + 1);
                                          const discountedFCF = valuation.dcf!.projectedDiscountedFCF ? valuation.dcf!.projectedDiscountedFCF[i] : fcf * discountFactor;
                                          return (
                                             <tr key={i} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                                                <td className="py-2">Year {i + 1}</td>
                                                <td className="py-2 text-right font-mono">${(fcf / 1e9).toFixed(2)}B</td>
                                                <td className="py-2 text-right font-mono">{discountFactor.toFixed(4)}</td>
                                                <td className="py-2 text-right font-mono text-primary">${(discountedFCF / 1e9).toFixed(2)}B</td>
                                             </tr>
                                          );
                                       })}
                                       {/* Terminal Value Row */}
                                       <tr className="bg-muted/20 font-medium">
                                          <td className="py-2 pt-4">Terminal Value</td>
                                          <td className="py-2 pt-4 text-right font-mono">${(valuation.dcf.terminalValue / 1e9).toFixed(2)}B</td>
                                          <td className="py-2 pt-4 text-right font-mono">{(1 / Math.pow(1 + valuation.dcf.wacc, valuation.dcf.projectedFCF.length)).toFixed(4)}</td>
                                          <td className="py-2 pt-4 text-right font-mono text-primary">${(valuation.dcf.terminalValue / Math.pow(1 + valuation.dcf.wacc, valuation.dcf.projectedFCF.length) / 1e9).toFixed(2)}B</td>
                                       </tr>
                                    </tbody>
                                 </table>
                              </CardContent>
                           </Card>
                       )}

                       {/* Quant & Risk Analysis (Citadel-Tier) */}
                       {valuation.quant && (
                           <Card className="material-card md:col-span-3 border-purple-500/20 bg-purple-500/5">
                               <CardHeader>
                                   <div className="flex items-center gap-2">
                                       <Activity className="h-5 w-5 text-purple-500" />
                                       <div>
                                           <CardTitle>Quant Factor Model</CardTitle>
                                           <CardDescription>Multi-Factor Scoring (0-100)</CardDescription>
                                       </div>
                                       <Badge variant="outline" className="ml-auto text-purple-500 border-purple-500/50">
                                           Total Score: {valuation.quant.total.toFixed(0)}/100
                                       </Badge>
                                   </div>
                               </CardHeader>
                               <CardContent>
                                   <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                       {Object.entries(valuation.quant).filter(([k]) => k !== 'total' && k !== 'details').map(([key, score]) => (
                                           <div key={key} className="space-y-2">
                                               <div className="flex justify-between text-sm">
                                                   <span className="capitalize text-muted-foreground">{key}</span>
                                                   <span className="font-mono font-bold">{Number(score).toFixed(0)}</span>
                                               </div>
                                               <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                                                   <div 
                                                       className={`h-full rounded-full ${Number(score) > 70 ? 'bg-green-500' : Number(score) > 40 ? 'bg-yellow-500' : 'bg-red-500'}`} 
                                                       style={{ width: `${score}%` }}
                                                   />
                                               </div>
                                               <div className="text-[10px] text-muted-foreground truncate">
                                                   {valuation.quant?.details?.[key.charAt(0).toUpperCase() + key.slice(1)]}
                                               </div>
                                           </div>
                                       ))}
                                   </div>
                               </CardContent>
                           </Card>
                       )}

                       {/* Residual Income Model (RIM) */}
                       {valuation.rim && (
                           <Card className="material-card md:col-span-3 border-indigo-500/20 bg-indigo-500/5">
                               <CardHeader>
                                   <div className="flex items-center gap-2">
                                       <DollarSign className="h-5 w-5 text-indigo-500" />
                                       <div>
                                           <CardTitle>Residual Income Model (RIM)</CardTitle>
                                           <CardDescription>Valuation based on Excess Returns over Book Value</CardDescription>
                                       </div>
                                   </div>
                               </CardHeader>
                               <CardContent>
                                   <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                       <div className="space-y-2">
                                           <div className="text-sm text-muted-foreground">Fair Value (RIM)</div>
                                           <div className="text-4xl font-bold text-indigo-500">${valuation.rim.fairValue.toFixed(2)}</div>
                                           <div className="text-xs text-muted-foreground">Book Value + PV of Residual Income</div>
                                       </div>
                                       <div className="space-y-2">
                                           <div className="text-sm text-muted-foreground">Key Metrics</div>
                                           <div className="grid grid-cols-2 gap-2 text-sm">
                                               <div className="text-muted-foreground">ROE:</div>
                                               <div className="font-mono">{(valuation.rim.roe * 100).toFixed(2)}%</div>
                                               <div className="text-muted-foreground">Cost of Equity:</div>
                                               <div className="font-mono">{(valuation.rim.costOfEquity * 100).toFixed(2)}%</div>
                                           </div>
                                       </div>
                                       <div className="space-y-2">
                                           <div className="text-sm text-muted-foreground">Book Value</div>
                                           <div className="text-2xl font-semibold">${valuation.rim.bookValuePerShare.toFixed(2)}</div>
                                           <div className="text-xs text-muted-foreground">Per Share</div>
                                       </div>
                                   </div>
                               </CardContent>
                           </Card>
                       )}

                       {/* Risk Profile */}
                       {valuation.risk && (
                           <Card className="material-card md:col-span-3">
                               <CardHeader><CardTitle>Risk Profile</CardTitle></CardHeader>
                               <CardContent>
                                   <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                                       <div className="p-3 bg-secondary/30 rounded-lg border border-border/50">
                                           <div className="text-xs text-muted-foreground mb-1">Beta</div>
                                           <div className="font-bold font-mono text-lg">{valuation.risk.beta.toFixed(2)}</div>
                                       </div>
                                       <div className="p-3 bg-secondary/30 rounded-lg border border-border/50">
                                           <div className="text-xs text-muted-foreground mb-1">Volatility (Ann.)</div>
                                           <div className="font-bold font-mono text-lg">{(valuation.risk.volatility * 100).toFixed(1)}%</div>
                                       </div>
                                       <div className="p-3 bg-secondary/30 rounded-lg border border-border/50">
                                           <div className="text-xs text-muted-foreground mb-1">Sharpe Ratio</div>
                                           <div className={`font-bold font-mono text-lg ${valuation.risk.sharpeRatio > 1 ? 'text-green-500' : 'text-foreground'}`}>
                                               {valuation.risk.sharpeRatio.toFixed(2)}
                                           </div>
                                       </div>
                                       <div className="p-3 bg-secondary/30 rounded-lg border border-border/50">
                                           <div className="text-xs text-muted-foreground mb-1">Max Drawdown (1Y)</div>
                                           <div className="font-bold font-mono text-lg text-red-500">{(valuation.risk.maxDrawdown * 100).toFixed(1)}%</div>
                                       </div>
                                   </div>
                               </CardContent>
                           </Card>
                       )}

                       {/* Sensitivity Analysis (Only if available) */}
                       {valuation.sensitivity && (
                           <Card className="material-card">
                              <CardHeader><CardTitle>Sensitivity Analysis (Share Price)</CardTitle></CardHeader>
                              <CardContent>
                                 <div className="overflow-x-auto">
                                    <table className="w-full text-sm text-center">
                                       <thead>
                                          <tr>
                                             <th className="p-2 text-left">WACC \ Growth</th>
                                            {valuation.sensitivity.growthRates.map(g => (
                                               <th key={g} className="p-2">{(g*100).toFixed(1)}%</th>
                                            ))}
                                         </tr>
                                      </thead>
                                      <tbody>
                                         {valuation.sensitivity.rows.map(row => (
                                            <tr key={row.wacc}>
                                               <td className="p-2 text-left font-medium">{(row.wacc*100).toFixed(1)}%</td>
                                               {row.prices.map(p => (
                                                  <td key={p} className={`p-2 font-mono ${p > result.data.price ? 'text-success bg-success/10' : 'text-muted-foreground'}`}>
                                                     ${p.toFixed(2)}
                                                  </td>
                                               ))}
                                            </tr>
                                         ))}
                                      </tbody>
                                   </table>
                                </div>
                             </CardContent>
                          </Card>
                       )}
                    </>
                 )}
              </div>
            </div>
        </motion.div>
      )}
    </div>
  );
};

export default FundamentalAnalysis;
