import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { 
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
    ResponsiveContainer, Legend, Area, AreaChart, ReferenceLine,
    BarChart, Bar
} from 'recharts';
import { motion } from 'framer-motion';
import { 
    FlaskConical, TrendingUp, TrendingDown, Play, Loader2, 
    ChevronDown, ChevronUp, Plus, X, BarChart3, RefreshCw, DollarSign,
    Shield, AlertTriangle, CheckCircle, Target
} from 'lucide-react';

interface BacktestMetrics {
    total_return: number;
    annual_return: number;
    volatility: number;
    sharpe_ratio: number;
    sharpe_ratio_gross?: number;
    max_drawdown: number;
    avg_turnover?: number;
    total_transaction_costs?: number;
    total_stop_losses?: number;
}

interface CurvePoint { date: string; value: number; }
interface DrawdownPoint { date: string; drawdown: number; }
interface TurnoverPoint { date: string; turnover: number; }
interface RollingSharpePoint { date: string; rolling_sharpe: number; }
interface MonthlyReturn { year: number; month: number; return: number; }
interface SubperiodResult { year: number; return: number; volatility: number; sharpe: number; max_drawdown: number; }

interface BacktestResult {
    strategy: {
        name: string;
        curve: CurvePoint[];
        curve_gross?: CurvePoint[];
        metrics: BacktestMetrics;
    };
    benchmarks: { [ticker: string]: { curve: CurvePoint[]; metrics: BacktestMetrics; }; };
    drawdown: DrawdownPoint[];
    turnover?: TurnoverPoint[];
    rolling_sharpe?: RollingSharpePoint[];
    monthly_returns?: MonthlyReturn[];
    robustness?: {
        bootstrap_sharpe: { mean: number; lower_95: number; upper_95: number; std: number; };
        subperiod: SubperiodResult[];
    };
    risk_controls?: { max_position: string; sector_cap: string; stop_loss: string; min_holding: string; };
    period: string;
    transaction_cost_bps?: number;
    generated_at: string;
}

const COLORS = {
    strategy: '#22c55e',
    strategyGross: '#86efac',
    benchmark: ['#3b82f6', '#a855f7', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899']
};

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// Risk Controls Badge Component
const RiskControlBadge: React.FC<{ label: string; value: string }> = ({ label, value }) => (
    <div className="flex items-center gap-2 bg-muted/50 px-3 py-1.5 rounded-full text-xs">
        <Shield className="w-3 h-3 text-cyan-500" />
        <span className="text-muted-foreground">{label}:</span>
        <span className="font-medium">{value}</span>
    </div>
);

// Monthly Return Heatmap
const MonthlyReturnHeatmap: React.FC<{ data: MonthlyReturn[] }> = ({ data }) => {
    const years = [...new Set(data.map(d => d.year))].sort();
    
    const getCellColor = (ret: number) => {
        if (ret >= 5) return 'bg-green-600 text-white';
        if (ret >= 2) return 'bg-green-400 text-white';
        if (ret >= 0) return 'bg-green-200 text-gray-800';
        if (ret >= -2) return 'bg-red-200 text-gray-800';
        if (ret >= -5) return 'bg-red-400 text-white';
        return 'bg-red-600 text-white';
    };

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-xs">
                <thead>
                    <tr>
                        <th className="p-2 text-left">Year</th>
                        {MONTH_NAMES.map(m => <th key={m} className="p-2 text-center w-12">{m}</th>)}
                        <th className="p-2 text-center font-bold">YTD</th>
                    </tr>
                </thead>
                <tbody>
                    {years.map(year => {
                        const yearData = data.filter(d => d.year === year);
                        const ytd = yearData.reduce((sum, d) => sum + d.return, 0);
                        return (
                            <tr key={year} className="border-t">
                                <td className="p-2 font-medium">{year}</td>
                                {MONTH_NAMES.map((_, idx) => {
                                    const monthData = yearData.find(d => d.month === idx + 1);
                                    return (
                                        <td key={idx} className="p-1">
                                            {monthData ? (
                                                <div className={`p-1 rounded text-center ${getCellColor(monthData.return)}`}>
                                                    {monthData.return > 0 ? '+' : ''}{monthData.return.toFixed(1)}
                                                </div>
                                            ) : <div className="p-1 text-center text-muted-foreground">-</div>}
                                        </td>
                                    );
                                })}
                                <td className="p-1">
                                    <div className={`p-1 rounded text-center font-bold ${getCellColor(ytd)}`}>
                                        {ytd > 0 ? '+' : ''}{ytd.toFixed(1)}%
                                    </div>
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
};

const BacktestLab: React.FC = () => {
    const [result, setResult] = useState<BacktestResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const [startYear, setStartYear] = useState(2021);
    const [endYear, setEndYear] = useState(2024);
    const [topN, setTopN] = useState(50);
    const [benchmarks, setBenchmarks] = useState(['SPY', 'QQQ']);
    const [customTicker, setCustomTicker] = useState('');
    
    const [showDrawdown, setShowDrawdown] = useState(true);
    const [showRollingSharpe, setShowRollingSharpe] = useState(true);
    const [showHeatmap, setShowHeatmap] = useState(true);
    const [showTurnover, setShowTurnover] = useState(false);
    const [showGrossCurve, setShowGrossCurve] = useState(false);

    const runBacktest = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(
                `http://localhost:8000/api/v1/quant/backtest/run?start_year=${startYear}&end_year=${endYear}&top_n=${topN}&benchmarks=${benchmarks.join(',')}`
            );
            const data = await res.json();
            if (data.error) setError(data.error);
            else setResult(data);
        } catch (e) {
            setError('Failed to run backtest. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    const addBenchmark = () => {
        if (customTicker && !benchmarks.includes(customTicker.toUpperCase())) {
            setBenchmarks([...benchmarks, customTicker.toUpperCase()]);
            setCustomTicker('');
        }
    };

    const removeBenchmark = (ticker: string) => setBenchmarks(benchmarks.filter(b => b !== ticker));

    const chartData = useMemo(() => {
        if (!result) return [];
        const dateMap: { [date: string]: any } = {};
        result.strategy.curve.forEach(p => { dateMap[p.date] = { date: p.date, 'Net': p.value }; });
        if (showGrossCurve && result.strategy.curve_gross) {
            result.strategy.curve_gross.forEach(p => { if (dateMap[p.date]) dateMap[p.date]['Gross'] = p.value; });
        }
        Object.entries(result.benchmarks).forEach(([ticker, { curve }]) => {
            curve.forEach(p => { if (dateMap[p.date]) dateMap[p.date][ticker] = p.value; });
        });
        return Object.values(dateMap).sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    }, [result, showGrossCurve]);

    const robustness = result?.robustness;

    const getBenchmarkColor = (ticker: string, idx: number): string => {
        return COLORS.benchmark[idx % COLORS.benchmark.length];
    };

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex justify-between items-start">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
                        <BarChart3 className="w-8 h-8 text-green-500" />
                        Backtest Lab
                        <Badge variant="outline" className="text-xs ml-2 border-cyan-500 text-cyan-500">AQR Standard</Badge>
                    </h1>
                    <p className="text-muted-foreground mt-1">
                        Professional Walk-Forward Backtest with Risk Controls
                    </p>
                </div>
                <Button onClick={runBacktest} disabled={loading} className="bg-green-600 hover:bg-green-700" size="lg">
                    {loading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Running...</> : <><Play className="w-4 h-4 mr-2" />Run Backtest</>}
                </Button>
            </motion.div>

            {/* Risk Controls Display */}
            {result?.risk_controls && (
                <div className="flex flex-wrap gap-2">
                    <RiskControlBadge label="Max Position" value={result.risk_controls.max_position} />
                    <RiskControlBadge label="Sector Cap" value={result.risk_controls.sector_cap} />
                    <RiskControlBadge label="Stop Loss" value={result.risk_controls.stop_loss} />
                    <RiskControlBadge label="Min Hold" value={result.risk_controls.min_holding} />
                </div>
            )}

            {/* Configuration Card */}
            <Card className="border-l-4 border-l-green-500">
                <CardHeader className="py-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                        <FlaskConical className="w-4 h-4 text-green-500" />
                        Backtest Configuration
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                        <div>
                            <label className="text-sm text-muted-foreground block mb-2">Period: {startYear} - {endYear}</label>
                            <div className="flex gap-2">
                                <Input type="number" value={startYear} onChange={(e) => setStartYear(parseInt(e.target.value))} className="w-24" min={2020} max={endYear - 1} />
                                <span className="self-center">to</span>
                                <Input type="number" value={endYear} onChange={(e) => setEndYear(parseInt(e.target.value))} className="w-24" min={startYear + 1} max={2024} />
                            </div>
                        </div>
                        <div>
                            <label className="text-sm text-muted-foreground block mb-2">Top N Stocks: {topN}</label>
                            <Slider value={[topN]} onValueChange={([val]) => setTopN(val)} min={10} max={100} step={10} />
                        </div>
                        <div className="col-span-2">
                            <label className="text-sm text-muted-foreground block mb-2">Compare Against</label>
                            <div className="flex flex-wrap gap-2 items-center">
                                {benchmarks.map(ticker => (
                                    <Badge key={ticker} variant="outline" className="px-2 py-1 cursor-pointer hover:bg-destructive/20" onClick={() => removeBenchmark(ticker)}>
                                        {ticker} <X className="w-3 h-3 ml-1" />
                                    </Badge>
                                ))}
                                <div className="flex gap-2">
                                    <Input value={customTicker} onChange={(e) => setCustomTicker(e.target.value.toUpperCase())} placeholder="Add ticker..." className="w-28 h-8" onKeyDown={(e) => e.key === 'Enter' && addBenchmark()} />
                                    <Button size="sm" variant="outline" onClick={addBenchmark}><Plus className="w-4 h-4" /></Button>
                                </div>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {error && <Card className="border-red-500 bg-red-500/10"><CardContent className="py-4 text-red-500">⚠️ {error}</CardContent></Card>}

            {result && (
                <>
                    {/* Key Metrics Summary */}
                    <div className="grid grid-cols-2 md:grid-cols-7 gap-3">
                        <Card className="bg-green-500/10 border-green-500/50">
                            <CardContent className="p-3 text-center">
                                <div className="text-2xl font-bold text-green-500">{result.strategy.metrics.sharpe_ratio.toFixed(2)}</div>
                                <div className="text-xs text-muted-foreground">Net Sharpe</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-muted/30">
                            <CardContent className="p-3 text-center">
                                <div className="text-2xl font-bold">{result.strategy.metrics.sharpe_ratio_gross?.toFixed(2) || '-'}</div>
                                <div className="text-xs text-muted-foreground">Gross Sharpe</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-muted/30">
                            <CardContent className="p-3 text-center">
                                <div className="text-2xl font-bold text-cyan-500">+{result.strategy.metrics.annual_return.toFixed(1)}%</div>
                                <div className="text-xs text-muted-foreground">Annual Return</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-muted/30">
                            <CardContent className="p-3 text-center">
                                <div className="text-2xl font-bold text-amber-500">{result.strategy.metrics.avg_turnover?.toFixed(1) || 0}%</div>
                                <div className="text-xs text-muted-foreground">Avg Turnover</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-muted/30">
                            <CardContent className="p-3 text-center">
                                <div className="text-2xl font-bold text-red-400">{result.strategy.metrics.max_drawdown.toFixed(1)}%</div>
                                <div className="text-xs text-muted-foreground">Max Drawdown</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-muted/30">
                            <CardContent className="p-3 text-center">
                                <div className="text-2xl font-bold text-purple-400">{result.strategy.metrics.total_transaction_costs?.toFixed(2) || 0}%</div>
                                <div className="text-xs text-muted-foreground">Total Costs</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-muted/30">
                            <CardContent className="p-3 text-center">
                                <div className="text-2xl font-bold text-orange-400">{result.strategy.metrics.total_stop_losses || 0}</div>
                                <div className="text-xs text-muted-foreground">Stop Losses</div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Bootstrap CI & Robustness */}
                    {robustness && (
                        <Card className="border-l-4 border-l-cyan-500">
                            <CardHeader className="py-3">
                                <CardTitle className="text-sm flex items-center gap-2">
                                    <Target className="w-4 h-4 text-cyan-500" />
                                    Robustness Analysis (Bootstrap 95% CI)
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* Bootstrap Sharpe CI */}
                                    <div className="bg-muted/30 p-4 rounded-lg">
                                        <div className="text-sm text-muted-foreground mb-2">Sharpe Ratio Confidence Interval</div>
                                        <div className="flex items-center gap-4">
                                            <div className="text-3xl font-bold text-green-500">{robustness.bootstrap_sharpe.mean.toFixed(2)}</div>
                                            <div className="text-sm">
                                                <div className="flex items-center gap-1">
                                                    <span className="text-muted-foreground">95% CI:</span>
                                                    <span className={robustness.bootstrap_sharpe.lower_95 > 0 ? 'text-green-400' : 'text-red-400'}>
                                                        [{robustness.bootstrap_sharpe.lower_95.toFixed(2)}, {robustness.bootstrap_sharpe.upper_95.toFixed(2)}]
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-1">
                                                    <span className="text-muted-foreground">Std:</span>
                                                    <span>{robustness.bootstrap_sharpe.std.toFixed(2)}</span>
                                                </div>
                                            </div>
                                            {robustness.bootstrap_sharpe.lower_95 > 0 ? (
                                                <Badge className="bg-green-500/20 text-green-500"><CheckCircle className="w-3 h-3 mr-1" />Significant</Badge>
                                            ) : (
                                                <Badge className="bg-amber-500/20 text-amber-500"><AlertTriangle className="w-3 h-3 mr-1" />Uncertain</Badge>
                                            )}
                                        </div>
                                    </div>
                                    
                                    {/* Subperiod Analysis */}
                                    <div className="bg-muted/30 p-4 rounded-lg">
                                        <div className="text-sm text-muted-foreground mb-2">Year-by-Year Performance</div>
                                        <div className="overflow-x-auto">
                                            <table className="w-full text-xs">
                                                <thead>
                                                    <tr className="border-b">
                                                        <th className="py-1 text-left">Year</th>
                                                        <th className="py-1 text-right">Return</th>
                                                        <th className="py-1 text-right">Sharpe</th>
                                                        <th className="py-1 text-right">Max DD</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {robustness.subperiod.map(sp => (
                                                        <tr key={sp.year} className="border-b">
                                                            <td className="py-1">{sp.year}</td>
                                                            <td className={`py-1 text-right ${sp.return >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                                {sp.return >= 0 ? '+' : ''}{sp.return}%
                                                            </td>
                                                            <td className={`py-1 text-right ${sp.sharpe >= 1 ? 'text-green-400' : ''}`}>{sp.sharpe}</td>
                                                            <td className="py-1 text-right text-red-400">{sp.max_drawdown}%</td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    )}

                    {/* Equity Curve Chart */}
                    <Card>
                        <CardHeader>
                            <div className="flex justify-between items-center">
                                <CardTitle className="flex items-center gap-2"><TrendingUp className="w-5 h-5 text-green-500" />Equity Curve Comparison</CardTitle>
                                <Button variant={showGrossCurve ? "default" : "outline"} size="sm" onClick={() => setShowGrossCurve(!showGrossCurve)}>
                                    <DollarSign className="w-4 h-4 mr-1" />{showGrossCurve ? 'Hide' : 'Show'} Gross
                                </Button>
                            </div>
                            <CardDescription>Normalized to $1.00 at start (Net includes 10 bps costs + risk controls)</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="h-[400px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={chartData}>
                                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                        <XAxis dataKey="date" tickFormatter={(d) => new Date(d).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })} />
                                        <YAxis domain={['auto', 'auto']} tickFormatter={(v) => `$${v.toFixed(2)}`} />
                                        <Tooltip formatter={(value: number) => [`$${value.toFixed(3)}`, '']} labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                                        <Legend />
                                        <ReferenceLine y={1} stroke="#666" strokeDasharray="3 3" />
                                        <Line type="monotone" dataKey="Net" stroke={COLORS.strategy} strokeWidth={3} dot={false} name="Strategy (Net)" />
                                        {showGrossCurve && <Line type="monotone" dataKey="Gross" stroke={COLORS.strategyGross} strokeWidth={2} dot={false} strokeDasharray="5 5" name="Strategy (Gross)" />}
                                        {Object.keys(result.benchmarks).map((ticker, idx) => (
                                            <Line key={ticker} type="monotone" dataKey={ticker} stroke={getBenchmarkColor(ticker, idx)} strokeWidth={2} dot={false} strokeDasharray={ticker !== 'SPY' ? '5 5' : undefined} />
                                        ))}
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Rolling Sharpe */}
                    {result.rolling_sharpe && result.rolling_sharpe.length > 0 && (
                        <Card>
                            <CardHeader className="cursor-pointer" onClick={() => setShowRollingSharpe(!showRollingSharpe)}>
                                <CardTitle className="flex items-center justify-between">
                                    <div className="flex items-center gap-2"><RefreshCw className="w-5 h-5 text-cyan-500" />Rolling 12-Month Sharpe</div>
                                    {showRollingSharpe ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                </CardTitle>
                            </CardHeader>
                            {showRollingSharpe && (
                                <CardContent>
                                    <div className="h-[200px]">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={result.rolling_sharpe}>
                                                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                                <XAxis dataKey="date" tickFormatter={(d) => new Date(d).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })} />
                                                <YAxis domain={['auto', 'auto']} />
                                                <Tooltip formatter={(value: number) => [value.toFixed(2), 'Sharpe']} labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                                                <ReferenceLine y={1} stroke="#22c55e" strokeDasharray="3 3" label="1.0" />
                                                <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="3 3" />
                                                <Line type="monotone" dataKey="rolling_sharpe" stroke="#22d3ee" strokeWidth={2} dot={false} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                </CardContent>
                            )}
                        </Card>
                    )}

                    {/* Monthly Return Heatmap */}
                    {result.monthly_returns && result.monthly_returns.length > 0 && (
                        <Card>
                            <CardHeader className="cursor-pointer" onClick={() => setShowHeatmap(!showHeatmap)}>
                                <CardTitle className="flex items-center justify-between">
                                    <div className="flex items-center gap-2"><BarChart3 className="w-5 h-5 text-amber-500" />Monthly Returns (%)</div>
                                    {showHeatmap ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                </CardTitle>
                            </CardHeader>
                            {showHeatmap && <CardContent><MonthlyReturnHeatmap data={result.monthly_returns} /></CardContent>}
                        </Card>
                    )}

                    {/* Drawdown */}
                    <Card>
                        <CardHeader className="cursor-pointer" onClick={() => setShowDrawdown(!showDrawdown)}>
                            <CardTitle className="flex items-center justify-between">
                                <div className="flex items-center gap-2"><TrendingDown className="w-5 h-5 text-red-500" />Drawdown Analysis</div>
                                {showDrawdown ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                            </CardTitle>
                        </CardHeader>
                        {showDrawdown && (
                            <CardContent>
                                <div className="h-[200px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={result.drawdown}>
                                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                            <XAxis dataKey="date" tickFormatter={(d) => new Date(d).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })} />
                                            <YAxis domain={['auto', 0]} tickFormatter={(v) => `${v.toFixed(0)}%`} />
                                            <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, 'Drawdown']} labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                                            <ReferenceLine y={-10} stroke="#ef4444" strokeDasharray="3 3" label="-10%" />
                                            <ReferenceLine y={-15} stroke="#dc2626" strokeDasharray="3 3" label="-15% (Stop)" />
                                            <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="#ef4444" fillOpacity={0.3} />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </CardContent>
                        )}
                    </Card>

                    {/* Turnover */}
                    {result.turnover && result.turnover.length > 0 && (
                        <Card>
                            <CardHeader className="cursor-pointer" onClick={() => setShowTurnover(!showTurnover)}>
                                <CardTitle className="flex items-center justify-between">
                                    <div className="flex items-center gap-2"><RefreshCw className="w-5 h-5 text-purple-500" />Monthly Turnover</div>
                                    {showTurnover ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                </CardTitle>
                            </CardHeader>
                            {showTurnover && (
                                <CardContent>
                                    <div className="h-[150px]">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={result.turnover}>
                                                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                                                <XAxis dataKey="date" tickFormatter={(d) => new Date(d).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })} />
                                                <YAxis tickFormatter={(v) => `${v}%`} />
                                                <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, 'Turnover']} labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                                                <Bar dataKey="turnover" fill="#a855f7" radius={[2, 2, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </CardContent>
                            )}
                        </Card>
                    )}
                </>
            )}

            {/* Empty State */}
            {!result && !loading && !error && (
                <Card className="text-center py-16">
                    <CardContent>
                        <BarChart3 className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                        <h3 className="text-xl font-semibold mb-2">No Backtest Results Yet</h3>
                        <p className="text-muted-foreground mb-4">Configure parameters and click "Run Backtest" to see factor model performance.</p>
                        <Button onClick={runBacktest} className="bg-green-600 hover:bg-green-700"><Play className="w-4 h-4 mr-2" />Run Your First Backtest</Button>
                    </CardContent>
                </Card>
            )}
        </div>
    );
};

export default BacktestLab;
