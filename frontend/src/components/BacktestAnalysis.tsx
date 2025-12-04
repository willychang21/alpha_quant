import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  AreaChart,
  Area,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Loader2, TrendingUp, AlertCircle } from "lucide-react";
import { DcaConfig } from '../store/useStore';
import api from '../api/axios';
import { API_ENDPOINTS } from '../api/endpoints';

interface BacktestAnalysisProps {
  dcaConfig: DcaConfig;
}

interface BacktestData {
  lumpSum: { date: string; value: number }[];
  dca: { date: string; value: number; invested: number }[];
  monthlyReturns: { year: number; returns: (number | null)[]; total: number | null }[];
  metrics: {
    cagr: number;
    maxDrawdown: number;
    bestYear: number;
    finalBalance: number;
    dcaFinalBalance: number;
    dcaTotalInvested: number;
    dcaTotalReturn: number;
    dcaMaxDrawdown: number;
  };
}

const BacktestAnalysis: React.FC<BacktestAnalysisProps> = ({ dcaConfig }) => {
  const [data, setData] = useState<BacktestData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeMode, setActiveMode] = useState<'lumpSum' | 'dca'>('lumpSum');

  const [timeRange, setTimeRange] = useState('ALL');

  useEffect(() => {
    const runBacktest = async () => {
      // Only run if we have assets selected
      if (!dcaConfig.selectedAssets || dcaConfig.selectedAssets.length === 0) {
        setData(null);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        // Prepare allocation map
        const allocation: Record<string, number> = {};
        dcaConfig.selectedAssets.forEach(asset => {
          allocation[asset.id] = typeof asset.allocation === 'string' ? parseFloat(asset.allocation) : asset.allocation;
        });

        const response = await api.post(API_ENDPOINTS.BACKTEST, {
          allocation,
          initialAmount: 10000, // Standard "Growth of 10k"
          monthlyAmount: dcaConfig.monthlyAmount || 1000
        });

        setData(response.data);
      } catch (err: any) {
        console.error(err);
        setError(err.message || 'Failed to fetch backtest data');
      } finally {
        setLoading(false);
      }
    };

    // Debounce slightly to avoid rapid calls
    const timeoutId = setTimeout(() => {
      runBacktest();
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [dcaConfig]);

  // Helper to simulate DCA on a subset of data
  const simulateDCA = (priceData: { date: string; value: number }[], monthlyAmount: number) => {
    if (!priceData || priceData.length === 0) return [];

    const amount = monthlyAmount || 0;
    let units = 0;
    let totalInvested = 0;
    const dcaData: { date: string; value: number; invested: number }[] = [];

    // We assume the priceData is monthly.
    // For each month, we contribute monthlyAmount.
    priceData.forEach((point) => {
      const price = point.value;
      // Buy units at current price
      const unitsBought = amount / price;
      units += unitsBought;
      totalInvested += amount;

      dcaData.push({
        date: point.date,
        value: units * price,
        invested: totalInvested
      });
    });

    return dcaData;
  };

  // Filter Data based on Time Range
  const getFilteredData = () => {
    if (!data) return { filteredLumpSum: [], filteredDca: [], filteredMetrics: null };

    const { lumpSum } = data;
    if (!lumpSum.length) return { filteredLumpSum: [], filteredDca: [], filteredMetrics: null };

    const endDate = new Date(lumpSum[lumpSum.length - 1].date);
    let startDate = new Date(lumpSum[0].date);

    const now = new Date();
    
    switch (timeRange) {
      case 'YTD':
        startDate = new Date(now.getFullYear(), 0, 1);
        break;
      case '1Y':
        startDate = new Date(now);
        startDate.setFullYear(now.getFullYear() - 1);
        break;
      case '3Y':
        startDate = new Date(now);
        startDate.setFullYear(now.getFullYear() - 3);
        break;
      case '5Y':
        startDate = new Date(now);
        startDate.setFullYear(now.getFullYear() - 5);
        break;
      case '10Y':
        startDate = new Date(now);
        startDate.setFullYear(now.getFullYear() - 10);
        break;
      case '20Y':
        startDate = new Date(now);
        startDate.setFullYear(now.getFullYear() - 20);
        break;
      case 'ALL':
      default:
        startDate = new Date(lumpSum[0].date);
        break;
    }

    // Filter Lump Sum (Price History)
    const filteredLumpSum = lumpSum.filter(d => new Date(d.date) >= startDate);
    
    // Determine DCA Data
    let filteredDca: { date: string; value: number; invested: number }[] = [];
    let dcaMetrics: Partial<BacktestData['metrics']> = {};

    // Always calculate dynamic DCA so we have the data ready
    filteredDca = simulateDCA(filteredLumpSum, dcaConfig.monthlyAmount || 1000);
      if (filteredDca.length > 0) {
      const finalDca = filteredDca[filteredDca.length - 1];
      const dcaBalance = finalDca.value;
      const totalInvested = finalDca.invested;
      const dcaTotalReturn = totalInvested > 0 ? (dcaBalance - totalInvested) / totalInvested : 0;
      
      let dcaMaxDrawdown = 0;
      let dcaPeak = -Infinity;
      for (const point of filteredDca) {
        if (point.value > dcaPeak) dcaPeak = point.value;
        const drawdown = (dcaPeak - point.value) / dcaPeak;
        if (drawdown > dcaMaxDrawdown) dcaMaxDrawdown = drawdown;
      }
      
      dcaMetrics = {
        dcaFinalBalance: dcaBalance,
        dcaTotalInvested: totalInvested,
        dcaTotalReturn: dcaTotalReturn * 100,
        dcaMaxDrawdown: dcaMaxDrawdown * 100
      };
    }

    // Recalculate Metrics for the filtered period (Lump Sum)
    if (filteredLumpSum.length < 2) return { filteredLumpSum, filteredDca, filteredMetrics: data.metrics };

    const startLumpSum = filteredLumpSum[0].value;
    const endLumpSum = filteredLumpSum[filteredLumpSum.length - 1].value;
    const years = (new Date(filteredLumpSum[filteredLumpSum.length - 1].date).getTime() - new Date(filteredLumpSum[0].date).getTime()) / (1000 * 60 * 60 * 24 * 365.25);
    
    // Lump Sum Metrics
    // const totalReturn = (endLumpSum - startLumpSum) / startLumpSum;
    const cagr = years > 0 ? Math.pow(endLumpSum / startLumpSum, 1 / years) - 1 : 0;
    
    let maxDrawdown = 0;
    let peak = -Infinity;
    for (const point of filteredLumpSum) {
      if (point.value > peak) peak = point.value;
      const drawdown = (peak - point.value) / peak;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    }

    // Best Year (approximate from filtered monthly returns)
    const startYear = startDate.getFullYear();
    const endYear = endDate.getFullYear();
    const filteredMonthlyReturns = data.monthlyReturns.filter(r => r.year >= startYear && r.year <= endYear);
    const yearTotals = filteredMonthlyReturns.map(r => r.total).filter((t): t is number => t !== null);
    const bestYear = yearTotals.length ? Math.max(...yearTotals) : 0;

    const filteredMetrics = {
      ...data.metrics,
      cagr: cagr * 100,
      maxDrawdown: maxDrawdown * 100,
      bestYear: bestYear,
      ...dcaMetrics
    };

    return { filteredLumpSum, filteredDca, filteredMetrics };
  };

  const { filteredLumpSum, filteredDca, filteredMetrics } = getFilteredData();
  const displayMetrics = filteredMetrics || (data ? data.metrics : null);

  if (!dcaConfig.selectedAssets || dcaConfig.selectedAssets.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
        <TrendingUp className="h-12 w-12 mb-4 opacity-20" />
        <p>Add assets to your DCA plan to see hypothetical performance.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96">
        <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
        <p className="text-muted-foreground text-sm">Simulating historical performance...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 rounded-lg bg-red-50 text-red-900 border border-red-200 flex items-center gap-3">
        <AlertCircle className="h-5 w-5 text-red-600" />
        <div>
          <h4 className="font-medium">Error</h4>
          <p className="text-sm">Failed to run backtest: {error}. Please try again later.</p>
        </div>
      </div>
    );
  }

  if (!data || (!data.lumpSum.length && !data.dca.length)) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground">
        <AlertCircle className="h-12 w-12 mb-4 opacity-20" />
        <p>No historical data available for this combination of assets.</p>
      </div>
    );
  }

  const { monthlyReturns } = data;

  // Helper for heatmap colors
  const getReturnColor = (value: number | null) => {
    if (value === null) return 'bg-muted/20';
    if (value >= 0) {
      // Green intensity
      if (value > 10) return 'bg-green-500 text-white font-bold';
      if (value > 5) return 'bg-green-400 text-black';
      if (value > 2) return 'bg-green-300 text-black';
      return 'bg-green-200 text-black';
    } else {
      // Red intensity
      if (value < -10) return 'bg-red-500 text-white font-bold';
      if (value < -5) return 'bg-red-400 text-black';
      if (value < -2) return 'bg-red-300 text-black';
      return 'bg-red-200 text-black';
    }
  };

  const ranges = ['YTD', '1Y', '3Y', '5Y', '10Y', '20Y', 'ALL'];

  return (
    <div className="space-y-8 animate-in fade-in duration-500 w-full">
      <div className="w-full flex flex-col md:flex-row justify-between items-center gap-4 mb-6">
        {/* Mode Switcher */}
        <div className="p-1 h-auto flex flex-row bg-secondary/50 rounded-full relative z-10 w-fit">
          <button
            onClick={() => setActiveMode('lumpSum')}
            className={`transition-all duration-200 px-6 py-2.5 rounded-full font-medium text-sm flex flex-row items-center gap-2 ${activeMode === 'lumpSum' ? 'bg-white dark:bg-card text-primary shadow-sm' : 'text-muted-foreground hover:text-foreground hover:bg-white/50 dark:hover:bg-white/10'}`}
          >
            Lump Sum ($10k)
          </button>
          <button
            onClick={() => setActiveMode('dca')}
            className={`transition-all duration-200 px-6 py-2.5 rounded-full font-medium text-sm flex flex-row items-center gap-2 ${activeMode === 'dca' ? 'bg-white dark:bg-card text-primary shadow-sm' : 'text-muted-foreground hover:text-foreground hover:bg-white/50 dark:hover:bg-white/10'}`}
          >
            DCA (Monthly)
          </button>
        </div>

        {/* Time Range Selector */}
        <div className="flex items-center gap-1 bg-secondary/30 p-1.5 rounded-full overflow-x-auto max-w-full">
          {ranges.map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-1.5 text-xs font-medium rounded-full transition-all whitespace-nowrap ${
                timeRange === range 
                  ? 'bg-primary text-primary-foreground shadow-sm' 
                  : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Summary Metrics */}
      {displayMetrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <Card className="material-card p-6 flex flex-col justify-between hover:shadow-md transition-shadow h-full">
              <CardHeader className="p-0 pb-2">
                <CardDescription className="text-sm font-medium">
                  {activeMode === 'lumpSum' ? 'CAGR' : 'Total Return'}
                </CardDescription>
                <CardTitle className="text-3xl font-normal text-primary">
                  {activeMode === 'lumpSum' 
                    ? `${displayMetrics.cagr.toFixed(2)}%`
                    : `${displayMetrics.dcaTotalReturn?.toFixed(2)}%`
                  }
                </CardTitle>
              </CardHeader>
            </Card>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <Card className="material-card p-6 flex flex-col justify-between hover:shadow-md transition-shadow h-full">
              <CardHeader className="p-0 pb-2">
                <CardDescription className="text-sm font-medium">Max Drawdown</CardDescription>
                <CardTitle className="text-3xl font-normal text-danger">
                  {activeMode === 'lumpSum'
                    ? `${displayMetrics.maxDrawdown.toFixed(2)}%`
                    : `${displayMetrics.dcaMaxDrawdown?.toFixed(2)}%`
                  }
                </CardTitle>
              </CardHeader>
            </Card>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            <Card className="material-card p-6 flex flex-col justify-between hover:shadow-md transition-shadow h-full">
              <CardHeader className="p-0 pb-2">
                <CardDescription className="text-sm font-medium">Best Year</CardDescription>
                <CardTitle className="text-3xl font-normal text-success">
                  {displayMetrics.bestYear ? `+${displayMetrics.bestYear.toFixed(2)}%` : '-'}
                </CardTitle>
              </CardHeader>
            </Card>
          </motion.div>
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
            <Card className="material-card p-6 flex flex-col justify-between hover:shadow-md transition-shadow h-full">
              <CardHeader className="p-0 pb-2">
                <CardDescription className="text-sm font-medium">Final Balance</CardDescription>
                <CardTitle className="text-3xl font-normal font-mono tracking-tight">
                  ${activeMode === 'lumpSum' 
                    ? (displayMetrics.finalBalance?.toLocaleString(undefined, { maximumFractionDigits: 0 }) || '-')
                    : (displayMetrics.dcaFinalBalance?.toLocaleString(undefined, { maximumFractionDigits: 0 }) || '-')
                  }
                </CardTitle>
              </CardHeader>
            </Card>
          </motion.div>
        </div>
      )}

      {activeMode === 'lumpSum' && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }} className="space-y-4">
          <Card className="material-card w-full p-6">
            <CardHeader className="px-0 pt-0 pb-6">
              <CardTitle className="text-xl font-normal">Growth of $10,000</CardTitle>
              <CardDescription>
                Hypothetical performance of a one-time $10,000 investment.
              </CardDescription>
            </CardHeader>
            <CardContent className="px-0 pb-0">
              <div className="h-[500px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={filteredLumpSum}>
                    <defs>
                      <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" opacity={0.5} />
                    <XAxis 
                      dataKey="date" 
                      tickFormatter={(str) => new Date(str).getFullYear().toString()}
                      minTickGap={50}
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                      dy={10}
                    />
                    <YAxis 
                      tickFormatter={(val) => `$${val?.toLocaleString() || '0'}`}
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                      dx={-10}
                      domain={['auto', 'auto']}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(var(--popover))', 
                        borderColor: 'transparent',
                        borderRadius: '16px',
                        boxShadow: 'var(--shadow-md)',
                        padding: '12px 16px'
                      }}
                      itemStyle={{ color: 'hsl(var(--foreground))' }}
                      formatter={(value: number) => [`$${value?.toLocaleString() || '0'}`, 'Portfolio Value']}
                      labelFormatter={(label) => new Date(label).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                      fillOpacity={1} 
                      fill="url(#colorValue)" 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {activeMode === 'dca' && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }} className="space-y-4">
          <Card className="material-card w-full p-6">
            <CardHeader className="px-0 pt-0 pb-6">
              <CardTitle className="text-xl font-normal">DCA Simulation</CardTitle>
              <CardDescription>
                Monthly investment of ${dcaConfig.monthlyAmount?.toLocaleString() || '0'} starting from $0.
              </CardDescription>
            </CardHeader>
            <CardContent className="px-0 pb-0">
              <div className="h-[500px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={filteredDca}>
                    <defs>
                      <linearGradient id="colorDcaValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" opacity={0.5} />
                    <XAxis 
                      dataKey="date" 
                      tickFormatter={(str) => new Date(str).getFullYear().toString()}
                      minTickGap={50}
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                      dy={10}
                    />
                    <YAxis 
                      tickFormatter={(val) => `$${val?.toLocaleString() || '0'}`}
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                      dx={-10}
                      domain={['auto', 'auto']}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'hsl(var(--popover))', 
                        borderColor: 'transparent',
                        borderRadius: '16px',
                        boxShadow: 'var(--shadow-md)',
                        padding: '12px 16px'
                      }}
                      itemStyle={{ color: 'hsl(var(--foreground))' }}
                      labelFormatter={(label) => new Date(label).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      name="Portfolio Value"
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                      fillOpacity={1} 
                      fill="url(#colorDcaValue)" 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="invested" 
                      name="Total Invested"
                      stroke="hsl(var(--muted-foreground))" 
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      fillOpacity={0} 
                      fill="transparent" 
                    />
                    <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Monthly Returns Heatmap */}
      <Card className="material-card overflow-hidden p-6">
        <CardHeader className="px-0 pt-0 pb-6">
          <CardTitle className="text-xl font-normal">Monthly Returns</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto px-0 pb-0">
          <table className="w-full text-sm text-center border-collapse">
            <thead>
              <tr>
                <th className="p-3 text-left font-medium text-muted-foreground">Year</th>
                {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map(m => (
                  <th key={m} className="p-3 font-medium text-muted-foreground">{m}</th>
                ))}
                <th className="p-3 font-bold text-foreground">Total</th>
              </tr>
            </thead>
            <tbody>
              {monthlyReturns.map((row) => (
                <tr key={row.year} className="border-b border-border/30 last:border-0 hover:bg-secondary/10 transition-colors">
                  <td className="p-3 text-left font-mono font-medium text-foreground">{row.year}</td>
                  {row.returns.map((ret, idx) => (
                    <td key={idx} className="p-1">
                      <div className={`w-full h-full py-2.5 rounded-lg text-xs font-medium transition-transform hover:scale-105 ${getReturnColor(ret)}`}>
                        {ret !== null ? `${ret.toFixed(1)}%` : '-'}
                      </div>
                    </td>
                  ))}
                  <td className="p-3 font-mono font-bold">
                    <span className={(row.total || 0) >= 0 ? "text-success" : "text-danger"}>
                      {row.total !== null ? `${row.total.toFixed(1)}%` : '-'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
};

export default BacktestAnalysis;
