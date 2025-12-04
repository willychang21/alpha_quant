
import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import api from "@/api/axios";
import { motion } from 'framer-motion';
import { Activity, TrendingUp, TrendingDown, Info, Gauge, AlertTriangle } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { AreaChart, Area, ResponsiveContainer } from 'recharts';

interface MarketRiskIndicator {
  name: string;
  value: number;
  score: number;
  description: string;
  methodology?: string;
  signal: string;
  history: number[];
}

interface MarketRiskResponse {
  timestamp: string;
  indicators: Record<string, MarketRiskIndicator>;
  score: number;
  rating: string;
}

const BubbleAnalysis: React.FC = () => {
  const [data, setData] = useState<MarketRiskResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await api.get('/market/risk');
        setData(res.data);
      } catch (err) {
        console.error(err);
        setError("Failed to load market risk data.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Unified Color Logic: High Score = High Risk = Red. Low Score = Low Risk = Green.
  const getScoreColor = (score: number) => {
    if (score >= 76) return "text-red-500";
    if (score >= 45) return "text-yellow-500";
    return "text-green-500";
  };

  const getProgressColor = (score: number) => {
    if (score >= 76) return "bg-red-500";
    if (score >= 45) return "bg-yellow-500";
    return "bg-green-500";
  };

  const getBadgeClassName = (score: number) => {
    if (score >= 76) return "bg-red-500 hover:bg-red-600 text-white";
    if (score >= 45) return "bg-yellow-500 hover:bg-yellow-600 text-black";
    return "bg-green-500 hover:bg-green-600 text-white";
  };

  if (loading) return <div className="p-8 text-center animate-pulse">Analyzing Market Conditions...</div>;
  if (error) return <Alert variant="destructive"><AlertTriangle className="h-4 w-4" /><AlertTitle>Error</AlertTitle><AlertDescription>{error}</AlertDescription></Alert>;
  if (!data) return null;

  return (
    <div className="space-y-8 p-6 max-w-7xl mx-auto">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl font-bold tracking-tight flex items-center justify-center gap-3">
          <Activity className="h-10 w-10 text-primary" /> Market Risk Dashboard
        </h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          AI-driven analysis of market "bubble" conditions. Scores represent <strong>Percentile Rank</strong> (0-100) of risk relative to history.
          <br/>
          <span className="text-red-500 font-bold">High Score</span> = High Risk. <span className="text-yellow-500 font-bold">Mid Score</span> = Neutral. <span className="text-green-500 font-bold">Low Score</span> = Low Risk.
        </p>
      </motion.div>

      {/* Overall Score Card */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="border-primary/20 bg-primary/5">
          <CardHeader className="text-center pb-2">
            <CardTitle>Overall Bubble Score</CardTitle>
            <CardDescription>Weighted aggregate of all risk indicators</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center justify-center pt-4 pb-8">
            <div className="relative flex items-center justify-center w-48 h-48">
               {/* Circular Progress Placeholder */}
               <div className={`w-40 h-40 rounded-full border-8 border-muted flex items-center justify-center ${getScoreColor(data.score)} border-t-current border-r-current rotate-45 transition-all duration-1000`} style={{ borderColor: `var(--${getProgressColor(data.score).replace('bg-', '')})` }}>
                  <div className="transform -rotate-45 text-center">
                    <span className="text-5xl font-bold">{data.score}</span>
                    <div className="text-sm font-medium mt-1 uppercase tracking-wider">{data.rating}</div>
                  </div>
               </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Indicators Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Object.entries(data.indicators).map(([key, indicator], index) => (
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + index * 0.1 }}
          >
            <Card className="h-full hover:shadow-lg transition-shadow duration-300 flex flex-col">
              <CardHeader className="pb-2">
                <div className="flex justify-between items-start">
                  <CardTitle className="text-lg font-semibold flex items-center gap-2">
                    {key === 'qqqDeviation' && <TrendingUp className="h-5 w-5 text-blue-500" />}
                    {key === 'vix' && <Activity className="h-5 w-5 text-purple-500" />}
                    {key === 'yield10y' && <TrendingDown className="h-5 w-5 text-green-500" />}
                    {key === 'smartMoney' && <Gauge className="h-5 w-5 text-orange-500" />}
                    {key === 'putCall' && <TrendingDown className="h-5 w-5 text-pink-500" />}
                    {key === 'fedRate' && <Activity className="h-5 w-5 text-indigo-500" />}
                    {key === 'fearGreed' && <Gauge className="h-5 w-5 text-red-500" />}
                    {indicator.name}
                  </CardTitle>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-4 w-4 text-muted-foreground hover:text-primary transition-colors" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs">
                        <p className="font-semibold mb-1">Description:</p>
                        <p className="mb-2">{indicator.description}</p>
                        {indicator.methodology && (
                          <>
                            <p className="font-semibold mb-1">Methodology:</p>
                            <p className="text-xs text-muted-foreground">{indicator.methodology}</p>
                          </>
                        )}
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </CardHeader>
              <CardContent className="space-y-4 flex-1 flex flex-col">
                <div className="flex justify-between items-end">
                  <div>
                    <div className="text-3xl font-bold">{indicator.score.toFixed(0)}</div>
                    <div className="text-xs text-muted-foreground">Risk Score</div>
                  </div>
                  <Badge className={getBadgeClassName(indicator.score)}>{indicator.signal}</Badge>
                </div>
                <div className="flex justify-between items-end">
                  <div>
                    <div className="text-3xl font-bold font-mono">
                      {indicator.value.toFixed(2)}
                      {key === 'qqqDeviation' && '%'}
                      {key === 'yield10y' && '%'}
                    </div>
                    <div className="text-sm text-muted-foreground mt-1">Current Value</div>
                  </div>

                    <div className={`text-xs font-bold ${getScoreColor(indicator.score)}`}>
                      Risk Score: {indicator.score.toFixed(0)}/100
                    </div>
                  </div>

                
                {/* Sparkline Chart */}
                <div className="h-24 w-full mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={indicator.history.map((val, i) => ({ i, val }))}>
                      <defs>
                        <linearGradient id={`color${key}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8884d8" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <Area type="monotone" dataKey="val" stroke="#8884d8" fillOpacity={1} fill={`url(#color${key})`} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* Progress Bar */}
                <div className="h-2 w-full bg-muted rounded-full overflow-hidden mt-auto">
                  <motion.div 
                    className={`h-full ${getProgressColor(indicator.score)}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${indicator.score}%` }}
                    transition={{ duration: 1, delay: 0.5 }}
                  />
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default BubbleAnalysis;
