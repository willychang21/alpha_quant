import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { PieChart, Pie, Cell, Tooltip as RechartsTooltip, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid } from 'recharts';
import { motion } from "framer-motion";

// ... (interfaces)

// ...



interface Ranking {
  rank: number;
  ticker: string;
  score: number;
  date: string;
  metadata: string;
}

interface PortfolioTarget {
  ticker: string;
  weight: number;
  date: string;
}

interface BacktestPoint {
  date: string;
  value: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1', '#a4de6c', '#d0ed57'];

export function QuantDashboard() {
  const [rankings, setRankings] = useState<Ranking[]>([]);
  const [portfolio, setPortfolio] = useState<PortfolioTarget[]>([]);
  const [backtest, setBacktest] = useState<BacktestPoint[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchRankings();
    fetchPortfolio();
  }, []);

  const fetchRankings = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/quant/rankings');
      const data = await res.json();
      setRankings(data);
    } catch (e) {
      console.error("Failed to fetch rankings", e);
    }
  };

  const fetchPortfolio = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/quant/portfolio');
      const data = await res.json();
      setPortfolio(data);
    } catch (e) {
      console.error("Failed to fetch portfolio", e);
    }
  };

  const runBacktest = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/v1/quant/backtest', { method: 'POST' });
      const data = await res.json();
      setBacktest(data);
    } catch (e) {
      console.error("Failed to run backtest", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex justify-between items-center"
      >
        <h1 className="text-3xl font-bold tracking-tight">Quant System</h1>
        <Button onClick={runBacktest} disabled={loading}>
          {loading ? "Running Simulation..." : "Run Backtest"}
        </Button>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Top Picks */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Top Ranked Stocks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="max-h-[400px] overflow-y-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Rank</TableHead>
                    <TableHead>Ticker</TableHead>
                    <TableHead>Score</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rankings.slice(0, 10).map((r) => (
                    <TableRow key={r.ticker}>
                      <TableCell className="font-medium">#{r.rank}</TableCell>
                      <TableCell>{r.ticker}</TableCell>
                      <TableCell>{r.score.toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>

        {/* Portfolio Allocation */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Target Allocation</CardTitle>
          </CardHeader>
          <CardContent className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={portfolio as any[]}
                  cx="50%"
                  cy="50%"
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="weight"
                  nameKey="ticker"
                  label={(props: any) => `${props.ticker} ${(props.percent * 100).toFixed(0)}%`}
                >
                  {portfolio.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Backtest Results */}
      {backtest.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Backtest Performance (1 Year)</CardTitle>
          </CardHeader>
          <CardContent className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={backtest}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis dataKey="date" tickFormatter={(str) => new Date(str).toLocaleDateString()} />
                <YAxis domain={['auto', 'auto']} />
                <RechartsTooltip labelFormatter={(str) => new Date(str).toLocaleDateString()} />
                <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
