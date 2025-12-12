import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';
import { motion, AnimatePresence } from "framer-motion";
import { TrendingUp, TrendingDown, ArrowRight, Loader2, RefreshCw, ChevronDown, ChevronUp, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface SectorData {
  symbol: string;
  name: string;
  rs_ratio: number;
  rs_momentum: number;
  quadrant: string;
  previous_quadrant: string | null;
  transition_signal: boolean;
  score: number;
}

interface StockData {
  ticker: string;
  score: number;
  rank: number;
  capital_flow: number;
  money_flow: number;
  sector_flow: number;
  mfi: number;
  obv_zscore: number;
  flow_signal: string;
  vsm: number;
  sentiment: number;
}

interface SectorStocksResponse {
  status: string;
  sector: string;
  stock_count: number;
  avg_money_flow: number;
  inflow_count: number;
  outflow_count: number;
  stocks: StockData[];
}

interface SectorRotationResponse {
  status: string;
  sectors: SectorData[];
  quadrant_summary: Record<string, number>;
  capital_inflow: string[];
  capital_outflow: string[];
  hot_sectors: string[];
  generated_at: string;
}

const QUADRANT_COLORS: Record<string, string> = {
  'Leading': '#22c55e',
  'Improving': '#3b82f6',
  'Weakening': '#f59e0b',
  'Lagging': '#ef4444',
};

const QUADRANT_BG: Record<string, string> = {
  'Leading': 'bg-green-500/20 text-green-400',
  'Improving': 'bg-blue-500/20 text-blue-400',
  'Weakening': 'bg-yellow-500/20 text-yellow-400',
  'Lagging': 'bg-red-500/20 text-red-400',
};

const FLOW_SIGNAL_COLORS: Record<string, string> = {
  'Strong Inflow': 'text-green-400',
  'Inflow': 'text-green-300',
  'Neutral': 'text-muted-foreground',
  'Outflow': 'text-red-300',
  'Strong Outflow': 'text-red-400',
};

export function SectorRotation() {
  const [data, setData] = useState<SectorRotationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSector, setExpandedSector] = useState<string | null>(null);
  const [sectorStocks, setSectorStocks] = useState<SectorStocksResponse | null>(null);
  const [loadingStocks, setLoadingStocks] = useState(false);

  useEffect(() => {
    fetchSectorRotation();
  }, []);

  const fetchSectorRotation = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/v1/quant/sector-rotation');
      if (!res.ok) throw new Error('Failed to fetch sector rotation data');
      const result = await res.json();
      setData(result);
      setError(null);
    } catch (e) {
      console.error("Failed to fetch sector rotation", e);
      setError("Failed to load sector rotation data");
    } finally {
      setLoading(false);
    }
  };

  const fetchSectorStocks = async (sectorName: string) => {
    if (expandedSector === sectorName) {
      setExpandedSector(null);
      setSectorStocks(null);
      return;
    }
    
    setExpandedSector(sectorName);
    setLoadingStocks(true);
    try {
      const res = await fetch(`http://localhost:8000/api/v1/quant/sector-stocks/${encodeURIComponent(sectorName)}`);
      if (!res.ok) throw new Error('Failed to fetch sector stocks');
      const result = await res.json();
      setSectorStocks(result);
    } catch (e) {
      console.error("Failed to fetch sector stocks", e);
      setSectorStocks(null);
    } finally {
      setLoadingStocks(false);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-[400px]">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (error || !data || data.status !== 'success') {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-[400px] text-muted-foreground">
          {error || "No data available"}
        </CardContent>
      </Card>
    );
  }

  const chartData = data.sectors.map(s => ({
    x: s.rs_ratio,
    y: s.rs_momentum,
    name: s.name,
    symbol: s.symbol,
    quadrant: s.quadrant,
    transition: s.transition_signal
  }));

  return (
    <div className="space-y-6">
      {/* Capital Flow Summary Cards */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-4"
      >
        <Card className="border-green-500/30 bg-green-500/5">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              Capital Inflow Sectors
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {data.hot_sectors.map(sector => (
                <Badge key={sector} variant="outline" className="bg-blue-500/20 text-blue-400 border-blue-500/30 cursor-pointer hover:bg-blue-500/30" onClick={() => fetchSectorStocks(sector)}>
                  🚀 {sector}
                </Badge>
              ))}
              {data.capital_inflow.filter(s => !data.hot_sectors.includes(s)).map(sector => (
                <Badge key={sector} variant="outline" className="bg-green-500/20 text-green-400 border-green-500/30 cursor-pointer hover:bg-green-500/30" onClick={() => fetchSectorStocks(sector)}>
                  {sector}
                </Badge>
              ))}
              {data.capital_inflow.length === 0 && <span className="text-muted-foreground text-sm">None</span>}
            </div>
          </CardContent>
        </Card>

        <Card className="border-red-500/30 bg-red-500/5">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <TrendingDown className="h-4 w-4 text-red-500" />
              Capital Outflow Sectors
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {data.capital_outflow.map(sector => (
                <Badge key={sector} variant="outline" className="bg-red-500/20 text-red-400 border-red-500/30 cursor-pointer hover:bg-red-500/30" onClick={() => fetchSectorStocks(sector)}>
                  {sector}
                </Badge>
              ))}
              {data.capital_outflow.length === 0 && <span className="text-muted-foreground text-sm">None</span>}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* RRG Scatter Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span>Relative Rotation Graph (RRG)</span>
              <span className="text-xs text-muted-foreground font-normal">
                Updated: {new Date(data.generated_at).toLocaleString()} • Rebalance: Weekly
              </span>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex gap-2 text-xs">
                {Object.entries(QUADRANT_COLORS).map(([quadrant, color]) => (
                  <span key={quadrant} className="flex items-center gap-1">
                    <span className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                    {quadrant}
                  </span>
                ))}
              </div>
              <Button variant="ghost" size="sm" onClick={fetchSectorRotation} disabled={loading}>
                <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
              <XAxis type="number" dataKey="x" name="RS Ratio" domain={['auto', 'auto']} label={{ value: 'RS Ratio (Relative Strength)', position: 'bottom', offset: 0 }} />
              <YAxis type="number" dataKey="y" name="RS Momentum" domain={['auto', 'auto']} label={{ value: 'RS Momentum', angle: -90, position: 'left' }} />
              <ReferenceLine x={100} stroke="#666" strokeDasharray="5 5" />
              <ReferenceLine y={0} stroke="#666" strokeDasharray="5 5" />
              <Tooltip content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload;
                  return (
                    <div className="bg-background border rounded-lg p-3 shadow-lg">
                      <p className="font-semibold">{d.name} ({d.symbol})</p>
                      <p className="text-sm text-muted-foreground">RS Ratio: {d.x.toFixed(2)}</p>
                      <p className="text-sm text-muted-foreground">RS Momentum: {d.y.toFixed(2)}</p>
                      <Badge className={QUADRANT_BG[d.quadrant]}>{d.quadrant}</Badge>
                      {d.transition && <p className="text-xs text-blue-400 mt-1">⚡ Quadrant Transition</p>}
                      <p className="text-xs text-blue-400 mt-1 cursor-pointer">Click sector row to see stocks →</p>
                    </div>
                  );
                }
                return null;
              }} />
              <Scatter data={chartData}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={QUADRANT_COLORS[entry.quadrant]} stroke={entry.transition ? '#fff' : 'none'} strokeWidth={entry.transition ? 2 : 0} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Sector Table */}
      <Card>
        <CardHeader>
          <CardTitle>Sector Details (Click to expand)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-3">Sector</th>
                  <th className="text-left py-2 px-3">ETF</th>
                  <th className="text-right py-2 px-3">RS Ratio</th>
                  <th className="text-right py-2 px-3">RS Momentum</th>
                  <th className="text-center py-2 px-3">Quadrant</th>
                  <th className="text-center py-2 px-3">Signal</th>
                  <th className="text-center py-2 px-3"></th>
                </tr>
              </thead>
              <tbody>
                {data.sectors.map((sector) => (
                  <tr 
                    key={sector.symbol} 
                    className={`border-b border-muted/50 hover:bg-muted/20 cursor-pointer ${expandedSector === sector.name ? 'bg-blue-500/10' : ''}`}
                    onClick={() => fetchSectorStocks(sector.name)}
                  >
                    <td className="py-2 px-3 font-medium">{sector.name}</td>
                    <td className="py-2 px-3 text-muted-foreground">{sector.symbol}</td>
                    <td className={`py-2 px-3 text-right ${sector.rs_ratio >= 100 ? 'text-green-400' : 'text-red-400'}`}>
                      {sector.rs_ratio.toFixed(2)}
                    </td>
                    <td className={`py-2 px-3 text-right ${sector.rs_momentum >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {sector.rs_momentum >= 0 ? '+' : ''}{sector.rs_momentum.toFixed(2)}
                    </td>
                    <td className="py-2 px-3 text-center">
                      <Badge className={QUADRANT_BG[sector.quadrant]}>{sector.quadrant}</Badge>
                    </td>
                    <td className="py-2 px-3 text-center">
                      {sector.transition_signal ? (
                        <span className="flex items-center justify-center gap-1 text-blue-400">
                          <ArrowRight className="h-3 w-3" /> Transition
                        </span>
                      ) : sector.rs_momentum > 1 ? (
                        <span className="flex items-center justify-center gap-1 text-green-400">
                          <TrendingUp className="h-3 w-3" /> Strong
                        </span>
                      ) : sector.rs_momentum < -1 ? (
                        <span className="flex items-center justify-center gap-1 text-red-400">
                          <TrendingDown className="h-3 w-3" /> Weak
                        </span>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-center">
                      {expandedSector === sector.name ? (
                        <ChevronUp className="h-4 w-4 text-blue-400" />
                      ) : (
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Expanded Sector Stocks Panel - Below Sector Table */}
      <AnimatePresence>
        {expandedSector && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Card className="border-blue-500/30">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    📊 {expandedSector} - Stock Money Flow
                    {sectorStocks && (
                      <span className="text-xs text-muted-foreground font-normal">
                        ({sectorStocks.stock_count} stocks • Avg Flow: {sectorStocks.avg_money_flow > 0 ? '+' : ''}{sectorStocks.avg_money_flow})
                      </span>
                    )}
                  </span>
                  <Button variant="ghost" size="sm" onClick={() => { setExpandedSector(null); setSectorStocks(null); }}>
                    <X className="h-4 w-4" />
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {loadingStocks ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  </div>
                ) : sectorStocks && sectorStocks.stocks.length > 0 ? (
                  <div className="overflow-x-auto max-h-[300px] overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 bg-background">
                        <tr className="border-b">
                          <th className="text-left py-2 px-2">Ticker</th>
                          <th className="text-right py-2 px-2">Rank</th>
                          <th className="text-right py-2 px-2">Score</th>
                          <th className="text-right py-2 px-2">Money Flow</th>
                          <th className="text-right py-2 px-2">MFI</th>
                          <th className="text-right py-2 px-2">OBV Z</th>
                          <th className="text-center py-2 px-2">Signal</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sectorStocks.stocks.map((stock) => (
                          <tr key={stock.ticker} className="border-b border-muted/30 hover:bg-muted/10">
                            <td className="py-1.5 px-2 font-medium">{stock.ticker}</td>
                            <td className="py-1.5 px-2 text-right text-muted-foreground">#{stock.rank}</td>
                            <td className="py-1.5 px-2 text-right">{stock.score.toFixed(2)}</td>
                            <td className={`py-1.5 px-2 text-right ${stock.money_flow >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {stock.money_flow >= 0 ? '+' : ''}{stock.money_flow.toFixed(2)}
                            </td>
                            <td className={`py-1.5 px-2 text-right ${stock.mfi < 20 ? 'text-green-400' : stock.mfi > 80 ? 'text-red-400' : ''}`}>
                              {stock.mfi.toFixed(0)}
                            </td>
                            <td className={`py-1.5 px-2 text-right ${stock.obv_zscore > 1 ? 'text-green-400' : stock.obv_zscore < -1 ? 'text-red-400' : ''}`}>
                              {stock.obv_zscore.toFixed(2)}
                            </td>
                            <td className={`py-1.5 px-2 text-center text-xs ${FLOW_SIGNAL_COLORS[stock.flow_signal] || ''}`}>
                              {stock.flow_signal}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-muted-foreground text-center py-4">No stocks found in this sector</p>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
