import React, { useEffect, useState } from 'react';
import { RealTimeTicker } from '@/components/RealTimeTicker';
import { RegimeBanner } from '@/components/RegimeBanner';
import { FactorRadar } from '@/components/FactorRadar';
import { SmartTradeTable } from '@/components/SmartTradeTable';
import { RiskMonitor } from '@/components/RiskMonitor';
import { SectorRotation } from '@/components/SectorRotation';
import { MLAlphaDashboard } from '@/components/MLAlphaDashboard';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Activity, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface DashboardSummary {
    regime: {
        state: 'AGGRESSIVE' | 'NEUTRAL' | 'DEFENSIVE';
        confidence_score: number;
        confidence_multiplier: number;
        base_target_vol: number;
        adjusted_target_vol: number;
    };
    attribution: {
        value: number;
        momentum: number;
        quality: number;
        low_risk: number;
        sentiment: number;
    };
    trades: Array<{
        ticker: string;
        name: string | null;
        sector: string;
        alpha_score: number;
        conviction: 'High' | 'Medium' | 'Low';
        raw_weight: number;
        final_weight: number;
        shares: number;
        value: number;
        reason: string;
    }>;
    generated_at: string;
}

const AdvancedDashboard: React.FC = () => {
    const [data, setData] = useState<DashboardSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchData = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch('/api/quant/dashboard/summary');
            if (!response.ok) throw new Error('Failed to fetch dashboard data');
            const result = await response.json();
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    if (loading) {
        return (
            <div className="min-h-screen bg-background flex items-center justify-center">
                <div className="text-center space-y-4">
                    <RefreshCw className="w-12 h-12 animate-spin text-primary mx-auto" />
                    <p className="text-muted-foreground">Loading Quant Command Center...</p>
                </div>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="min-h-screen bg-background flex items-center justify-center">
                <Card className="max-w-md">
                    <CardContent className="p-6 text-center space-y-4">
                        <p className="text-destructive">{error || 'No data available'}</p>
                        <Button onClick={fetchData}>Retry</Button>
                    </CardContent>
                </Card>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background text-foreground flex flex-col">
            {/* Top Bar: Real-Time Ticker */}
            <RealTimeTicker />

            <div className="flex-1 p-6 space-y-6">
                {/* Header with Refresh */}
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">Quant Command Center 2.0</h1>
                        <p className="text-muted-foreground">
                            Institutional Decision Support • Last updated: {new Date(data.generated_at).toLocaleTimeString()}
                        </p>
                    </div>
                    <Button variant="outline" onClick={fetchData} disabled={loading}>
                        <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                        Refresh
                    </Button>
                </div>

                {/* Regime HUD */}
                <RegimeBanner regime={data.regime} />

                {/* Main Grid: Factor Attribution + ML Dashboard + Risk Monitor */}
                <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
                    {/* Left: Factor Attribution (4 cols) */}
                    <div className="md:col-span-4">
                        <FactorRadar attribution={data.attribution} />
                    </div>

                    {/* Center: ML Alpha Dashboard (4 cols) */}
                    <div className="md:col-span-4">
                        <MLAlphaDashboard />
                    </div>

                    {/* Right: Risk Monitor (4 cols) */}
                    <div className="md:col-span-4">
                        <RiskMonitor />
                    </div>
                </div>

                {/* Sector Rotation - Capital Flow Detection */}
                <SectorRotation />

                {/* Smart Execution Table */}
                <SmartTradeTable 
                    trades={data.trades} 
                    confidenceMultiplier={data.regime.confidence_multiplier}
                />

                {/* System Logs */}
                <Card className="h-[150px]">
                    <CardHeader className="py-3">
                        <div className="flex items-center space-x-2">
                            <Activity className="w-4 h-4 text-blue-500" />
                            <CardTitle className="text-sm">System Activity</CardTitle>
                        </div>
                    </CardHeader>
                    <CardContent className="text-xs font-mono text-muted-foreground space-y-1 overflow-y-auto max-h-[80px]">
                        <p>[{new Date().toLocaleTimeString()}] Dashboard summary fetched successfully.</p>
                        <p>[{new Date().toLocaleTimeString()}] Regime detected: {data.regime.state}</p>
                        <p>[{new Date().toLocaleTimeString()}] {data.trades.length} positions loaded.</p>
                        <p>[{new Date().toLocaleTimeString()}] Confidence multiplier: {data.regime.confidence_multiplier}x</p>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default AdvancedDashboard;
