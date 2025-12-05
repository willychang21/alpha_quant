import React from 'react';
import { RealTimeTicker } from '@/components/RealTimeTicker';
import { MLSignalCard } from '@/components/MLSignalCard';
import { RiskMonitor } from '@/components/RiskMonitor';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Activity } from 'lucide-react';

const AdvancedDashboard: React.FC = () => {
    return (
        <div className="min-h-screen bg-background text-foreground flex flex-col">
            {/* Top Bar: Real-Time Ticker */}
            <RealTimeTicker />

            <div className="flex-1 p-6 space-y-6">
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">Quant Command Center</h1>
                        <p className="text-muted-foreground">Tier-3 Enterprise System • Live</p>
                    </div>
                    <div className="flex items-center space-x-2">
                        <span className="flex h-3 w-3 relative">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                        </span>
                        <span className="text-sm font-medium text-green-500">System Operational</span>
                    </div>
                </div>

                {/* Main Grid */}
                <div className="grid grid-cols-1 md:grid-cols-12 gap-6 h-[600px]">
                    
                    {/* Left Column: ML Signals (4 cols) */}
                    <div className="md:col-span-4 h-full">
                        <MLSignalCard />
                    </div>

                    {/* Middle Column: Market Overview / Execution (4 cols) */}
                    <div className="md:col-span-4 h-full flex flex-col space-y-6">
                        {/* Execution Monitor Placeholder */}
                        <Card className="flex-1 border-l-4 border-l-blue-500">
                            <CardHeader>
                                <div className="flex items-center space-x-2">
                                    <Activity className="w-5 h-5 text-blue-500" />
                                    <CardTitle>Execution Algo (VWAP)</CardTitle>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-4">
                                    <div className="flex justify-between text-sm">
                                        <span>Progress</span>
                                        <span>45%</span>
                                    </div>
                                    <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
                                        <div className="bg-blue-500 h-full w-[45%] animate-pulse"></div>
                                    </div>
                                    <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground mt-4">
                                        <div>Target: 10,000 shrs</div>
                                        <div>Filled: 4,500 shrs</div>
                                        <div>Avg Price: $152.45</div>
                                        <div>Slippage: 0.5 bps</div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                        
                        {/* System Health / Logs Placeholder */}
                        <Card className="h-1/3">
                            <CardHeader className="py-3">
                                <CardTitle className="text-sm">System Logs</CardTitle>
                            </CardHeader>
                            <CardContent className="text-xs font-mono text-muted-foreground space-y-1">
                                <p>[20:05:12] Connected to WebSocket stream.</p>
                                <p>[20:05:15] Received 12 new ticks.</p>
                                <p>[20:05:18] ML Model updated (v2.1).</p>
                                <p>[20:05:20] Risk check passed.</p>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Right Column: Risk Management (4 cols) */}
                    <div className="md:col-span-4 h-full">
                        <RiskMonitor />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AdvancedDashboard;
