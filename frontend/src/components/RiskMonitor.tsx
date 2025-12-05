import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { quantApi, RiskMetrics } from '@/api/quant';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { ShieldCheck, TrendingDown } from 'lucide-react';

export const RiskMonitor: React.FC = () => {
    const [data, setData] = useState<RiskMetrics | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const result = await quantApi.getRiskMetrics();
                setData(result);
            } catch (error) {
                console.error("Failed to fetch Risk metrics", error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) return <Card className="h-[300px] animate-pulse bg-muted/20" />;

    const varData = data?.var || { portfolio_var: 0, component_var: [], weights: [] };
    const hedgeData = data?.hedge || { total_cost: 0, hedge_ratio: 0, contracts: 0 };

    // Prepare Pie Chart Data
    const pieData = [
        { name: 'Equity (SPY)', value: Math.abs(varData.component_var[0] || 60) },
        { name: 'Bonds (AGG)', value: Math.abs(varData.component_var[1] || 40) },
    ];
    
    const COLORS = ['#ef4444', '#3b82f6']; // Red for Equity Risk, Blue for Bonds

    return (
        <Card className="h-full border-l-4 border-l-red-500">
            <CardHeader>
                <div className="flex justify-between items-center">
                    <div className="flex items-center space-x-2">
                        <ShieldCheck className="w-5 h-5 text-red-500" />
                        <CardTitle>Risk Management</CardTitle>
                    </div>
                    <div className="text-right">
                        <span className="text-xs text-muted-foreground block">Portfolio VaR (95%)</span>
                        <span className="font-bold text-red-500">
                            ${varData.portfolio_var.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </span>
                    </div>
                </div>
                <CardDescription>Component VaR & Tail Hedging</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                
                {/* VaR Decomposition */}
                <div className="h-[180px] w-full relative">
                    <p className="text-xs text-muted-foreground absolute top-0 left-0">Risk Contribution</p>
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={pieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={40}
                                outerRadius={60}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {pieData.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip 
                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '4px', fontSize: '12px' }}
                                itemStyle={{ color: '#f3f4f6' }}
                            />
                            <Legend verticalAlign="bottom" height={36} iconSize={8} wrapperStyle={{ fontSize: '12px' }}/>
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* Tail Hedge Cost */}
                <div className="bg-muted/30 p-3 rounded-lg flex justify-between items-center">
                    <div className="flex items-center space-x-2">
                        <TrendingDown className="w-4 h-4 text-yellow-500" />
                        <div>
                            <span className="text-sm font-medium block">Tail Hedge Cost</span>
                            <span className="text-xs text-muted-foreground">Put Options (OTM)</span>
                        </div>
                    </div>
                    <div className="text-right">
                        <span className="block font-bold">${hedgeData.total_cost.toFixed(2)}</span>
                        <span className="text-xs text-muted-foreground">{hedgeData.contracts} Contracts</span>
                    </div>
                </div>

            </CardContent>
        </Card>
    );
};
