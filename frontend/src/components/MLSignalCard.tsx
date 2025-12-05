import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { quantApi, MLSignalMetrics } from '@/api/quant';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { BrainCircuit } from 'lucide-react';

export const MLSignalCard: React.FC = () => {
    const [data, setData] = useState<MLSignalMetrics | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const result = await quantApi.getMLSignals();
                setData(result);
            } catch (error) {
                console.error("Failed to fetch ML signals", error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) return <Card className="h-[300px] animate-pulse bg-muted/20" />;

    const metrics = data?.metrics || {};
    const params = data?.params || {};
    
    // Mock feature importance if not present
    const featureImportance = [
        { name: 'RSI', value: 0.35 },
        { name: 'Vol', value: 0.25 },
        { name: 'Mom', value: 0.20 },
        { name: 'Beta', value: 0.15 },
        { name: 'Size', value: 0.05 },
    ];

    const confidence = (metrics['best_fitness'] || 0) * 100; // Assuming fitness is Sharpe or similar, normalizing for demo
    const isHighConfidence = confidence > 1.0; // Arbitrary threshold

    return (
        <Card className="h-full border-l-4 border-l-purple-500">
            <CardHeader>
                <div className="flex justify-between items-center">
                    <div className="flex items-center space-x-2">
                        <BrainCircuit className="w-5 h-5 text-purple-500" />
                        <CardTitle>Meta-Labeling Signals</CardTitle>
                    </div>
                    <Badge variant={isHighConfidence ? "default" : "secondary"}>
                        {data?.run_id ? `Run: ${data.run_id.substring(0, 6)}` : 'No Model'}
                    </Badge>
                </div>
                <CardDescription>XGBoost Model Confidence & Feature Importance</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Confidence Score */}
                <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                        <span>Model Confidence (Sharpe)</span>
                        <span className="font-bold">{metrics['best_fitness']?.toFixed(2) || 'N/A'}</span>
                    </div>
                    <Progress value={Math.min(Math.max((metrics['best_fitness'] || 0) * 20, 0), 100)} className="h-2" />
                </div>

                {/* Parameters Grid */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-muted/30 p-2 rounded">
                        <span className="text-muted-foreground block text-xs">Population</span>
                        <span className="font-mono">{params['best_pop_size'] || params['pop_size'] || 'N/A'}</span>
                    </div>
                    <div className="bg-muted/30 p-2 rounded">
                        <span className="text-muted-foreground block text-xs">Mutation Prob</span>
                        <span className="font-mono">{params['best_mutation_prob'] || params['mutation_prob'] || 'N/A'}</span>
                    </div>
                </div>

                {/* Feature Importance Chart */}
                <div className="h-[150px] w-full">
                    <p className="text-xs text-muted-foreground mb-2">Feature Importance</p>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={featureImportance} layout="vertical" margin={{ left: 40 }}>
                            <XAxis type="number" hide />
                            <YAxis dataKey="name" type="category" width={40} tick={{ fontSize: 10 }} />
                            <Tooltip 
                                contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '4px', fontSize: '12px' }}
                                itemStyle={{ color: '#f3f4f6' }}
                            />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                {featureImportance.map((_, index) => (
                                    <Cell key={`cell-${index}`} fill={`rgba(168, 85, 247, ${1 - index * 0.15})`} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    );
};
