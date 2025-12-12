/**
 * ML Alpha Dashboard Component
 * 
 * Displays comprehensive ML enhancement status and metrics:
 * - Regime indicator with animated status (Bull/Bear)
 * - Traditional vs ML score breakdown
 * - Active ML modules status
 * - SHAP factor attribution chart
 * 
 * @module components/MLAlphaDashboard
 * @state Local state managed with useState hooks
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { 
    BrainCircuit, TrendingUp, TrendingDown, Activity, 
    CheckCircle2, XCircle, Zap, AlertTriangle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// =============================================================================
// Types & Interfaces
// =============================================================================

/** ML Alpha status from backend API */
export interface MLAlphaStatus {
    regime: 'Bull' | 'Bear' | 'Unknown';
    regimeConfidence: number;
    activeModules: string[];
    lastUpdated: string | null;
    scoringBreakdown: {
        traditionalWeight: number;
        mlWeight: number;
        avgMlContribution: number;
    };
    featureAttribution?: FactorAttribution[];
}

/** SHAP-like factor contribution */
interface FactorAttribution {
    factor: string;
    contribution: number;
    direction: 'positive' | 'negative';
}

/** Props for RegimeIndicator */
interface RegimeIndicatorProps {
    regime: 'Bull' | 'Bear' | 'Unknown';
    confidence: number;
}

/** Props for ModuleStatusBadge */
interface ModuleStatusBadgeProps {
    name: string;
    active: boolean;
}

/** Props for ScoreBreakdownCard */
interface ScoreBreakdownProps {
    traditionalWeight: number;
    mlWeight: number;
    avgContribution: number;
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Animated regime indicator with Bull/Bear status
 */
const RegimeIndicator: React.FC<RegimeIndicatorProps> = ({ regime, confidence }) => {
    const isBull = regime === 'Bull';
    const isBear = regime === 'Bear';
    
    const bgColor = isBull 
        ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20' 
        : isBear 
        ? 'bg-gradient-to-r from-red-500/20 to-rose-500/20'
        : 'bg-gradient-to-r from-gray-500/20 to-slate-500/20';
    
    const textColor = isBull ? 'text-green-400' : isBear ? 'text-red-400' : 'text-gray-400';
    const Icon = isBull ? TrendingUp : isBear ? TrendingDown : Activity;
    
    return (
        <motion.div
            className={`flex items-center gap-3 px-4 py-3 rounded-lg ${bgColor}`}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
        >
            <motion.div
                animate={{ 
                    scale: [1, 1.1, 1],
                    opacity: [0.8, 1, 0.8] 
                }}
                transition={{ 
                    duration: 2, 
                    repeat: Infinity,
                    ease: "easeInOut"
                }}
            >
                <Icon className={`w-8 h-8 ${textColor}`} />
            </motion.div>
            <div className="flex flex-col">
                <span className={`text-lg font-bold ${textColor}`}>
                    {regime} Market
                </span>
                <span className="text-xs text-muted-foreground">
                    Confidence: {(confidence * 100).toFixed(0)}%
                </span>
            </div>
            <div className="ml-auto">
                <motion.div
                    className={`w-3 h-3 rounded-full ${isBull ? 'bg-green-500' : isBear ? 'bg-red-500' : 'bg-gray-500'}`}
                    animate={{ 
                        boxShadow: [
                            `0 0 0 0 ${isBull ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`,
                            `0 0 0 8px ${isBull ? 'rgba(34, 197, 94, 0)' : 'rgba(239, 68, 68, 0)'}`,
                        ]
                    }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                />
            </div>
        </motion.div>
    );
};

/**
 * Badge showing ML module activation status
 */
const ModuleStatusBadge: React.FC<ModuleStatusBadgeProps> = ({ name, active }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
            className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm ${
                active 
                    ? 'bg-green-500/10 text-green-400 border border-green-500/30' 
                    : 'bg-muted/30 text-muted-foreground border border-muted'
            }`}
        >
            {active ? (
                <CheckCircle2 className="w-4 h-4" />
            ) : (
                <XCircle className="w-4 h-4" />
            )}
            <span className="font-medium">{name}</span>
        </motion.div>
    );
};

/**
 * Score breakdown visualization with progress bars
 */
const ScoreBreakdown: React.FC<ScoreBreakdownProps> = ({ 
    traditionalWeight, 
    mlWeight, 
    avgContribution 
}) => {
    const isPositive = avgContribution >= 0;
    
    return (
        <div className="grid grid-cols-3 gap-4">
            <div className="bg-muted/20 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-blue-400">
                    {(traditionalWeight * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-muted-foreground mt-1">Traditional</div>
                <Progress value={traditionalWeight * 100} className="h-1.5 mt-2" />
            </div>
            
            <div className="bg-muted/20 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold text-purple-400">
                    {(mlWeight * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-muted-foreground mt-1">ML Blend</div>
                <Progress value={mlWeight * 100} className="h-1.5 mt-2" />
            </div>
            
            <div className="bg-muted/20 rounded-lg p-4 text-center">
                <div className={`text-2xl font-bold flex items-center justify-center gap-1 ${
                    isPositive ? 'text-green-400' : 'text-red-400'
                }`}>
                    {isPositive ? '+' : ''}{avgContribution.toFixed(3)}
                    {isPositive ? (
                        <TrendingUp className="w-4 h-4" />
                    ) : (
                        <TrendingDown className="w-4 h-4" />
                    )}
                </div>
                <div className="text-xs text-muted-foreground mt-1">Avg ML Δ</div>
            </div>
        </div>
    );
};

// =============================================================================
// Main Component
// =============================================================================

/**
 * ML Alpha Dashboard - Main component
 * 
 * Displays comprehensive ML enhancement status including:
 * - Market regime with confidence
 * - Score attribution breakdown
 * - Active ML modules
 * - Factor contribution chart
 */
export const MLAlphaDashboard: React.FC = () => {
    const [data, setData] = useState<MLAlphaStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Fetch ML status from backend
    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                setError(null);
                
                // Import API dynamically to avoid circular deps
                const { quantApi } = await import('@/api/quant');
                const result = await quantApi.getMLAlphaStatus();
                setData(result);
                
            } catch (err) {
                console.error("Failed to fetch ML alpha status:", err);
                setError("Failed to load ML Enhancement data");
            } finally {
                setLoading(false);
            }
        };
        
        fetchData();
        
        // Refresh every 60 seconds
        const interval = setInterval(fetchData, 60000);
        return () => clearInterval(interval);
    }, []);

    // All possible ML modules
    const allModules = useMemo(() => [
        'SHAP', 'ResidualAlpha', 'ConstrainedGBM', 'OnlineLearning', 'SupplyChain'
    ], []);

    // Prepare chart data with colors
    const chartData = useMemo(() => {
        if (!data?.featureAttribution) return [];
        
        return data.featureAttribution
            .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
            .map(item => ({
                name: item.factor,
                value: item.contribution,
                fill: item.direction === 'positive' ? '#22c55e' : '#ef4444'
            }));
    }, [data]);

    // Loading skeleton
    if (loading) {
        return (
            <Card className="h-full border-l-4 border-l-purple-500">
                <CardHeader>
                    <div className="flex items-center gap-2">
                        <div className="h-5 w-5 rounded bg-muted/50 animate-pulse" />
                        <div className="h-6 w-48 rounded bg-muted/50 animate-pulse" />
                    </div>
                </CardHeader>
                <CardContent className="space-y-6">
                    <div className="h-16 w-full rounded-lg bg-muted/30 animate-pulse" />
                    <div className="grid grid-cols-3 gap-4">
                        <div className="h-24 rounded-lg bg-muted/30 animate-pulse" />
                        <div className="h-24 rounded-lg bg-muted/30 animate-pulse" />
                        <div className="h-24 rounded-lg bg-muted/30 animate-pulse" />
                    </div>
                    <div className="h-32 w-full rounded bg-muted/30 animate-pulse" />
                </CardContent>
            </Card>
        );
    }

    // Error state
    if (error) {
        return (
            <Card className="h-full border-l-4 border-l-red-500">
                <CardContent className="flex flex-col items-center justify-center h-full py-12">
                    <AlertTriangle className="w-12 h-12 text-red-400 mb-4" />
                    <p className="text-red-400 font-medium">{error}</p>
                    <button 
                        onClick={() => window.location.reload()}
                        className="mt-4 px-4 py-2 bg-red-500/20 text-red-400 rounded-md hover:bg-red-500/30 transition"
                    >
                        Retry
                    </button>
                </CardContent>
            </Card>
        );
    }

    // Empty state
    if (!data) {
        return (
            <Card className="h-full border-l-4 border-l-gray-500">
                <CardContent className="flex flex-col items-center justify-center h-full py-12">
                    <BrainCircuit className="w-12 h-12 text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No ML Enhancement data available</p>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="h-full border-l-4 border-l-purple-500">
            <CardHeader className="pb-2">
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <motion.div
                            animate={{ rotate: [0, 360] }}
                            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                        >
                            <BrainCircuit className="w-5 h-5 text-purple-500" />
                        </motion.div>
                        <CardTitle className="text-lg">ML Alpha Enhancement</CardTitle>
                    </div>
                    <Badge variant="outline" className="text-xs">
                        <Zap className="w-3 h-3 mr-1" />
                        {data.activeModules.length} Active
                    </Badge>
                </div>
                <CardDescription className="text-xs">
                    Last updated: {data.lastUpdated ? new Date(data.lastUpdated).toLocaleTimeString() : 'N/A'}
                </CardDescription>
            </CardHeader>
            
            <CardContent className="space-y-5">
                {/* Regime Indicator */}
                <RegimeIndicator 
                    regime={data.regime} 
                    confidence={data.regimeConfidence} 
                />
                
                {/* Score Breakdown */}
                <div className="space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">Scoring Weights</h3>
                    <ScoreBreakdown 
                        traditionalWeight={data.scoringBreakdown.traditionalWeight}
                        mlWeight={data.scoringBreakdown.mlWeight}
                        avgContribution={data.scoringBreakdown.avgMlContribution}
                    />
                </div>
                
                {/* Active Modules */}
                <div className="space-y-2">
                    <h3 className="text-sm font-medium text-muted-foreground">ML Modules</h3>
                    <div className="flex flex-wrap gap-2">
                        <AnimatePresence>
                            {allModules.map((module) => (
                                <ModuleStatusBadge
                                    key={module}
                                    name={module}
                                    active={data.activeModules.includes(module)}
                                />
                            ))}
                        </AnimatePresence>
                    </div>
                </div>
                
                {/* Factor Attribution Chart */}
                {chartData.length > 0 && (
                    <div className="space-y-2">
                        <h3 className="text-sm font-medium text-muted-foreground">
                            Factor Attribution (SHAP)
                        </h3>
                        <div className="h-[160px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart 
                                    data={chartData} 
                                    layout="vertical" 
                                    margin={{ left: 60, right: 20, top: 5, bottom: 5 }}
                                >
                                    <XAxis 
                                        type="number" 
                                        domain={['dataMin', 'dataMax']}
                                        tickFormatter={(v) => v.toFixed(2)}
                                        tick={{ fontSize: 10 }}
                                    />
                                    <YAxis 
                                        dataKey="name" 
                                        type="category" 
                                        width={55} 
                                        tick={{ fontSize: 11 }}
                                    />
                                    <Tooltip 
                                        contentStyle={{ 
                                            backgroundColor: '#1f2937', 
                                            border: 'none', 
                                            borderRadius: '6px',
                                            fontSize: '12px'
                                        }}
                                        itemStyle={{ color: '#f3f4f6' }}
                                        formatter={(value: number) => [value.toFixed(3), 'Contribution']}
                                    />
                                    <Bar 
                                        dataKey="value" 
                                        radius={[0, 4, 4, 0]}
                                    >
                                        {chartData.map((entry, index) => (
                                            <Cell 
                                                key={`cell-${index}`} 
                                                fill={entry.fill}
                                                fillOpacity={0.85}
                                            />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

export default MLAlphaDashboard;
