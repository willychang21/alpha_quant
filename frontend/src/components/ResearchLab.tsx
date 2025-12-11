import React, { useEffect, useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { getLatestSignals, Signal } from '../api/signals';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { FlaskConical, Filter, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';

// Factor breakdown component for each signal
const FactorBreakdown: React.FC<{ signal: Signal }> = ({ signal }) => {
    const meta = JSON.parse(signal.metadata_json || '{}');
    
    const factors = [
        { name: 'VSM', value: meta.z_vsm ?? meta.vsm ?? 0, color: '#22d3ee' },
        { name: 'BAB', value: meta.z_bab ?? meta.bab ?? 0, color: '#a855f7' },
        { name: 'QMJ', value: meta.z_qmj ?? meta.qmj ?? 0, color: '#eab308' },
        { name: 'Upside', value: meta.z_upside ?? (meta.upside ?? 0) * 10, color: '#22c55e' },
    ];

    return (
        <div className="h-[60px] w-[150px]">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={factors} layout="vertical" margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                    <XAxis type="number" hide domain={[-3, 3]} />
                    <YAxis type="category" dataKey="name" width={35} tick={{ fontSize: 10 }} />
                    <Bar dataKey="value" radius={2}>
                        {factors.map((entry, index) => (
                            <Cell 
                                key={`cell-${index}`} 
                                fill={entry.value >= 0 ? entry.color : '#ef4444'} 
                                fillOpacity={0.8}
                            />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

const ResearchLab: React.FC = () => {
    const [signals, setSignals] = useState<Signal[]>([]);
    const [loading, setLoading] = useState(true);
    
    // Filtering State
    const [minScore, setMinScore] = useState(0);
    const [showFilters, setShowFilters] = useState(true);
    
    // Pagination State
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 15;
    
    // Sorting State
    const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' }>({
        key: 'score',
        direction: 'desc'
    });

    // Expanded row for factor breakdown
    const [expandedRow, setExpandedRow] = useState<number | null>(null);

    useEffect(() => {
        const fetchSignals = async () => {
            try {
                // Use getLatestSignals to prevent duplicates from historical data
                const data = await getLatestSignals('ranking_v3', 1000);
                setSignals(data);
            } catch (error) {
                console.error("Failed to fetch signals:", error);
            } finally {
                setLoading(false);
            }
        };
        fetchSignals();
    }, []);

    const getScoreColor = (score: number) => {
        if (score >= 2.0) return "bg-green-600";
        if (score >= 1.0) return "bg-green-400";
        if (score >= 0) return "bg-amber-400";
        if (score >= -1.0) return "bg-red-400";
        return "bg-red-600";
    };

    const getConviction = (score: number) => {
        if (score >= 2.0) return { label: 'High', color: 'bg-green-500/20 text-green-400 border-green-500/50' };
        if (score >= 1.0) return { label: 'Medium', color: 'bg-amber-500/20 text-amber-400 border-amber-500/50' };
        return { label: 'Low', color: 'bg-gray-500/20 text-gray-400 border-gray-500/50' };
    };

    // Apply filtering and sorting
    const filteredSignals = useMemo(() => {
        let result = signals.filter(s => s.score >= minScore);
        
        result.sort((a, b) => {
            let aValue: any = a[sortConfig.key as keyof Signal];
            let bValue: any = b[sortConfig.key as keyof Signal];
            
            // Handle nested metadata
            if (['vsm', 'bab', 'qmj', 'upside'].includes(sortConfig.key)) {
                const metaA = JSON.parse(a.metadata_json || '{}');
                const metaB = JSON.parse(b.metadata_json || '{}');
                aValue = metaA[sortConfig.key] ?? -999;
                bValue = metaB[sortConfig.key] ?? -999;
            }
            
            if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
            if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
            return 0;
        });
        
        return result;
    }, [signals, minScore, sortConfig]);

    // Stats
    const highConvictionCount = signals.filter(s => s.score >= 2.0).length;
    const avgScore = signals.length > 0 
        ? (signals.reduce((sum, s) => sum + s.score, 0) / signals.length).toFixed(2) 
        : '0.00';

    // Pagination
    const totalPages = Math.ceil(filteredSignals.length / itemsPerPage);
    const currentSignals = filteredSignals.slice(
        (currentPage - 1) * itemsPerPage,
        currentPage * itemsPerPage
    );

    const requestSort = (key: string) => {
        setSortConfig(prev => ({
            key,
            direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc'
        }));
    };

    const SortIcon: React.FC<{ column: string }> = ({ column }) => {
        if (sortConfig.key !== column) return null;
        return sortConfig.direction === 'desc' 
            ? <ChevronDown className="w-4 h-4 inline ml-1" />
            : <ChevronUp className="w-4 h-4 inline ml-1" />;
    };

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-between items-center"
            >
                <div>
                    <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
                        <FlaskConical className="w-8 h-8 text-purple-500" />
                        Research Lab
                    </h1>
                    <p className="text-muted-foreground mt-1">
                        Factor Research & Signal Analysis
                    </p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="text-center px-4 py-2 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-green-500">{highConvictionCount}</div>
                        <div className="text-xs text-muted-foreground">High Conviction</div>
                    </div>
                    <div className="text-center px-4 py-2 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-mono font-bold">{avgScore}</div>
                        <div className="text-xs text-muted-foreground">Avg Z-Score</div>
                    </div>
                </div>
            </motion.div>

            {/* Filters Card */}
            <Card className="border-l-4 border-l-purple-500">
                <CardHeader className="py-3 cursor-pointer" onClick={() => setShowFilters(!showFilters)}>
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <Filter className="w-4 h-4 text-purple-500" />
                            <CardTitle className="text-sm">Filters</CardTitle>
                        </div>
                        {showFilters ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </div>
                </CardHeader>
                {showFilters && (
                    <CardContent className="pt-0">
                        <div className="flex items-center gap-8">
                            <div className="flex-1 max-w-md">
                                <label className="text-sm text-muted-foreground mb-2 block">
                                    Minimum Score: <span className="font-bold text-foreground">{minScore.toFixed(1)}</span>
                                </label>
                                <Slider
                                    value={[minScore]}
                                    onValueChange={([val]) => { setMinScore(val); setCurrentPage(1); }}
                                    min={-2}
                                    max={4}
                                    step={0.5}
                                    className="w-full"
                                />
                                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                                    <span>-2.0 (Sell)</span>
                                    <span>0 (Neutral)</span>
                                    <span>2.0+ (Strong Buy)</span>
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <Button 
                                    variant={minScore >= 2 ? "default" : "outline"} 
                                    size="sm"
                                    onClick={() => { setMinScore(2); setCurrentPage(1); }}
                                >
                                    High Conviction Only
                                </Button>
                                <Button 
                                    variant="ghost" 
                                    size="sm"
                                    onClick={() => { setMinScore(0); setCurrentPage(1); }}
                                >
                                    Reset
                                </Button>
                            </div>
                        </div>
                    </CardContent>
                )}
            </Card>

            {/* Signals Table */}
            <Card>
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle className="flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-cyan-500" />
                                Signal Analysis
                            </CardTitle>
                            <CardDescription>
                                Showing {filteredSignals.length} of {signals.length} signals
                            </CardDescription>
                        </div>
                    </div>
                </CardHeader>
                <CardContent>
                    {loading ? (
                        <div className="text-center py-10">Loading signals...</div>
                    ) : (
                        <div className="space-y-4">
                            <div className="rounded-md border border-border overflow-hidden">
                                <Table>
                                    <TableHeader className="bg-muted/50">
                                        <TableRow>
                                            <TableHead className="w-[80px] cursor-pointer hover:text-primary" onClick={() => requestSort('ticker')}>
                                                Ticker <SortIcon column="ticker" />
                                            </TableHead>
                                            <TableHead className="text-center cursor-pointer hover:text-primary" onClick={() => requestSort('score')}>
                                                Score <SortIcon column="score" />
                                            </TableHead>
                                            <TableHead className="text-center">Conviction</TableHead>
                                            <TableHead className="text-center">Factor Breakdown</TableHead>
                                            <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => requestSort('vsm')}>
                                                VSM <SortIcon column="vsm" />
                                            </TableHead>
                                            <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => requestSort('bab')}>
                                                BAB <SortIcon column="bab" />
                                            </TableHead>
                                            <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => requestSort('qmj')}>
                                                QMJ <SortIcon column="qmj" />
                                            </TableHead>
                                            <TableHead className="text-right cursor-pointer hover:text-primary" onClick={() => requestSort('upside')}>
                                                Upside <SortIcon column="upside" />
                                            </TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {currentSignals.map((signal) => {
                                            const meta = JSON.parse(signal.metadata_json || '{}');
                                            const vsm = meta.vsm ?? meta.momentum ?? 0;
                                            const bab = meta.bab ?? 0;
                                            const qmj = meta.qmj ?? 0;
                                            const upside = meta.upside ?? meta.earnings_yield ?? 0;
                                            const conviction = getConviction(signal.score);
                                            
                                            return (
                                                <TableRow 
                                                    key={signal.id} 
                                                    className="hover:bg-muted/50 transition-colors"
                                                >
                                                    <TableCell className="font-bold">{signal.ticker}</TableCell>
                                                    <TableCell className="text-center">
                                                        <span className={`px-3 py-1 rounded text-white text-sm font-bold ${getScoreColor(signal.score)}`}>
                                                            {signal.score >= 0 ? '+' : ''}{signal.score.toFixed(2)}z
                                                        </span>
                                                    </TableCell>
                                                    <TableCell className="text-center">
                                                        <Badge variant="outline" className={conviction.color}>
                                                            {conviction.label}
                                                        </Badge>
                                                    </TableCell>
                                                    <TableCell>
                                                        <FactorBreakdown signal={signal} />
                                                    </TableCell>
                                                    <TableCell className={`text-right font-mono ${vsm > 0 ? 'text-cyan-400' : 'text-red-400'}`}>
                                                        {vsm.toFixed(2)}
                                                    </TableCell>
                                                    <TableCell className={`text-right font-mono ${bab > 0 ? 'text-purple-400' : 'text-red-400'}`}>
                                                        {bab.toFixed(2)}
                                                    </TableCell>
                                                    <TableCell className={`text-right font-mono ${qmj > 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                                                        {qmj.toFixed(2)}
                                                    </TableCell>
                                                    <TableCell className={`text-right font-mono ${upside > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                        {(upside * 100).toFixed(1)}%
                                                    </TableCell>
                                                </TableRow>
                                            );
                                        })}
                                        {filteredSignals.length === 0 && (
                                            <TableRow>
                                                <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">
                                                    No signals match the filter criteria.
                                                </TableCell>
                                            </TableRow>
                                        )}
                                    </TableBody>
                                </Table>
                            </div>

                            {/* Pagination */}
                            <div className="flex items-center justify-between px-2">
                                <div className="text-sm text-muted-foreground">
                                    Page {currentPage} of {totalPages || 1}
                                </div>
                                <div className="flex gap-2">
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                        disabled={currentPage === 1}
                                    >
                                        Previous
                                    </Button>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                        disabled={currentPage >= totalPages}
                                    >
                                        Next
                                    </Button>
                                </div>
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
};

export default ResearchLab;
