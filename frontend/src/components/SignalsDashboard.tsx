import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { getSignals, Signal } from '../api/signals';
import { motion } from 'framer-motion';

const SignalsDashboard: React.FC = () => {
    const [signals, setSignals] = useState<Signal[]>([]);
    const [loading, setLoading] = useState(true);
    
    // Pagination State
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 20;
    
    // Sorting State
    const [sortConfig, setSortConfig] = useState<{ key: keyof Signal | 'momentum' | 'valuation' | 'vsm' | 'bab' | 'qmj' | 'upside' | 'pead' | 'sentiment'; direction: 'asc' | 'desc' }>({
        key: 'score',
        direction: 'desc'
    });

    useEffect(() => {
        const fetchSignals = async () => {
            try {
                const data = await getSignals({ limit: 1000 }); // Fetch all for client-side sorting
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
        if (score > 1.5) return "bg-green-500";
        if (score > 0.5) return "bg-green-300";
        if (score < -1.5) return "bg-red-500";
        if (score < -0.5) return "bg-red-300";
        return "bg-gray-400";
    };

    // Sorting Logic
    const sortedSignals = React.useMemo(() => {
        let sortableItems = [...signals];
        if (sortConfig !== null) {
            sortableItems.sort((a, b) => {
                let aValue: any = a[sortConfig.key as keyof Signal];
                let bValue: any = b[sortConfig.key as keyof Signal];
                
                // Handle nested metadata for sorting
                const metaA = JSON.parse(a.metadata_json || '{}');
                const metaB = JSON.parse(b.metadata_json || '{}');
                
                if (['vsm', 'bab', 'qmj', 'upside', 'momentum', 'valuation'].includes(sortConfig.key as string)) {
                    switch (sortConfig.key) {
                        case 'vsm':
                            aValue = metaA.vsm ?? metaA.momentum ?? -999;
                            bValue = metaB.vsm ?? metaB.momentum ?? -999;
                            break;
                        case 'bab':
                            aValue = metaA.bab ?? -999;
                            bValue = metaB.bab ?? -999;
                            break;
                        case 'qmj':
                            aValue = metaA.qmj ?? -999;
                            bValue = metaB.qmj ?? -999;
                            break;
                        case 'upside':
                        case 'valuation':
                            aValue = metaA.upside ?? metaA.earnings_yield ?? -999;
                            bValue = metaB.upside ?? metaB.earnings_yield ?? -999;
                            break;
                        case 'momentum': // Legacy support
                             aValue = metaA.momentum ?? metaA.vsm ?? -999;
                             bValue = metaB.momentum ?? metaB.vsm ?? -999;
                             break;
                        case 'pead':
                             aValue = metaA.pead ?? -999;
                             bValue = metaB.pead ?? -999;
                             break;
                        case 'sentiment':
                             aValue = metaA.sentiment ?? -999;
                             bValue = metaB.sentiment ?? -999;
                             break;
                    }
                }

                if (aValue < bValue) {
                    return sortConfig.direction === 'asc' ? -1 : 1;
                }
                if (aValue > bValue) {
                    return sortConfig.direction === 'asc' ? 1 : -1;
                }
                return 0;
            });
        }
        return sortableItems;
    }, [signals, sortConfig]);

    // Pagination Logic
    const totalPages = Math.ceil(sortedSignals.length / itemsPerPage);
    const currentSignals = sortedSignals.slice(
        (currentPage - 1) * itemsPerPage,
        currentPage * itemsPerPage
    );

    const requestSort = (key: any) => {
        let direction: 'asc' | 'desc' = 'desc';
        if (sortConfig.key === key && sortConfig.direction === 'desc') {
            direction = 'asc';
        }
        setSortConfig({ key, direction });
    };

    return (
        <div className="p-6 space-y-6">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                <Card className="bg-card border-border shadow-sm">
                    <CardHeader className="flex flex-row items-center justify-between">
                        <CardTitle className="text-2xl font-bold tracking-tight">Daily Trading Signals</CardTitle>
                        <div className="text-sm text-muted-foreground">
                            Showing {sortedSignals.length} signals
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
                                                <TableHead className="font-semibold cursor-pointer hover:text-primary" onClick={() => requestSort('ticker')}>Ticker</TableHead>
                                                <TableHead className="font-semibold">Model</TableHead>
                                                <TableHead className="font-semibold text-right cursor-pointer hover:text-primary" onClick={() => requestSort('score')}>Score</TableHead>
                                                <TableHead className="font-semibold text-right cursor-pointer hover:text-primary" onClick={() => requestSort('pead')}>PEAD</TableHead>
                                                <TableHead className="font-semibold text-right cursor-pointer hover:text-primary" onClick={() => requestSort('sentiment')}>Sent.</TableHead>
                                                <TableHead className="font-semibold text-right cursor-pointer hover:text-primary" onClick={() => requestSort('vsm')}>VSM</TableHead>
                                                <TableHead className="font-semibold text-right cursor-pointer hover:text-primary" onClick={() => requestSort('bab')}>BAB</TableHead>
                                                <TableHead className="font-semibold text-right cursor-pointer hover:text-primary" onClick={() => requestSort('qmj')}>QMJ</TableHead>
                                                <TableHead className="font-semibold text-right cursor-pointer hover:text-primary" onClick={() => requestSort('upside')}>Upside</TableHead>
                                                <TableHead className="font-semibold text-right">Date</TableHead>
                                            </TableRow>
                                        </TableHeader>
                                        <TableBody>
                                            {currentSignals.map((signal) => {
                                                const meta = JSON.parse(signal.metadata_json || '{}');
                                                // Handle legacy v1 vs v2 vs v3
                                                const vsm = meta.vsm ?? meta.momentum; // Fallback to momentum for v1
                                                const bab = meta.bab ?? 0;
                                                const qmj = meta.qmj ?? 0;
                                                const upside = meta.upside ?? meta.earnings_yield; // Fallback to E/Y for v1
                                                const pead = meta.pead ?? 0;
                                                const sentiment = meta.sentiment ?? 0;
                                                
                                                // Format model name (remove duplicate v)
                                                const modelName = signal.model_name.replace('ranking_', '').toUpperCase();
                                                
                                                // Sentiment icon
                                                const sentimentIcon = sentiment > 0.1 ? '🟢' : sentiment < -0.1 ? '🔴' : '🟡';
                                                
                                                return (
                                                    <TableRow key={signal.id} className="hover:bg-muted/50 transition-colors">
                                                        <TableCell className="font-medium">{signal.ticker}</TableCell>
                                                        <TableCell>
                                                            <Badge variant="outline" className="font-normal">
                                                                {modelName}
                                                            </Badge>
                                                        </TableCell>
                                                        <TableCell className="text-right">
                                                            <span className={`px-2 py-1 rounded text-white text-xs font-bold ${getScoreColor(signal.score)}`}>
                                                                {signal.score.toFixed(2)}
                                                            </span>
                                                        </TableCell>
                                                        <TableCell className={`text-right text-sm ${pead > 0.5 ? 'text-green-400' : pead < -0.5 ? 'text-red-400' : 'text-muted-foreground'}`}>
                                                            {pead !== undefined && pead !== null ? pead.toFixed(2) : 'N/A'}
                                                        </TableCell>
                                                        <TableCell className="text-right text-sm text-muted-foreground">
                                                            {sentimentIcon} {sentiment !== undefined && sentiment !== null ? sentiment.toFixed(2) : 'N/A'}
                                                        </TableCell>
                                                        <TableCell className="text-right text-sm text-muted-foreground">
                                                            {vsm !== undefined && vsm !== null ? vsm.toFixed(2) : 'N/A'}
                                                        </TableCell>
                                                        <TableCell className="text-right text-sm text-muted-foreground">
                                                            {bab !== undefined && bab !== null ? bab.toFixed(2) : 'N/A'}
                                                        </TableCell>
                                                        <TableCell className="text-right text-sm text-muted-foreground">
                                                            {qmj !== undefined && qmj !== null ? qmj.toFixed(2) : 'N/A'}
                                                        </TableCell>
                                                        <TableCell className={`text-right text-sm ${upside > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                            {upside !== undefined && upside !== null ? (upside * 100).toFixed(1) + '%' : 'N/A'}
                                                        </TableCell>
                                                        <TableCell className="text-right text-sm text-muted-foreground">
                                                            {new Date(signal.timestamp).toLocaleDateString()}
                                                        </TableCell>
                                                    </TableRow>
                                                );
                                            })}
                                            {signals.length === 0 && (
                                                <TableRow>
                                                <TableCell colSpan={10} className="text-center py-8 text-muted-foreground">
                                                        No signals found. Run the daily job to generate signals.
                                                    </TableCell>
                                                </TableRow>
                                            )}
                                        </TableBody>
                                    </Table>
                                </div>

                                {/* Pagination Controls */}
                                <div className="flex items-center justify-between px-2">
                                    <div className="text-sm text-muted-foreground">
                                        Page {currentPage} of {totalPages}
                                    </div>
                                    <div className="flex gap-2">
                                        <button
                                            className="px-3 py-1 text-sm border rounded hover:bg-muted disabled:opacity-50"
                                            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                            disabled={currentPage === 1}
                                        >
                                            Previous
                                        </button>
                                        <button
                                            className="px-3 py-1 text-sm border rounded hover:bg-muted disabled:opacity-50"
                                            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                            disabled={currentPage === totalPages}
                                        >
                                            Next
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </motion.div>
        </div>
    );
};

export default SignalsDashboard;
