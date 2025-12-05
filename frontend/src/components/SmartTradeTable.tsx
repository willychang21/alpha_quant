import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ArrowDownRight, ArrowUpRight, Info, ListChecks } from 'lucide-react';

interface Trade {
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
}

interface SmartTradeTableProps {
    trades: Trade[];
    confidenceMultiplier: number;
}

export const SmartTradeTable: React.FC<SmartTradeTableProps> = ({ trades, confidenceMultiplier }) => {
    const getConvictionColor = (conviction: string) => {
        switch (conviction) {
            case 'High': return 'bg-green-500/20 text-green-400 border-green-500/50';
            case 'Medium': return 'bg-amber-500/20 text-amber-400 border-amber-500/50';
            default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
        }
    };

    const reductionPct = confidenceMultiplier < 1 ? Math.round((1 - confidenceMultiplier) * 100) : 0;

    return (
        <Card className="w-full border-l-4 border-l-indigo-500">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                        <ListChecks className="w-5 h-5 text-indigo-500" />
                        <CardTitle>Smart Execution Plan</CardTitle>
                    </div>
                    {reductionPct > 0 && (
                        <Badge variant="destructive" className="text-xs">
                            <ArrowDownRight className="w-3 h-3 mr-1" />
                            {reductionPct}% Risk Reduction Applied
                        </Badge>
                    )}
                </div>
                <CardDescription>
                    {trades.length} positions • Regime-adjusted sizing
                </CardDescription>
            </CardHeader>
            <CardContent>
                <TooltipProvider>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Ticker</TableHead>
                                <TableHead>Sector</TableHead>
                                <TableHead className="text-center">Alpha (Z)</TableHead>
                                <TableHead className="text-center">Conviction</TableHead>
                                <TableHead className="text-right">Weight</TableHead>
                                <TableHead className="text-right">Shares</TableHead>
                                <TableHead className="text-right">Value</TableHead>
                                <TableHead className="text-center">
                                    <Info className="w-4 h-4 inline" />
                                </TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {trades.map((trade) => (
                                <TableRow key={trade.ticker} className="hover:bg-muted/30 transition-colors">
                                    <TableCell className="font-bold">
                                        {trade.ticker}
                                        {trade.name && trade.name !== trade.ticker && (
                                            <span className="text-xs text-muted-foreground ml-2">{trade.name}</span>
                                        )}
                                    </TableCell>
                                    <TableCell>
                                        <span className="text-xs px-2 py-1 rounded bg-muted">
                                            {trade.sector}
                                        </span>
                                    </TableCell>
                                    <TableCell className="text-center">
                                        <div className={`inline-flex items-center px-2 py-1 rounded ${
                                            trade.alpha_score >= 2 ? 'bg-green-500/20 text-green-400' :
                                            trade.alpha_score >= 1 ? 'bg-amber-500/20 text-amber-400' :
                                            'bg-gray-500/20 text-gray-400'
                                        }`}>
                                            {trade.alpha_score >= 0 ? '+' : ''}{trade.alpha_score.toFixed(1)}z
                                        </div>
                                    </TableCell>
                                    <TableCell className="text-center">
                                        <Badge 
                                            variant="outline" 
                                            className={getConvictionColor(trade.conviction)}
                                        >
                                            {trade.conviction}
                                        </Badge>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <span className="font-mono">
                                                    {(trade.final_weight * 100).toFixed(1)}%
                                                </span>
                                                {trade.raw_weight !== trade.final_weight && (
                                                    <span className="text-xs text-muted-foreground line-through ml-2">
                                                        {(trade.raw_weight * 100).toFixed(1)}%
                                                    </span>
                                                )}
                                            </TooltipTrigger>
                                            <TooltipContent>
                                                <p>Raw: {(trade.raw_weight * 100).toFixed(2)}%</p>
                                                <p>Final: {(trade.final_weight * 100).toFixed(2)}%</p>
                                                <p className="text-amber-400">{trade.reason}</p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TableCell>
                                    <TableCell className="text-right font-mono">
                                        {trade.shares.toLocaleString()}
                                    </TableCell>
                                    <TableCell className="text-right font-mono">
                                        ${trade.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                    </TableCell>
                                    <TableCell className="text-center">
                                        <Tooltip>
                                            <TooltipTrigger>
                                                <Info className="w-4 h-4 text-muted-foreground hover:text-foreground cursor-help" />
                                            </TooltipTrigger>
                                            <TooltipContent side="left" className="max-w-[200px]">
                                                <p className="font-bold">{trade.reason}</p>
                                                <p className="text-xs text-muted-foreground mt-1">
                                                    System adjusted position size based on current market regime.
                                                </p>
                                            </TooltipContent>
                                        </Tooltip>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TooltipProvider>
            </CardContent>
        </Card>
    );
};
