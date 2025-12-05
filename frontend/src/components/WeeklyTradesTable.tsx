import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

interface Trade {
    Ticker: string;
    Name: string;
    Sector: string;
    Weight: number;
    Value: number;
    Price: number;
    Shares: number;
    Action: string;
}

export const WeeklyTradesTable = () => {
    const [trades, setTrades] = useState<Trade[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchTrades = async () => {
            try {
                const response = await fetch('/api/quant/weekly/trades');
                if (response.ok) {
                    const data = await response.json();
                    setTrades(data);
                }
            } catch (error) {
                console.error("Failed to fetch weekly trades:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchTrades();
    }, []);

    if (loading) return <div className="p-4">Loading Trades...</div>;
    
    if (trades.length === 0) {
        return (
            <Card className="w-full">
                <CardHeader>
                    <CardTitle>Weekly Rebalancing Targets</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="text-muted-foreground text-sm">
                        No trades generated yet. Run the weekly job to populate this table.
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="w-full">
            <CardHeader>
                <CardTitle>Weekly Rebalancing Targets (Kelly v1)</CardTitle>
            </CardHeader>
            <CardContent>
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Ticker</TableHead>
                            <TableHead>Sector</TableHead>
                            <TableHead>Action</TableHead>
                            <TableHead className="text-right">Weight</TableHead>
                            <TableHead className="text-right">Shares</TableHead>
                            <TableHead className="text-right">Value ($)</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {trades.map((trade) => (
                            <TableRow key={trade.Ticker}>
                                <TableCell className="font-medium">{trade.Ticker}</TableCell>
                                <TableCell>{trade.Sector}</TableCell>
                                <TableCell>
                                    <Badge variant={trade.Action === 'BUY' ? 'default' : 'destructive'}>
                                        {trade.Action}
                                    </Badge>
                                </TableCell>
                                <TableCell className="text-right">{(trade.Weight * 100).toFixed(2)}%</TableCell>
                                <TableCell className="text-right">{trade.Shares.toLocaleString()}</TableCell>
                                <TableCell className="text-right">${trade.Value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </CardContent>
        </Card>
    );
};
