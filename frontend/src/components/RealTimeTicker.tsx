import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/badge';
import { ArrowUpIcon, ArrowDownIcon } from 'lucide-react';

interface TickerData {
    symbol: string;
    price: number;
    change: number;
    changePct: number;
}

const INITIAL_DATA: TickerData[] = [
    { symbol: 'SPY', price: 450.00, change: 1.50, changePct: 0.33 },
    { symbol: 'QQQ', price: 380.00, change: -0.80, changePct: -0.21 },
    { symbol: 'IWM', price: 190.00, change: 0.40, changePct: 0.21 },
    { symbol: 'GLD', price: 185.00, change: 0.10, changePct: 0.05 },
    { symbol: 'BTC', price: 45000.00, change: 1200.00, changePct: 2.74 },
    { symbol: 'ETH', price: 2400.00, change: 80.00, changePct: 3.45 },
];

export const RealTimeTicker: React.FC = () => {
    const [data, setData] = useState<TickerData[]>(INITIAL_DATA);

    useEffect(() => {
        // Simulate real-time updates
        const interval = setInterval(() => {
            setData(prevData => prevData.map(item => {
                const move = (Math.random() - 0.5) * (item.price * 0.001); // 0.1% move
                const newPrice = item.price + move;
                const change = newPrice - (item.price - item.change); // Approx
                return {
                    ...item,
                    price: newPrice,
                    change: change,
                    changePct: (change / (newPrice - change)) * 100
                };
            }));
        }, 1000);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="w-full overflow-hidden bg-background border-b border-border py-2">
            <motion.div 
                className="flex space-x-8 whitespace-nowrap"
                animate={{ x: [0, -1000] }}
                transition={{ 
                    repeat: Infinity, 
                    duration: 30, 
                    ease: "linear" 
                }}
            >
                {[...data, ...data, ...data].map((item, index) => (
                    <div key={`${item.symbol}-${index}`} className="flex items-center space-x-2">
                        <span className="font-bold text-sm">{item.symbol}</span>
                        <span className="text-sm font-mono">{item.price.toFixed(2)}</span>
                        <Badge 
                            variant={item.change >= 0 ? "default" : "destructive"}
                            className={`text-xs px-1 ${item.change >= 0 ? 'bg-green-500/10 text-green-500 hover:bg-green-500/20' : 'bg-red-500/10 text-red-500 hover:bg-red-500/20'}`}
                        >
                            {item.change >= 0 ? <ArrowUpIcon className="w-3 h-3 mr-1" /> : <ArrowDownIcon className="w-3 h-3 mr-1" />}
                            {Math.abs(item.changePct).toFixed(2)}%
                        </Badge>
                    </div>
                ))}
            </motion.div>
        </div>
    );
};
