import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Dna } from 'lucide-react';

interface AttributionData {
    value: number;
    momentum: number;
    quality: number;
    low_risk: number;
    sentiment: number;
}

interface FactorRadarProps {
    attribution: AttributionData;
}

export const FactorRadar: React.FC<FactorRadarProps> = ({ attribution }) => {
    // Normalize to 0-1 scale for radar (assuming z-scores typically -3 to +3)
    const normalize = (val: number) => Math.max(0, Math.min((val + 3) / 6, 1)) * 100;
    
    const data = [
        { factor: 'Value', score: normalize(attribution.value), raw: attribution.value },
        { factor: 'Momentum', score: normalize(attribution.momentum), raw: attribution.momentum },
        { factor: 'Quality', score: normalize(attribution.quality), raw: attribution.quality },
        { factor: 'Low Risk', score: normalize(attribution.low_risk), raw: attribution.low_risk },
        { factor: 'Sentiment', score: normalize(attribution.sentiment), raw: attribution.sentiment },
    ];

    // Find dominant factor
    const dominantFactor = data.reduce((prev, curr) => 
        curr.raw > prev.raw ? curr : prev
    );

    return (
        <Card className="h-full border-l-4 border-l-cyan-500">
            <CardHeader>
                <div className="flex items-center space-x-2">
                    <Dna className="w-5 h-5 text-cyan-500" />
                    <CardTitle>Factor Attribution</CardTitle>
                </div>
                <CardDescription>
                    Portfolio "DNA" • Dominant: <span className="font-bold text-cyan-400">{dominantFactor.factor}</span>
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
                            <PolarGrid stroke="#374151" />
                            <PolarAngleAxis 
                                dataKey="factor" 
                                tick={{ fill: '#9ca3af', fontSize: 11 }}
                            />
                            <PolarRadiusAxis 
                                angle={90} 
                                domain={[0, 100]} 
                                tick={{ fill: '#6b7280', fontSize: 9 }}
                            />
                            <Radar
                                name="Factor Exposure"
                                dataKey="score"
                                stroke="#22d3ee"
                                fill="#22d3ee"
                                fillOpacity={0.3}
                                strokeWidth={2}
                            />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>
                
                {/* Factor Legend */}
                <div className="grid grid-cols-5 gap-2 mt-2 text-xs text-center">
                    {data.map((d) => (
                        <div key={d.factor} className="space-y-1">
                            <div className={`font-mono ${d.raw > 0 ? 'text-green-400' : d.raw < 0 ? 'text-red-400' : 'text-gray-400'}`}>
                                {d.raw > 0 ? '+' : ''}{d.raw.toFixed(1)}
                            </div>
                            <div className="text-muted-foreground truncate">{d.factor}</div>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    );
};
