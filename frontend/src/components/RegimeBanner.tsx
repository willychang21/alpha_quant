import React from 'react';
import { Shield, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';

interface RegimeData {
    state: 'AGGRESSIVE' | 'NEUTRAL' | 'DEFENSIVE';
    confidence_score: number;
    confidence_multiplier: number;
    base_target_vol: number;
    adjusted_target_vol: number;
}

interface RegimeBannerProps {
    regime: RegimeData;
}

export const RegimeBanner: React.FC<RegimeBannerProps> = ({ regime }) => {
    const getStateConfig = () => {
        switch (regime.state) {
            case 'AGGRESSIVE':
                return {
                    bg: 'bg-gradient-to-r from-green-600 to-emerald-500',
                    icon: TrendingUp,
                    label: 'AGGRESSIVE MODE',
                    description: 'High Confidence • Full Risk Deployment'
                };
            case 'NEUTRAL':
                return {
                    bg: 'bg-gradient-to-r from-amber-500 to-yellow-500',
                    icon: AlertTriangle,
                    label: 'NEUTRAL MODE',
                    description: 'Moderate Confidence • Standard Exposure'
                };
            case 'DEFENSIVE':
            default:
                return {
                    bg: 'bg-gradient-to-r from-red-600 to-rose-500',
                    icon: Shield,
                    label: 'DEFENSIVE MODE',
                    description: 'Low Confidence • Risk Reduction Active'
                };
        }
    };

    const config = getStateConfig();
    const Icon = config.icon;

    return (
        <div className={`${config.bg} text-white px-6 py-4 rounded-lg shadow-lg`}>
            <div className="flex items-center justify-between">
                {/* Left: State */}
                <div className="flex items-center space-x-4">
                    <div className="p-3 bg-white/20 rounded-full">
                        <Icon className="w-6 h-6" />
                    </div>
                    <div>
                        <div className="text-xl font-bold tracking-wide">{config.label}</div>
                        <div className="text-sm opacity-90">{config.description}</div>
                    </div>
                </div>

                {/* Right: Metrics */}
                <div className="flex items-center space-x-8">
                    <div className="text-center">
                        <div className="text-2xl font-mono font-bold">
                            {(regime.confidence_score * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs uppercase tracking-wide opacity-80">Sharpe</div>
                    </div>
                    <div className="text-center">
                        <div className="text-2xl font-mono font-bold">
                            {regime.confidence_multiplier.toFixed(1)}x
                        </div>
                        <div className="text-xs uppercase tracking-wide opacity-80">Leverage</div>
                    </div>
                    <div className="text-center">
                        <div className="text-lg font-mono">
                            <span className="opacity-60 line-through">{(regime.base_target_vol * 100).toFixed(0)}%</span>
                            {' → '}
                            <span className="font-bold">{(regime.adjusted_target_vol * 100).toFixed(1)}%</span>
                        </div>
                        <div className="text-xs uppercase tracking-wide opacity-80">Target Vol</div>
                    </div>
                </div>
            </div>
        </div>
    );
};
