import api from './axios';

/** ML Alpha status from backend */
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
    featureAttribution?: Array<{
        factor: string;
        contribution: number;
        direction: 'positive' | 'negative';
    }>;
}

export interface MLSignalMetrics {
    status: string;
    run_id?: string;
    metrics?: Record<string, number>;
    params?: Record<string, string>;
}

export interface RiskMetrics {
    var: {
        portfolio_var: number;
        component_var: number[];
        weights: number[];
    };
    hedge: {
        total_cost: number;
        hedge_ratio: number;
        contracts: number;
    };
}

export interface VWAPSchedule {
    schedule: Array<{
        Time: string;
        Shares: number;
        Pct: number;
    }>;
    impact_cost_bps: number;
    total_shares: number;
}

export const quantApi = {
    /** Get ML Alpha Enhancement status */
    getMLAlphaStatus: async (): Promise<MLAlphaStatus> => {
        const response = await api.get('/quant/ml/status');
        return response.data;
    },

    getMLSignals: async (): Promise<MLSignalMetrics> => {
        const response = await api.get('/quant/ml/signals');
        return response.data;
    },

    getRiskMetrics: async (portfolioValue: number = 1000000, volatility: number = 0.15): Promise<RiskMetrics> => {
        const response = await api.get('/quant/risk/metrics', {
            params: { portfolio_value: portfolioValue, volatility }
        });
        return response.data;
    },

    getVWAPSchedule: async (shares: number = 10000): Promise<VWAPSchedule> => {
        const response = await api.get('/quant/execution/vwap', {
            params: { shares }
        });
        return response.data;
    }
};

