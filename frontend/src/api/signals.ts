import api from './axios';

export interface Signal {
    id: number;
    timestamp: string;
    ticker: string;
    model_name: string;
    model_version: string;
    score: number;
    metadata_json: string;
}

export const getLatestSignals = async (model_name: string = 'ranking_v3', limit: number = 1000) => {
    const params = new URLSearchParams();
    params.append('model_name', model_name);
    params.append('limit', limit.toString());
    
    const response = await api.get<{ signals: Signal[], date: string, model: string }>(`/signals/latest?${params.toString()}`);
    return response.data.signals;
};
