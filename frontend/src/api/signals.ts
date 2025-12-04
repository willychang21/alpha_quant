import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1';

export interface Signal {
    id: number;
    timestamp: string;
    ticker: string;
    model_name: string;
    model_version: string;
    score: number;
    metadata_json: string;
}

export const getSignals = async (filters: {
    ticker?: string;
    model_name?: string;
    limit?: number;
} = {}) => {
    const params = new URLSearchParams();
    if (filters.ticker) params.append('ticker', filters.ticker);
    if (filters.model_name) params.append('model_name', filters.model_name);
    if (filters.limit) params.append('limit', filters.limit.toString());

    const response = await axios.get<Signal[]>(`${API_URL}/signals/?${params.toString()}`);
    return response.data;
};
