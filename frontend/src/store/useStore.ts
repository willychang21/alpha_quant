import { create } from 'zustand';
import api from '../api/axios';
import { API_ENDPOINTS } from '../api/endpoints';

export interface Portfolio {
  id: string;
  name: string;
  // Add other properties as needed
}

export interface PortfolioData {
  summary: {
    totalValue: number;
    dayChange: number;
    dayChangePercent: number;
    holdingsCount: number;
  };
  matrixHoldings: any[];
  etfHeaders: any[];
  profitLossHoldings: any[];
  lastUpdated: string;
  // Add other properties as needed
}

export interface DcaConfig {
  monthlyAmount: number;
  selectedAssets: any[]; // Define specific type
  dynamicCompositions: Record<string, any>; // Define specific type
  stockNames: Record<string, string>;
}

interface AppState {
  portfolios: Portfolio[];
  currentPortfolioId: string | null;
  data: PortfolioData | null;
  loading: boolean;
  error: string | null;
  dcaConfig: DcaConfig;
  
  // Actions
  setPortfolios: (portfolios: Portfolio[]) => void;
  setCurrentPortfolioId: (id: string | null) => void;
  setData: (data: PortfolioData | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setDcaConfig: (config: DcaConfig) => void;
  
  // Async Actions
  fetchPortfolios: () => Promise<void>;
  fetchData: () => Promise<void>;
  createPortfolio: (name: string) => Promise<void>;
  renamePortfolio: (id: string, name: string) => Promise<void>;
  deletePortfolio: (id: string) => Promise<void>;
}

export const useStore = create<AppState>((set, get) => ({
  portfolios: [],
  currentPortfolioId: null,
  data: null,
  loading: true,
  error: null,
  dcaConfig: {
    monthlyAmount: 1000,
    selectedAssets: [],
    dynamicCompositions: {},
    stockNames: {}
  },

  setPortfolios: (portfolios) => set({ portfolios }),
  setCurrentPortfolioId: (currentPortfolioId) => set({ currentPortfolioId }),
  setData: (data) => set({ data }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setDcaConfig: (dcaConfig) => set({ dcaConfig }),

  fetchPortfolios: async () => {
    try {
      const response = await api.get(API_ENDPOINTS.PORTFOLIOS);
      const list = response.data;
      set({ portfolios: Array.isArray(list) ? list : [] });
      const { currentPortfolioId } = get();
      if (list.length > 0 && !currentPortfolioId) {
        set({ currentPortfolioId: list[0].id });
      }
    } catch (err) {
      console.error('Failed to fetch portfolios', err);
    }
  },

  fetchData: async () => {
    const { currentPortfolioId } = get();
    if (!currentPortfolioId) return;
    
    set({ loading: true });
    try {
      const response = await api.get(API_ENDPOINTS.PORTFOLIO_DATA(currentPortfolioId));
      set({ data: response.data, error: null });
    } catch (err: any) {
      set({ error: err.message || 'Failed to fetch data' });
    } finally {
      set({ loading: false });
    }
  },

  createPortfolio: async (name) => {
    try {
      const response = await api.post(API_ENDPOINTS.PORTFOLIOS, { name });
      const newPortfolio = response.data;
      const { portfolios } = get();
      set({ 
        portfolios: [...portfolios, newPortfolio],
        currentPortfolioId: newPortfolio.id 
      });
    } catch (err) {
      console.error('Failed to create portfolio', err);
    }
  },

  renamePortfolio: async (id, name) => {
    try {
      await api.put(API_ENDPOINTS.PORTFOLIO_BY_ID(id), { name });
      get().fetchPortfolios();
    } catch (err) {
      console.error('Failed to rename portfolio', err);
    }
  },

  deletePortfolio: async (id) => {
    try {
      await api.delete(API_ENDPOINTS.PORTFOLIO_BY_ID(id));
      const { portfolios } = get();
      const newList = portfolios.filter(p => p.id !== id);
      set({ portfolios: newList });
      
      if (newList.length > 0) {
        set({ currentPortfolioId: newList[0].id });
      } else {
        set({ currentPortfolioId: null, data: null });
      }
    } catch (err) {
      console.error('Failed to delete portfolio', err);
    }
  }
}));
