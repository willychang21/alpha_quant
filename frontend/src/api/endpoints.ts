export const API_ENDPOINTS = {
  PORTFOLIOS: '/portfolios',
  PORTFOLIO_BY_ID: (id: number | string) => `/portfolios/${id}`,
  PORTFOLIO_DATA: (id: number | string) => `/portfolio/${id}`,
  PORTFOLIO_CONFIG: (id: number | string) => `/config/${id}`,
  SEARCH: '/search',
  ASSET_DETAILS: (ticker: string) => `/asset-details/${ticker}`,
  ANALYZE: '/analyze',
  BACKTEST: '/backtest',
  FUNDAMENTAL: (ticker: string) => `/fundamental/${ticker}`,
  VALUATION: (ticker: string) => `/valuation/${ticker}`,
};
