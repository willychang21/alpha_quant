export interface MatrixHolding {
  ticker: string;
  name: string;
  etfs: Record<string, number>;
  [key: string]: any;
}

export interface EtfHeader {
  id: string;
  weight: number;
  [key: string]: any;
}

export interface SelectedAsset {
  id: string;
  type: string;
  allocation: number | string;
  label?: string;
}

export interface ProjectedHolding {
  ticker: string;
  name: string;
  marketValue: number;
  percentOfPortfolio: number;
  etfs: Record<string, number>;
  direct: number;
  fundsHolding: number;
}
