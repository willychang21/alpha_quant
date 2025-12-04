import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { X, Check } from 'lucide-react';
import { cn } from "@/lib/utils"
import { DcaConfig } from '../store/useStore';
import api from '../api/axios';
import { API_ENDPOINTS } from '../api/endpoints';

interface DCAAnalysisProps {
  dcaConfig: DcaConfig;
  setDcaConfig: (config: DcaConfig) => void;
}

interface SearchResult {
  id: string;
  label: string;
  type: string;
}

interface MatrixHolding {
    ticker: string;
    name: string;
    percentOfPortfolio: number;
    fundsHolding: number;
    etfs: Record<string, number>;
    direct: number;
    marketValue: number;
}

interface EtfHeader {
    id: string;
    label: string;
    weight: number;
}

const DCAAnalysis: React.FC<DCAAnalysisProps> = ({ dcaConfig, setDcaConfig }) => {
  // Destructure from config, providing defaults just in case
  const { monthlyAmount = 1000, selectedAssets = [], dynamicCompositions = {}, stockNames = {} } = dcaConfig || {};

  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  
  // State for matrix data fetched from backend
  const [matrixHoldings, setMatrixHoldings] = useState<MatrixHolding[]>([]);
  const [etfHeaders, setEtfHeaders] = useState<EtfHeader[]>([]);
  const [isLoadingMatrix, setIsLoadingMatrix] = useState(false);

  // Helper to update config
  const updateConfig = (updates: Partial<DcaConfig>) => {
    setDcaConfig({ ...dcaConfig, ...updates });
  };

  // Search Effect
  useEffect(() => {
    const searchAssets = async () => {
      if (!searchQuery || searchQuery.length < 2) {
        setSearchResults([]);
        return;
      }

      setIsSearching(true);
      try {
        const res = await api.get(`${API_ENDPOINTS.SEARCH}?q=${encodeURIComponent(searchQuery)}`);
        const data = res.data;
        if (!Array.isArray(data)) {
          console.error("Search response is not an array:", data);
          setSearchResults([]);
          return;
        }
        // Map API results to our format
        const formatted = data.map((item: any) => ({
          id: item.symbol,
          label: item.shortname || item.symbol,
          type: item.typeDisp || 'Asset'
        }));
        setSearchResults(formatted);
      } catch (err) {
        console.error("Search failed", err);
      } finally {
        setIsSearching(false);
      }
    };

    const timeoutId = setTimeout(searchAssets, 300);
    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  // Fetch Matrix Analysis
  const fetchAnalysis = useCallback(async () => {
      setIsLoadingMatrix(true);
      try {
          const res = await api.post(API_ENDPOINTS.ANALYZE, {
              monthlyAmount,
              selectedAssets,
              dynamicCompositions,
              stockNames
          });
          setMatrixHoldings(res.data.matrixHoldings);
          setEtfHeaders(res.data.etfHeaders);
      } catch (error) {
          console.error("Failed to fetch analysis", error);
      } finally {
          setIsLoadingMatrix(false);
      }
  }, [monthlyAmount, selectedAssets, dynamicCompositions, stockNames]);

  // Debounce the analysis fetch to avoid too many calls while sliding sliders
  useEffect(() => {
      const handler = setTimeout(() => {
          fetchAnalysis();
      }, 500);
      return () => clearTimeout(handler);
  }, [fetchAnalysis]);


  const handleAllocationChange = (id: string, val: string | number, type: 'amount' | 'percent' = 'amount') => {
    const numVal = typeof val === 'string' ? parseFloat(val) : val;
    const safeVal = isNaN(numVal) ? 0 : numVal;
    
    const newAssets = selectedAssets.map(asset => {
      if (asset.id !== id) return asset;

      if (type === 'percent') {
        // Calculate amount from percent
        const newAmount = (safeVal / 100) * monthlyAmount;
        return { ...asset, allocation: newAmount };
      } else {
        // Direct amount update
        return { ...asset, allocation: safeVal };
      }
    });

    updateConfig({ selectedAssets: newAssets });
  };

  const handleAddAsset = async (asset: SearchResult) => {
    console.log("handleAddAsset called for:", asset);
    if (asset && !selectedAssets.find(a => a.id === asset.id)) {
      let newDynamicCompositions = { ...dynamicCompositions };
      let newStockNames = { ...stockNames };
      let assetType = asset.type;

      // Fetch details to check if it's an ETF with holdings
      try {
        const res = await api.get(API_ENDPOINTS.ASSET_DETAILS(asset.id));
        const details = res.data;
        if (details.holdings && details.holdings.length > 0) {
          // It's an ETF, store composition
          const composition: Record<string, number> = {};
          details.holdings.forEach((h: any) => {
            composition[h.ticker] = h.percent;
            newStockNames[h.ticker] = h.name; // Store name
          });
          newDynamicCompositions[asset.id] = composition;
          
          // Update asset type if needed
          assetType = 'ETF'; 
        }
      } catch (e) {
        console.error("Failed to fetch asset details", e);
      }

      const newAssets = [...selectedAssets, { ...asset, type: assetType, allocation: 0 }];
      updateConfig({ 
        selectedAssets: newAssets,
        dynamicCompositions: newDynamicCompositions,
        stockNames: newStockNames
      });
      
      setOpen(false);
      setSearchQuery("");
    } else {
      console.log("Asset already selected or invalid");
    }
  };

  const handleRemoveAsset = (id: string) => {
    const newAssets = selectedAssets.filter(a => a.id !== id);
    updateConfig({ selectedAssets: newAssets });
  };

  const distributeEvenly = () => {
    if (selectedAssets.length === 0) return;
    const share = monthlyAmount / selectedAssets.length;
    const newAssets = selectedAssets.map(a => ({ ...a, allocation: share }));
    updateConfig({ selectedAssets: newAssets });
  };

  const totalAllocated = selectedAssets.reduce((sum, asset) => sum + (typeof asset.allocation === 'number' ? asset.allocation : parseFloat(asset.allocation as string) || 0), 0);
  const remaining = monthlyAmount - totalAllocated;

  // Active ETFs are now just the headers returned from backend
  const activeEtfs = etfHeaders;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Section */}
        <Card className="lg:col-span-1 material-card p-6">
          <CardHeader className="px-0 pt-0 pb-6">
            <CardTitle className="text-xl font-normal">DCA Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6 px-0 pb-0">
            <div className="space-y-3">
              <Label htmlFor="monthly-amount" className="text-base font-medium">Monthly Investment</Label>
              <div className="relative">
                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground font-medium">$</span>
                <Input
                  id="monthly-amount"
                  type="number"
                  value={monthlyAmount}
                  onChange={(e) => updateConfig({ monthlyAmount: parseFloat(e.target.value) || 0 })}
                  className="pl-8 h-12 input-material text-lg font-normal bg-secondary/30"
                />
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label className="text-base font-medium">Asset Allocation</Label>
                <span className={`text-xs font-bold px-2 py-1 rounded-full ${Math.abs(remaining) < 0.01 ? 'bg-success/10 text-success' : 'bg-warning/10 text-warning-dark'}`}>
                  {Math.abs(remaining) < 0.01 ? 'Fully Allocated' : `Remaining: $${remaining.toFixed(2)}`}
                </span>
              </div>

              {/* Add Asset Control (Combobox) */}
              <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={open}
                    className="w-full justify-between h-12 rounded-full border-none bg-secondary/50 hover:bg-secondary/80 text-foreground px-5 transition-all"
                  >
                    <span className="text-muted-foreground">Select asset to add...</span>
                    <span className="text-xs bg-background/50 px-2 py-0.5 rounded-full text-muted-foreground">+</span>
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-[300px] p-0 bg-popover border border-border shadow-xl rounded-2xl overflow-hidden">
                  <Command shouldFilter={false} className="bg-transparent">
                    <CommandInput 
                      placeholder="Search assets (e.g. AAPL, BTC)..." 
                      value={searchQuery}
                      onValueChange={setSearchQuery}
                      className="h-12 border-b border-border/40 bg-transparent"
                    />
                    <CommandList className="p-2">
                      {isSearching && <div className="py-6 text-center text-sm text-muted-foreground">Searching...</div>}
                      {!isSearching && searchResults.length === 0 && searchQuery.length > 1 && (
                        <CommandEmpty>No asset found.</CommandEmpty>
                      )}
                      {!isSearching && searchResults.length > 0 && (
                        <CommandGroup heading="Search Results">
                          {searchResults.map((asset) => {
                            const isSelected = selectedAssets.some(a => a.id === asset.id);
                            return (
                              <CommandItem
                                key={asset.id}
                                value={asset.id.toLowerCase()}
                                onSelect={(currentValue) => {
                                  console.log("Asset selected:", asset.id, "Current Value:", currentValue);
                                  handleAddAsset(asset);
                                }}
                                className="hover:bg-secondary cursor-pointer rounded-lg py-3 px-4 mb-1 aria-selected:bg-primary/10 aria-selected:text-primary"
                                disabled={isSelected}
                              >
                                <Check
                                  className={cn(
                                    "mr-2 h-4 w-4 text-primary",
                                    isSelected ? "opacity-100" : "opacity-0"
                                  )}
                                />
                                <div className="flex flex-col">
                                  <span className="font-medium">{asset.label}</span>
                                  <span className="text-xs text-muted-foreground">{asset.type} • {asset.id}</span>
                                </div>
                              </CommandItem>
                            );
                          })}
                        </CommandGroup>
                      )}
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
              
              <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                {selectedAssets.map(asset => {
                  const allocation = typeof asset.allocation === 'number' ? asset.allocation : parseFloat(asset.allocation as string) || 0;
                  const percent = monthlyAmount > 0 ? (allocation / monthlyAmount * 100) : 0;
                  return (
                    <motion.div
                      key={asset.id}
                      layout
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="space-y-3 p-4 bg-secondary/30 rounded-[20px] group transition-colors hover:bg-secondary/50"
                    >
                      <div className="flex justify-between items-center mb-1">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-base">{asset.id}</span>
                          {asset.label !== asset.id && (
                            <span className="text-xs text-muted-foreground truncate max-w-[120px]" title={asset.label}>
                              {asset.label}
                            </span>
                          )}
                          <span className="text-[10px] font-bold text-primary/80 bg-primary/10 px-2 py-0.5 rounded-full uppercase tracking-wide">
                            {asset.type}
                          </span>
                        </div>
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-8 w-8 rounded-full text-muted-foreground hover:text-destructive hover:bg-destructive/10 opacity-0 group-hover:opacity-100 transition-all"
                          onClick={() => handleRemoveAsset(asset.id)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                      
                      <div className="flex items-center gap-3">
                        <div className="relative flex-1">
                          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground font-medium">$</span>
                          <Input
                            type="number"
                            value={allocation || 0}
                            onChange={(e) => handleAllocationChange(asset.id, e.target.value, 'amount')}
                            className="pl-6 h-10 bg-background border-none shadow-none focus-visible:ring-1 focus-visible:ring-primary/30 rounded-lg"
                          />
                        </div>
                        <div className="relative w-24">
                          <Input
                            type="number"
                            value={percent.toFixed(1)}
                            onChange={(e) => handleAllocationChange(asset.id, e.target.value, 'percent')}
                            className="pr-6 h-10 text-right bg-background border-none shadow-none focus-visible:ring-1 focus-visible:ring-primary/30 rounded-lg"
                          />
                          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground font-medium">%</span>
                        </div>
                      </div>
                      
                      <Slider
                        value={[allocation || 0]}
                        max={monthlyAmount}
                        step={10}
                        onValueChange={(vals: number[]) => handleAllocationChange(asset.id, vals[0], 'amount')}
                        className="w-full py-2"
                      />
                    </motion.div>
                  );
                })}
                
                {selectedAssets.length === 0 && (
                  <div className="text-center py-12 text-muted-foreground text-sm border-2 border-dashed border-border/50 rounded-2xl bg-secondary/10">
                    No assets selected. Add assets above.
                  </div>
                )}
              </div>
              
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full h-10 rounded-full font-medium border-primary/20 hover:bg-primary/5 hover:text-primary transition-colors"
                onClick={distributeEvenly}
                disabled={selectedAssets.length === 0}
              >
                Distribute Evenly
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Analysis Section */}
        <Card className="lg:col-span-2 material-card p-6 flex flex-col h-full">
          <CardHeader className="px-0 pt-0 pb-6">
            <CardTitle className="flex justify-between items-center text-xl font-normal">
                <span>Projected Matrix Analysis</span>
                {isLoadingMatrix && <span className="text-xs text-primary font-medium animate-pulse bg-primary/10 px-3 py-1 rounded-full">Updating...</span>}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-hidden flex flex-col px-0 pb-0">
            <div className="rounded-2xl border border-border/40 overflow-hidden bg-surface flex-1 shadow-sm">
              <div className="overflow-x-auto h-full max-h-[600px] custom-scrollbar">
                <Table>
                  <TableHeader>
                    <TableRow className="sticky top-0 bg-background/95 backdrop-blur-md z-10 border-b border-border/60 hover:bg-background/95">
                      <TableHead className="w-[100px] font-semibold text-foreground pl-6 whitespace-nowrap">Ticker</TableHead>
                      <TableHead className="min-w-[180px] font-semibold text-foreground whitespace-nowrap">Name</TableHead>
                      <TableHead className="text-right font-semibold text-foreground whitespace-nowrap">Projected $</TableHead>
                      <TableHead className="text-right font-semibold text-foreground whitespace-nowrap">% Portfolio</TableHead>
                      {activeEtfs.map((etf) => (
                        <TableHead key={etf.id} className="text-center font-semibold text-xs text-muted-foreground whitespace-nowrap max-w-[120px] truncate" title={etf.label}>
                          {etf.label} %
                        </TableHead>
                      ))}
                      <TableHead className="text-right font-semibold text-xs text-muted-foreground pr-6 whitespace-nowrap">Direct %</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {matrixHoldings.map((holding, idx) => (
                      <motion.tr 
                        key={holding.ticker}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.2, delay: idx * 0.02 }}
                        className={`border-b border-border/30 last:border-0 hover:bg-secondary/30 transition-colors ${idx % 2 === 0 ? 'bg-transparent' : 'bg-secondary/10'}`}
                      >
                        <TableCell className="font-medium pl-6">
                          <span className="font-mono font-bold text-xs bg-secondary/50 px-2 py-1 rounded text-foreground">{holding.ticker}</span>
                        </TableCell>
                        <TableCell className="max-w-[180px] truncate text-sm font-medium text-muted-foreground" title={holding.name}>
                          {holding.name}
                        </TableCell>
                        <TableCell className="text-right font-mono font-medium text-foreground">
                          ${holding.marketValue.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-foreground">
                          {holding.percentOfPortfolio.toFixed(2)}%
                        </TableCell>
                        {activeEtfs.map((etf) => (
                          <TableCell key={etf.id} className="text-center font-mono text-xs text-muted-foreground">
                            {holding.etfs[etf.id] > 0
                              ? <span className="text-foreground/80">{holding.etfs[etf.id].toFixed(1)}%</span>
                              : <span className="opacity-20">-</span>}
                          </TableCell>
                        ))}
                        <TableCell className="text-right font-mono text-xs text-muted-foreground pr-6">
                          {holding.direct > 0
                            ? <span className="text-primary font-bold">{holding.direct.toFixed(1)}%</span>
                            : <span className="opacity-20">-</span>}
                        </TableCell>
                      </motion.tr>
                    ))}
                    {matrixHoldings.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={5 + activeEtfs.length} className="text-center py-16 text-muted-foreground">
                          <div className="flex flex-col items-center gap-2">
                            <div className="p-4 rounded-full bg-secondary/30 mb-2">
                              <svg className="w-8 h-8 text-muted-foreground/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                              </svg>
                            </div>
                            <p className="font-medium">No holdings projected</p>
                            <p className="text-xs text-muted-foreground/70">Add assets and allocate funds to see results</p>
                          </div>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default DCAAnalysis;
