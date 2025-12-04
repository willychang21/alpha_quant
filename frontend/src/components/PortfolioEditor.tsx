import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
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
import { X, Plus } from 'lucide-react';
import api from '../api/axios';
import { API_ENDPOINTS } from '../api/endpoints';

interface PortfolioEditorProps {
  portfolioId: string;
  onClose: () => void;
  onSave: () => void;
}

interface Holding {
  ticker: string;
  quantity: number;
  averageCost: number;
  purchaseDate: string;
  account?: string;
}

interface PortfolioConfig {
  cash: number;
  etfs: Holding[];
  direct: Holding[];
}

interface SearchResult {
  symbol: string;
  shortname: string;
}

const PortfolioEditor: React.FC<PortfolioEditorProps> = ({ portfolioId, onClose, onSave }) => {
  const [config, setConfig] = useState<PortfolioConfig | null>(null);
  const [portfolios, setPortfolios] = useState<any[]>([]);
  const [activePortfolioId, setActivePortfolioId] = useState<string>(portfolioId);
  const [loading, setLoading] = useState(true);
  const [, setError] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<Record<string, SearchResult[]>>({}); 
  const [activeSearch, setActiveSearch] = useState<{ type: 'etf' | 'direct'; index: number } | null>(null);

  useEffect(() => {
    // Fetch all portfolios for tabs
    api.get(API_ENDPOINTS.PORTFOLIOS)
      .then(res => {
        const data = res.data;
    if (!Array.isArray(data)) {
      console.error("Portfolios response is not an array:", data);
      setPortfolios([]);
      return;
    }
    // The original code had a check and assignment: setPortfolios(Array.isArray(list) ? list : []);
    // This new block replaces that with a more explicit check and error logging.
    setPortfolios(data);
  })
      .catch(err => console.error('Failed to load portfolios', err));
  }, []);

  useEffect(() => {
    let mounted = true;
    const loadConfig = async () => {
      try {
        const res = await api.get(API_ENDPOINTS.PORTFOLIO_CONFIG(activePortfolioId));
        if (mounted) {
          setConfig(res.data);
          setLoading(false);
        }
      } catch {
        if (mounted) {
          setError('Failed to load config');
          setLoading(false);
        }
      }
    };
    loadConfig();
    return () => { mounted = false; };
  }, [activePortfolioId]);

  const handleSearch = async (query: string, type: 'etf' | 'direct', index: number) => {
    if (!query) {
      setSearchResults(prev => ({ ...prev, [`${type}-${index}`]: [] }));
      return;
    }
    try {
      const res = await api.get(`${API_ENDPOINTS.SEARCH}?q=${encodeURIComponent(query)}`);
      setSearchResults(prev => ({ ...prev, [`${type}-${index}`]: res.data }));
      setActiveSearch({ type, index });
    } catch (err) {
      console.error('Search failed', err);
    }
  };

  const selectResult = (result: SearchResult, type: 'etf' | 'direct', index: number) => {
    if (type === 'etf') {
      updateEtf(index, 'ticker', result.symbol, false);
    } else {
      updateDirect(index, 'ticker', result.symbol, false);
    }
    setSearchResults(prev => ({ ...prev, [`${type}-${index}`]: [] }));
    setActiveSearch(null);
  };

  const handleSave = async () => {
    if (!config) return;
    try {
      await api.post(API_ENDPOINTS.PORTFOLIO_CONFIG(activePortfolioId), config);
      onSave(); 
      onClose();
    } catch {
      setError('Failed to save config');
    }
  };

  const addEtf = () => {
    if (!config) return;
    // Add to direct holdings (which render last) so new items always appear at the bottom
    setConfig({ ...config, direct: [...config.direct, { ticker: '', quantity: 0, averageCost: 0, purchaseDate: '', account: '' }] });
  };

  const removeEtf = (index: number) => {
    if (!config) return;
    const newEtfs = [...config.etfs];
    newEtfs.splice(index, 1);
    setConfig({ ...config, etfs: newEtfs });
  };

  const updateEtf = (index: number, field: keyof Holding, value: any, shouldSearch = true) => {
    if (!config) return;
    const newEtfs = [...config.etfs];
    newEtfs[index] = { ...newEtfs[index], [field]: (field === 'quantity' || field === 'averageCost') ? parseFloat(value) : value };
    setConfig({ ...config, etfs: newEtfs });
    
    if (field === 'ticker' && shouldSearch) {
      handleSearch(value, 'etf', index);
    }
  };

  const removeDirect = (index: number) => {
    if (!config) return;
    const newDirect = [...config.direct];
    newDirect.splice(index, 1);
    setConfig({ ...config, direct: newDirect });
  };

  const updateDirect = (index: number, field: keyof Holding, value: any, shouldSearch = true) => {
    if (!config) return;
    const newDirect = [...config.direct];
    newDirect[index] = { ...newDirect[index], [field]: (field === 'quantity' || field === 'averageCost') ? parseFloat(value) : value };
    setConfig({ ...config, direct: newDirect });

    if (field === 'ticker' && shouldSearch) {
      handleSearch(value, 'direct', index);
    }
  };

  // Keep dialog open even while loading to prevent flash
  if (!config && loading) return (
    <Dialog open={true} onOpenChange={() => onClose()}>
      <DialogContent className="w-[1200px] h-[85vh] overflow-hidden p-0 gap-0 glass-card border-2 border-primary/30 shadow-2xl fixed left-[50%] top-[50%] translate-x-[-50%] translate-y-[-50%] z-50 rounded-2xl flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading...</p>
        </div>
      </DialogContent>
    </Dialog>
  );
  if (!config) return null;

  return (
    <Dialog open={true} onOpenChange={() => onClose()}>
      <DialogContent className="w-[1200px] h-[85vh] overflow-hidden p-0 gap-0 material-card shadow-xl fixed left-[50%] top-[50%] translate-x-[-50%] translate-y-[-50%] z-50 rounded-[28px] flex flex-col border-none">
        <DialogHeader className="bg-surface px-8 py-6 shrink-0">
          <DialogTitle className="text-3xl font-normal text-foreground flex items-center gap-3">
            <div className="p-2 rounded-full bg-primary/10">
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </div>
            <span>Edit Portfolio</span>
          </DialogTitle>
          <p className="text-sm text-muted-foreground mt-1 ml-14">Manage your holdings, ETFs, and cash positions</p>
        </DialogHeader>
        
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-8 custom-scrollbar">
          <Tabs
            value={activePortfolioId.toString()}
            onValueChange={(val) => setActivePortfolioId(val)}
            className="w-full"
          >
            <TabsList className="w-full justify-start bg-transparent p-0 h-auto gap-2 border-b border-border/40 pb-4">
              {portfolios.map((p) => (
                <TabsTrigger 
                  key={p.id} 
                  value={p.id.toString()}
                  className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:shadow-none transition-all duration-200 px-6 py-2 rounded-full font-medium text-sm border border-transparent data-[state=active]:border-primary/20 hover:bg-secondary/50"
                >
                  {p.name}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>

          <div className="bg-secondary/30 rounded-[24px] p-6">
            <div className="flex items-center gap-2 mb-4">
              <div className="p-1.5 rounded-full bg-primary/10">
                <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <Label className="text-base font-medium">Cash Position</Label>
            </div>
            <div className="flex items-center gap-3 max-w-md">
              <div className="relative flex-1">
                <span className="absolute left-4 top-3.5 text-muted-foreground font-medium">USD</span>
                <Input 
                  type="number" 
                  className="pl-12 h-12 input-material text-lg font-normal bg-white dark:bg-black/10"
                  value={config.cash || 0} 
                  onChange={(e) => setConfig({ ...config, cash: parseFloat(e.target.value) })}
                />
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-full bg-blue-500/10">
                  <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                </div>
                <h3 className="text-xl font-normal text-foreground">Holdings</h3>
              </div>
            </div>
            
            <div className="px-6 py-2 grid grid-cols-10 gap-4 text-xs font-medium text-muted-foreground uppercase tracking-wider">
              <div className="col-span-3">Ticker</div>
              <div className="col-span-2">Quantity</div>
              <div className="col-span-2">Avg. Cost</div>
              <div className="col-span-2">Purchase Date</div>
              <div className="col-span-1" />
            </div>
            
            <div className="space-y-2">
              {/* Render all ETFs first */}
              {config.etfs.map((etf, index) => (
                <div key={`etf-${index}`} className="grid grid-cols-10 gap-4 items-start bg-card hover:bg-secondary/30 transition-all duration-200 rounded-2xl p-4 group">
                  <div className="col-span-3 relative">
                  <Popover 
                    open={activeSearch?.type === 'etf' && activeSearch?.index === index}
                    onOpenChange={(open) => {
                      if (open) setActiveSearch({ type: 'etf', index });
                      else setActiveSearch(null);
                    }}
                  >
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        role="combobox"
                        aria-expanded={activeSearch?.type === 'etf' && activeSearch?.index === index}
                        className="w-full justify-start font-medium h-12 rounded-t-lg rounded-b-none border-b-2 border-transparent hover:border-primary/50 bg-secondary/30 hover:bg-secondary/50 border-x-0 border-t-0"
                      >
                        {etf.ticker || 'Select Ticker'}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[300px] p-0 bg-popover border border-border shadow-xl rounded-2xl overflow-hidden" align="start">
                      <Command shouldFilter={false} className="bg-transparent">
                        <CommandInput 
                          placeholder="Search ticker..." 
                          value={etf.ticker}
                          onValueChange={(val) => updateEtf(index, 'ticker', val)}
                          className="border-b border-border/40 h-12"
                        />
                        <CommandList className="max-h-[280px] p-2">
                          <CommandEmpty className="py-8 text-center text-muted-foreground">No results found.</CommandEmpty>
                          <CommandGroup>
                            {searchResults[`etf-${index}`]?.map((r) => (
                              <CommandItem 
                                key={r.symbol} 
                                value={r.symbol} 
                                onSelect={() => selectResult(r, 'etf', index)}
                                className="hover:bg-secondary cursor-pointer rounded-lg py-3 px-4 mb-1 aria-selected:bg-primary/10 aria-selected:text-primary"
                              >
                                <div className="flex flex-col w-full">
                                  <span className="font-semibold text-sm">{r.symbol}</span>
                                  <span className="text-xs text-muted-foreground/80 mt-0.5">{r.shortname}</span>
                                </div>
                              </CommandItem>
                            ))}
                          </CommandGroup>
                        </CommandList>
                      </Command>
                    </PopoverContent>
                  </Popover>
                  </div>
                  <div className="col-span-2">
                    <Input type="number" value={etf.quantity} onChange={(e) => updateEtf(index, 'quantity', e.target.value)} className="input-material h-12" />
                  </div>
                  <div className="col-span-2 relative">
                    <span className="absolute left-3 top-3.5 text-xs text-muted-foreground">USD</span>
                    <Input 
                      type="number" 
                      className="pl-10 input-material h-12"
                      value={etf.averageCost || 0} 
                      onChange={(e) => updateEtf(index, 'averageCost', e.target.value)} 
                    />
                  </div>
                  <div className="col-span-2">
                    <Input 
                      type="date" 
                      value={etf.purchaseDate || ''} 
                      onChange={(e) => updateEtf(index, 'purchaseDate', e.target.value)}
                      className="input-material h-12"
                    />
                  </div>
                  <div className="col-span-1 flex gap-1 items-center justify-center pt-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-10 w-10 rounded-full opacity-0 group-hover:opacity-100 transition-all duration-200 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                      onClick={() => removeEtf(index)}
                    >
                      <X className="h-5 w-5" />
                    </Button>
                  </div>
                </div>
              ))}
              
              {/* Then render all Direct Holdings */}
              {config.direct.map((item, index) => (
                <div key={`direct-${index}`} className="grid grid-cols-10 gap-4 items-start bg-card hover:bg-secondary/30 transition-all duration-200 rounded-2xl p-4 group">
                  <div className="col-span-3 relative">
                  <Popover 
                    open={activeSearch?.type === 'direct' && activeSearch?.index === index}
                    onOpenChange={(open) => {
                      if (open) setActiveSearch({ type: 'direct', index });
                      else setActiveSearch(null);
                    }}
                  >
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        role="combobox"
                        aria-expanded={activeSearch?.type === 'direct' && activeSearch?.index === index}
                        className="w-full justify-start font-medium h-12 rounded-t-lg rounded-b-none border-b-2 border-transparent hover:border-primary/50 bg-secondary/30 hover:bg-secondary/50 border-x-0 border-t-0"
                      >
                        {item.ticker || 'Select Ticker'}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[380px] p-0 material-card shadow-xl" align="start">
                      <Command shouldFilter={false} className="bg-transparent">
                        <CommandInput 
                          placeholder="Search ticker..." 
                          value={item.ticker}
                          onValueChange={(val) => updateDirect(index, 'ticker', val)}
                          className="border-b border-border/40 h-12"
                        />
                        <CommandList className="max-h-[280px] p-2">
                          <CommandEmpty className="py-8 text-center text-muted-foreground">No results found.</CommandEmpty>
                          <CommandGroup>
                            {searchResults[`direct-${index}`]?.map((r) => (
                              <CommandItem 
                                key={r.symbol} 
                                value={r.symbol} 
                                onSelect={() => selectResult(r, 'direct', index)}
                                className="py-3 px-4 rounded-full cursor-pointer transition-all duration-200 hover:bg-primary/10 aria-selected:bg-primary/15"
                              >
                                <div className="flex flex-col w-full">
                                  <span className="font-semibold text-sm">{r.symbol}</span>
                                  <span className="text-xs text-muted-foreground/80 mt-0.5">{r.shortname}</span>
                                </div>
                              </CommandItem>
                            ))}
                          </CommandGroup>
                        </CommandList>
                      </Command>
                    </PopoverContent>
                  </Popover>
                  </div>
                  <div className="col-span-2">
                    <Input type="number" value={item.quantity} onChange={(e) => updateDirect(index, 'quantity', e.target.value)} className="input-material h-12" />
                  </div>
                  <div className="col-span-2 relative">
                    <span className="absolute left-3 top-3.5 text-xs text-muted-foreground">USD</span>
                    <Input 
                      type="number" 
                      className="pl-10 input-material h-12"
                      value={item.averageCost || 0} 
                      onChange={(e) => updateDirect(index, 'averageCost', e.target.value)} 
                    />
                  </div>
                  <div className="col-span-2">
                    <Input type="date" value={item.purchaseDate || ''} onChange={(e) => updateDirect(index, 'purchaseDate', e.target.value)} className="input-material h-12" />
                  </div>
                  <div className="col-span-1 flex gap-1 items-center justify-center pt-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-10 w-10 rounded-full opacity-0 group-hover:opacity-100 transition-all duration-200 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                      onClick={() => removeDirect(index)}
                    >
                      <X className="h-5 w-5" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
            
            <Button 
              variant="ghost" 
              onClick={addEtf} 
              className="w-full border border-dashed border-border hover:border-primary hover:bg-primary/5 transition-all duration-300 h-12 rounded-full font-medium group flex items-center justify-center gap-2 mt-4"
            >
              <Plus className="h-5 w-5 group-hover:rotate-90 transition-transform duration-300" /> 
              <span>Add Ticker</span>
            </Button>
          </div>
        </div>

        {/* Fixed Footer */}
        <div className="border-t border-border/40 px-8 py-6 bg-surface shrink-0">
          <div className="flex justify-end gap-4">
            <Button 
              variant="ghost" 
              onClick={onClose}
              className="px-8 h-12 rounded-full font-medium hover:bg-secondary"
            >
              Cancel
            </Button>
            <Button 
              onClick={handleSave} 
              className="btn-material bg-primary text-primary-foreground px-8 h-12 shadow-lg shadow-primary/20 hover:shadow-primary/30 flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span>Save Changes</span>
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export default PortfolioEditor;
