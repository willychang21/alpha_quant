import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation, useNavigate, Routes, Route, Navigate } from 'react-router-dom';
import HoldingsTable from './components/HoldingsTable';
import ProfitLossTable from './components/ProfitLossTable'
import BubbleAnalysis from './components/BubbleAnalysis';
import BacktestAnalysis from './components/BacktestAnalysis';
import PortfolioEditor from './components/PortfolioEditor';
import PortfolioSelector from './components/PortfolioSelector';
import DcaAnalysisComponent from './components/DCAAnalysis';
import FundamentalAnalysis from './components/FundamentalAnalysis';
import { QuantDashboard } from './components/QuantDashboard';
import SignalsDashboard from './components/SignalsDashboard';
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowUp, ArrowDown } from 'lucide-react';
import { useStore } from './store/useStore';

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<string>('matrix');
  const [showEditor, setShowEditor] = useState<boolean>(false);
  
  const { 
    portfolios, 
    currentPortfolioId, 
    data, 
    loading, 
    error, 
    dcaConfig,
    fetchPortfolios,
    fetchData,
    createPortfolio,
    renamePortfolio,
    deletePortfolio,
    setCurrentPortfolioId,
    setDcaConfig
  } = useStore();

  // Sync URL with activeTab
  useEffect(() => {
    const path = location.pathname.substring(1).split('/')[0]; // get first segment
    if (path === '' || path === 'matrix') setActiveTab('matrix');
    else if (path === 'pl') setActiveTab('pl');
    else if (path === 'dca') setActiveTab('dca');
    else if (path === 'backtest') setActiveTab('backtest');
    else if (path === 'fundamental') setActiveTab('fundamental');
    else if (path === 'bubble') setActiveTab('bubble');
    else if (path === 'quant') setActiveTab('quant');
    else if (path === 'signals') setActiveTab('signals');
  }, [location]);

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    navigate(`/${tab === 'matrix' ? '' : tab}`);
  };

  useEffect(() => {
    fetchPortfolios();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (currentPortfolioId) {
      fetchData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPortfolioId]);

  if (loading && !data && currentPortfolioId) {
    return (
      <div className="min-h-screen flex justify-center items-center">
        <div className="text-center space-y-4 animate-scale-in">
          <div className="w-16 h-16 mx-auto">
            <div className="w-16 h-16 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
          </div>
          <div className="text-lg font-medium text-muted-foreground">Loading portfolio data...</div>
          <div className="text-sm text-muted-foreground">Fetching live market prices</div>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="container mx-auto mt-20">
        <div className="glass-card rounded-2xl p-12 text-center max-w-md mx-auto animate-scale-in">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold mb-2">Error Loading Data</h3>
          <p className="text-muted-foreground">{error}</p>
          <Button onClick={fetchData} className="mt-6">
            Try Again
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
      <div className="container mx-auto p-4 sm:p-6 space-y-8 animate-fade-in max-w-7xl">
        {/* Top App Bar Style Header */}
        <header className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4 py-2">
          <div className="flex-1">
            <h1 className="text-4xl font-normal tracking-tight text-foreground mb-1">
              My Portfolio
            </h1>
            <div className="text-sm text-muted-foreground flex items-center gap-2 font-medium">
              <span className="inline-block w-2 h-2 rounded-full bg-success animate-pulse"></span>
              Last Updated • {data ? new Date(data.lastUpdated).toLocaleString() : '-'}
            </div>
          </div>
          <div className="flex items-center gap-3">
            <PortfolioSelector 
              currentPortfolioId={currentPortfolioId}
              portfolios={portfolios}
              onSelect={setCurrentPortfolioId}
              onCreate={createPortfolio}
              onRename={renamePortfolio}
              onDelete={deletePortfolio}
            />
            <Button 
              onClick={() => setShowEditor(true)} 
              className="btn-material bg-primary text-primary-foreground hover:bg-primary/90 px-6 h-12 shadow-sm hover:shadow-md transition-all"
            >
              <span className="flex items-center font-medium tracking-wide">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              Edit Portfolio
              </span>
            </Button>
          </div>
        </header>

        {/* Summary Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card className="material-card p-6 flex flex-col justify-between h-full">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-muted-foreground">Total Value</span>
              <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                <span className="text-lg font-sans">$</span>
              </div>
            </div>
            <div>
              <div className="text-3xl font-normal tracking-tight font-mono text-foreground">
                {data?.summary ? `$${data.summary.totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '-'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">Total portfolio value</p>
            </div>
          </Card>

          <Card className="material-card p-6 flex flex-col justify-between h-full">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-muted-foreground">Day Change</span>
              <div className={`w-10 h-10 rounded-full flex items-center justify-center ${data?.summary && data.summary.dayChange >= 0 ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}>
                {data?.summary && data.summary.dayChange >= 0 ? (
                  <ArrowUp className="h-5 w-5" />
                ) : (
                  <ArrowDown className="h-5 w-5" />
                )}
              </div>
            </div>
            <div>
              <div className={`text-3xl font-normal tracking-tight font-mono ${data?.summary && data.summary.dayChange >= 0 ? 'text-success' : 'text-danger'}`}>
                {data?.summary ? (
                  <>
                    {data.summary.dayChange >= 0 ? '+' : ''}{data.summary.dayChange.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </>
                ) : '-'}
              </div>
              <p className={`text-sm font-medium mt-1 ${data?.summary && data.summary.dayChange >= 0 ? 'text-success' : 'text-danger'}`}>
                {data?.summary ? `${data.summary.dayChangePercent >= 0 ? '+' : ''}${data.summary.dayChangePercent.toFixed(2)}%` : '-'}
              </p>
            </div>
          </Card>

          <Card className="material-card p-6 flex flex-col justify-between h-full">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-muted-foreground">P/L (All)</span>
              <div className="w-10 h-10 rounded-full bg-accent/10 flex items-center justify-center text-accent">
                <span className="text-lg font-bold">%</span>
              </div>
            </div>
            <div>
              <div className="text-3xl font-normal tracking-tight font-mono text-foreground">$0.00</div>
              <p className="text-xs text-muted-foreground mt-1">Coming soon</p>
            </div>
          </Card>

          <Card className="material-card p-6 flex flex-col justify-between h-full">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-muted-foreground">Holdings</span>
              <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
              </div>
            </div>
            <div>
              <div className="text-3xl font-normal tracking-tight font-mono text-foreground">
                {data?.summary ? data.summary.holdingsCount : '-'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">Unique positions</p>
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <div className="relative">
            <div className="p-1.5 h-auto flex flex-row bg-muted/50 backdrop-blur-sm rounded-full relative z-10 w-fit mx-auto sm:mx-0 border border-white/10">
              {['pl', 'matrix', 'dca', 'backtest', 'fundamental', 'bubble', 'quant', 'signals'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => handleTabChange(tab)}
                  className={`relative px-6 py-2.5 rounded-full font-medium text-sm flex flex-row items-center gap-2 transition-colors duration-200 ${
                    activeTab === tab ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {activeTab === tab && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute inset-0 bg-white dark:bg-card rounded-full shadow-sm"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                  <span className="relative z-10">
                    {tab === 'pl' && 'Profit/Loss'}
                    {tab === 'matrix' && 'Holdings Matrix'}
                    {tab === 'dca' && 'DCA Analysis'}
                    {tab === 'backtest' && 'Performance'}
                    {tab === 'fundamental' && 'Fundamental'}
                    {tab === 'bubble' && 'Market Bubble'}
                    {tab === 'quant' && 'Quant Strategy'}
                    {tab === 'signals' && 'Trading Signals'}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <AnimatePresence mode="wait">
             <Routes>
                <Route path="/" element={
                  <motion.div
                    key="matrix"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-4"
                  >
                    {data && <HoldingsTable data={data} />}
                  </motion.div>
                } />
                <Route path="/matrix" element={<Navigate to="/" replace />} />
                
                <Route path="/pl" element={
                  <motion.div
                    key="pl"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-4"
                  >
                    {data && <ProfitLossTable data={data} />}
                  </motion.div>
                } />

                <Route path="/dca" element={
                  <motion.div
                    key="dca"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-6"
                  >
                    <DcaAnalysisComponent dcaConfig={dcaConfig} setDcaConfig={setDcaConfig} />
                  </motion.div>
                } />

                <Route path="/backtest" element={
                  <motion.div
                    key="backtest"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-6"
                  >
                    <div className="material-card p-8">
                      <div className="mb-8">
                        <h2 className="text-2xl font-normal tracking-tight mb-2">Hypothetical Performance</h2>
                        <p className="text-muted-foreground max-w-2xl">
                          Backtest your current DCA allocation against historical market data. 
                          Assumes a $10,000 initial investment with monthly rebalancing.
                        </p>
                      </div>
                      <BacktestAnalysis dcaConfig={dcaConfig} />
                    </div>
                  </motion.div>
                } />

                <Route path="/fundamental" element={
                  <motion.div
                    key="fundamental"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-6"
                  >
                    <FundamentalAnalysis />
                  </motion.div>
                } />
                
                <Route path="/fundamental/:ticker" element={
                  <motion.div
                    key="fundamental-ticker"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-6"
                  >
                    <FundamentalAnalysis />
                  </motion.div>
                } />

                <Route path="/bubble" element={
                  <motion.div
                    key="bubble"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-6"
                  >
                    <BubbleAnalysis />
                  </motion.div>
                } />

                <Route path="/quant" element={
                  <motion.div
                    key="quant"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-6"
                  >
                    <QuantDashboard />
                  </motion.div>
                } />

                <Route path="/signals" element={
                  <motion.div
                    key="signals"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="space-y-6"
                  >
                    <SignalsDashboard />
                  </motion.div>
                } />
             </Routes>
          </AnimatePresence>
        </div>

        {showEditor && currentPortfolioId && (
          <PortfolioEditor 
            portfolioId={currentPortfolioId}
            onClose={() => setShowEditor(false)} 
            onSave={() => {
              fetchData(); // Refresh data after save
            }} 
          />
        )}
      </div>
    </div>
  );
}

export default App;
