import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Progress } from './ui/progress';
import { 
  Play, 
  Square, 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Shield,
  Zap,
  BarChart3,
  Settings,
  RefreshCw
} from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API_BASE = `${BACKEND_URL}/api`;

const TradingDashboard = () => {
  const [engineStatus, setEngineStatus] = useState(null);
  const [portfolios, setPortfolios] = useState([]);
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchEngineStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/trading/engine/status`);
      if (!response.ok) throw new Error('Failed to fetch engine status');
      const data = await response.json();
      setEngineStatus(data);
    } catch (err) {
      setError('Failed to fetch engine status: ' + err.message);
    }
  }, []);

  const fetchPortfolios = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/trading/portfolios`);
      if (!response.ok) throw new Error('Failed to fetch portfolios');
      const data = await response.json();
      setPortfolios(data.portfolios || []);
    } catch (err) {
      console.error('Failed to fetch portfolios:', err);
    }
  }, []);

  const fetchSignals = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/trading/signals?limit=10`);
      if (!response.ok) throw new Error('Failed to fetch signals');
      const data = await response.json();
      setSignals(data);
    } catch (err) {
      console.error('Failed to fetch signals:', err);
    }
  }, []);

  const fetchAllData = useCallback(async () => {
    setLoading(true);
    try {
      await Promise.all([fetchEngineStatus(), fetchPortfolios(), fetchSignals()]);
    } finally {
      setLoading(false);
    }
  }, [fetchEngineStatus, fetchPortfolios, fetchSignals]);

  useEffect(() => {
    fetchAllData();
  }, [fetchAllData]);

  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(fetchAllData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [autoRefresh, fetchAllData]);

  const startEngine = async () => {
    try {
      const response = await fetch(`${API_BASE}/trading/engine/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
          paper_trading: true
        })
      });
      
      if (!response.ok) throw new Error('Failed to start engine');
      
      // Refresh status after starting
      setTimeout(fetchEngineStatus, 2000);
    } catch (err) {
      setError('Failed to start engine: ' + err.message);
    }
  };

  const stopEngine = async () => {
    try {
      const response = await fetch(`${API_BASE}/trading/engine/stop`, {
        method: 'POST'
      });
      
      if (!response.ok) throw new Error('Failed to stop engine');
      
      setTimeout(fetchEngineStatus, 2000);
    } catch (err) {
      setError('Failed to stop engine: ' + err.message);
    }
  };

  const createPortfolio = async () => {
    try {
      const response = await fetch(`${API_BASE}/trading/portfolios`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'demo_user',
          name: 'Demo Portfolio',
          initial_balance: 10000
        })
      });
      
      if (!response.ok) throw new Error('Failed to create portfolio');
      
      await fetchPortfolios();
    } catch (err) {
      setError('Failed to create portfolio: ' + err.message);
    }
  };

  if (loading && !engineStatus) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-lg">Loading FinGPT Trading System...</p>
        </div>
      </div>
    );
  }

  const getStatusColor = (isRunning) => {
    return isRunning ? 'bg-green-500' : 'bg-red-500';
  };

  const getSignalIcon = (action) => {
    return action === 'buy' ? (
      <TrendingUp className="h-4 w-4 text-green-500" />
    ) : (
      <TrendingDown className="h-4 w-4 text-red-500" />
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center">
              <Zap className="h-8 w-8 mr-3 text-blue-600" />
              FinGPT Trading System
            </h1>
            <p className="text-gray-600 mt-2">
              AI-Powered Trading System optimized for RTX 4050 (6GB VRAM)
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            <Button
              variant={autoRefresh ? "default" : "outline"}
              onClick={() => setAutoRefresh(!autoRefresh)}
              size="sm"
            >
              <Refresh className="h-4 w-4 mr-2" />
              Auto Refresh
            </Button>
            
            <Button onClick={fetchAllData} variant="outline" size="sm">
              <Refresh className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {error && (
          <Alert className="mt-4 border-red-200 bg-red-50">
            <AlertDescription className="text-red-800">
              {error}
            </AlertDescription>
          </Alert>
        )}
      </div>

      {/* Engine Status */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
        <Card className="lg:col-span-1">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Engine Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${getStatusColor(engineStatus?.is_running)}`} />
              <span className="font-semibold">
                {engineStatus?.is_running ? 'Running' : 'Stopped'}
              </span>
            </div>
            
            <div className="mt-4 space-y-2">
              <Button
                onClick={engineStatus?.is_running ? stopEngine : startEngine}
                className={engineStatus?.is_running ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
                size="sm"
              >
                {engineStatus?.is_running ? (
                  <>
                    <Square className="h-4 w-4 mr-2" />
                    Stop Engine
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Engine
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Key Metrics */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Active Symbols</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{engineStatus?.subscribed_symbols?.length || 0}</div>
            <div className="text-xs text-gray-500 mt-1">
              {engineStatus?.subscribed_symbols?.join(', ') || 'None'}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Active Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{engineStatus?.total_positions || 0}</div>
            <div className="text-xs text-gray-500 mt-1">Portfolio positions</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Signals Generated</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{engineStatus?.signals_generated || 0}</div>
            <div className="text-xs text-gray-500 mt-1">Total signals</div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Portfolios */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <DollarSign className="h-5 w-5 mr-2" />
              Portfolios
            </CardTitle>
          </CardHeader>
          <CardContent>
            {portfolios.length === 0 ? (
              <div className="text-center py-6">
                <p className="text-gray-500 mb-4">No portfolios created yet</p>
                <Button onClick={createPortfolio} size="sm">
                  Create Demo Portfolio
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {portfolios.map((portfolio) => (
                  <div key={portfolio.portfolio_id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold">{portfolio.name}</h3>
                      <Badge variant="outline">{portfolio.total_trades} trades</Badge>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">Balance:</span>
                        <div className="font-medium">${portfolio.current_balance?.toFixed(2)}</div>
                      </div>
                      <div>
                        <span className="text-gray-500">P&L:</span>
                        <div className={`font-medium ${portfolio.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          ${portfolio.total_pnl?.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Signals */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart3 className="h-5 w-5 mr-2" />
              Recent Signals
            </CardTitle>
          </CardHeader>
          <CardContent>
            {signals.length === 0 ? (
              <div className="text-center py-6">
                <p className="text-gray-500">No signals generated yet</p>
                {!engineStatus?.is_running && (
                  <p className="text-xs text-gray-400 mt-2">Start the engine to generate signals</p>
                )}
              </div>
            ) : (
              <div className="space-y-3">
                {signals.map((signal, index) => (
                  <div key={index} className="border rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {getSignalIcon(signal.action)}
                        <span className="font-medium">{signal.symbol}</span>
                        <Badge variant={signal.action === 'buy' ? 'default' : 'destructive'}>
                          {signal.action.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(signal.created_at).toLocaleTimeString()}
                      </div>
                    </div>
                    
                    <div className="text-xs text-gray-600">
                      <div className="flex justify-between mb-1">
                        <span>Confidence:</span>
                        <span className="font-medium">{(signal.confidence * 100).toFixed(1)}%</span>
                      </div>
                      {signal.price_target && (
                        <div className="flex justify-between mb-1">
                          <span>Target:</span>
                          <span className="font-medium">${signal.price_target.toFixed(4)}</span>
                        </div>
                      )}
                      <div className="text-xs text-gray-500 mt-2">
                        Strategy: {signal.strategy_used}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Strategy Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Shield className="h-5 w-5 mr-2" />
              Strategy Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            {engineStatus?.strategy_status ? (
              <div className="space-y-4">
                {Object.entries(engineStatus.strategy_status).map(([name, stats]) => (
                  <div key={name} className="border rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium capitalize">{name.replace('_', ' ')}</h3>
                      <Badge variant="outline">
                        {stats.trades_today}/{stats.max_trades}
                      </Badge>
                    </div>
                    
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Win Rate:</span>
                        <span className="font-medium">
                          {(stats.performance.win_rate * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-gray-500">Signals:</span>
                        <span className="font-medium">{stats.performance.total_signals}</span>
                      </div>
                      
                      <div className="flex justify-between">
                        <span className="text-gray-500">Avg Confidence:</span>
                        <span className="font-medium">
                          {(stats.performance.avg_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="mt-2">
                        <div className="flex justify-between text-xs mb-1">
                          <span>Daily Trades</span>
                          <span>{stats.trades_today}/{stats.max_trades}</span>
                        </div>
                        <Progress 
                          value={(stats.trades_today / stats.max_trades) * 100} 
                          className="h-1"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6">
                <p className="text-gray-500">Strategy data unavailable</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Storage Stats */}
      {engineStatus?.storage_stats && (
        <Card className="mt-6">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Settings className="h-5 w-5 mr-2" />
              System Resources
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div>
                <div className="text-sm text-gray-500">Storage Used</div>
                <div className="text-lg font-semibold">
                  {engineStatus.storage_stats.total_size_mb?.toFixed(2)} MB
                </div>
                <Progress 
                  value={engineStatus.storage_stats.usage_percentage} 
                  className="mt-2"
                />
              </div>
              
              <div>
                <div className="text-sm text-gray-500">Data Points</div>
                <div className="text-lg font-semibold">
                  {engineStatus.data_points_processed?.toLocaleString()}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-gray-500">Compression Ratio</div>
                <div className="text-lg font-semibold">
                  {engineStatus.storage_stats.avg_compression_ratio?.toFixed(1)}x
                </div>
              </div>
              
              <div>
                <div className="text-sm text-gray-500">Mode</div>
                <div className="text-lg font-semibold flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${engineStatus.is_paper_trading ? 'bg-blue-500' : 'bg-green-500'}`} />
                  {engineStatus.is_paper_trading ? 'Paper Trading' : 'Live Trading'}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default TradingDashboard;