"""Supply Chain GNN Module.

Models supply chain relationships using a graph neural network approach.
Captures momentum spillover effects between related companies.

Key features:
- Graph structure for customer-supplier relationships
- Message passing algorithm for signal propagation
- Decay factor for time-based signal attenuation
- Incremental graph updates

Example:
    >>> graph = SupplyChainGraph()
    >>> graph.add_edge('AAPL', 'TSM', weight=0.8)  # TSM supplies to AAPL
    >>> momentum = SupplyChainMomentum(graph)
    >>> scores = momentum.compute({'TSM': 0.05}, ['AAPL', 'TSM'])
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SupplyChainEdge:
    """Represents a supply chain relationship.
    
    Attributes:
        supplier: Ticker of the supplying company.
        customer: Ticker of the customer company.
        weight: Relationship strength (0-1), e.g., revenue dependency.
    """
    supplier: str
    customer: str
    weight: float = 1.0


class SupplyChainGraph:
    """Graph structure for supply chain relationships.
    
    Uses adjacency lists for efficient neighbor lookup.
    Supports both forward (supplier → customer) and reverse lookups.
    
    Attributes:
        edges: List of all supply chain edges.
    """
    
    def __init__(self):
        """Initialize empty supply chain graph."""
        self.edges: List[SupplyChainEdge] = []
        # ticker → [(neighbor, weight)]
        self._adjacency: Dict[str, List[Tuple[str, float]]] = {}
        self._reverse_adjacency: Dict[str, List[Tuple[str, float]]] = {}
    
    def add_edge(
        self, 
        supplier: str, 
        customer: str, 
        weight: float = 1.0
    ) -> None:
        """Add a supply chain relationship.
        
        Args:
            supplier: Ticker of the supplying company.
            customer: Ticker of the customer company.
            weight: Relationship strength (0-1).
        """
        edge = SupplyChainEdge(supplier, customer, weight)
        self.edges.append(edge)
        
        # Update forward adjacency (supplier → customers)
        if supplier not in self._adjacency:
            self._adjacency[supplier] = []
        self._adjacency[supplier].append((customer, weight))
        
        # Update reverse adjacency (customer → suppliers)
        if customer not in self._reverse_adjacency:
            self._reverse_adjacency[customer] = []
        self._reverse_adjacency[customer].append((supplier, weight))
    
    def get_customers(self, ticker: str) -> List[Tuple[str, float]]:
        """Get downstream customers of a company.
        
        Args:
            ticker: Supplier company ticker.
            
        Returns:
            List of (customer_ticker, weight) tuples.
        """
        return self._adjacency.get(ticker, [])
    
    def get_suppliers(self, ticker: str) -> List[Tuple[str, float]]:
        """Get upstream suppliers of a company.
        
        Args:
            ticker: Customer company ticker.
            
        Returns:
            List of (supplier_ticker, weight) tuples.
        """
        return self._reverse_adjacency.get(ticker, [])
    
    def get_all_neighbors(self, ticker: str) -> List[Tuple[str, float]]:
        """Get all connected companies (both directions).
        
        Args:
            ticker: Company ticker.
            
        Returns:
            List of (neighbor_ticker, weight) tuples.
        """
        customers = self.get_customers(ticker)
        suppliers = self.get_suppliers(ticker)
        return customers + suppliers
    
    def get_all_tickers(self) -> List[str]:
        """Get all unique tickers in the graph.
        
        Returns:
            List of all ticker symbols.
        """
        tickers = set()
        for edge in self.edges:
            tickers.add(edge.supplier)
            tickers.add(edge.customer)
        return list(tickers)
    
    def get_edge_count(self) -> int:
        """Get number of edges in the graph.
        
        Returns:
            Number of supply chain relationships.
        """
        return len(self.edges)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'SupplyChainGraph':
        """Load graph from DataFrame.
        
        Expected columns: supplier, customer, weight (optional).
        
        Args:
            df: DataFrame with relationship data.
            
        Returns:
            Populated SupplyChainGraph instance.
        """
        graph = cls()
        
        for _, row in df.iterrows():
            graph.add_edge(
                supplier=str(row['supplier']),
                customer=str(row['customer']),
                weight=float(row.get('weight', 1.0))
            )
        
        logger.info(f"Loaded supply chain graph: {len(graph.edges)} edges")
        return graph
    
    @classmethod
    def from_csv(cls, path: str) -> 'SupplyChainGraph':
        """Load graph from CSV file.
        
        Args:
            path: Path to CSV file.
            
        Returns:
            Populated SupplyChainGraph instance.
        """
        df = pd.read_csv(path)
        return cls.from_dataframe(df)


class SupplyChainMomentum:
    """Computes supply chain momentum factor using message passing.
    
    Propagates price signals through the supply chain graph with decay.
    Companies connected to stocks with strong momentum receive spillover signals.
    
    Attributes:
        graph: Supply chain relationship graph.
        decay_factor: Signal decay per propagation step (0-1).
        propagation_steps: Number of message passing iterations.
    """
    
    def __init__(
        self,
        graph: SupplyChainGraph,
        decay_factor: float = 0.5,
        propagation_steps: int = 2
    ):
        """Initialize SupplyChainMomentum.
        
        Args:
            graph: SupplyChainGraph with company relationships.
            decay_factor: Signal decay factor (0.3-0.7 typical).
            propagation_steps: Number of hops to propagate signals.
        """
        self.graph = graph
        self.decay_factor = decay_factor
        self.propagation_steps = propagation_steps
    
    def compute(
        self,
        price_changes: Dict[str, float],
        tickers: List[str]
    ) -> Dict[str, float]:
        """Compute supply chain momentum for each ticker.
        
        Uses message passing algorithm:
        1. Initialize signals with own price changes
        2. For each step, aggregate neighbor signals with decay
        3. Return final momentum scores
        
        Args:
            price_changes: Dict of ticker → recent price change (%).
            tickers: List of tickers to compute momentum for.
            
        Returns:
            Dict of ticker → supply chain momentum score.
        """
        # Initialize signals with own price changes
        signals = {t: price_changes.get(t, 0.0) for t in tickers}
        
        # Message passing iterations
        for step in range(self.propagation_steps):
            new_signals = {}
            decay = self.decay_factor ** (step + 1)
            
            for ticker in tickers:
                # Aggregate neighbor signals
                neighbors = self.graph.get_all_neighbors(ticker)
                
                if not neighbors:
                    # No connections, keep own signal
                    new_signals[ticker] = signals.get(ticker, 0.0)
                    continue
                
                neighbor_signal = 0.0
                total_weight = 0.0
                
                for neighbor, weight in neighbors:
                    if neighbor in signals:
                        neighbor_signal += weight * signals[neighbor]
                        total_weight += weight
                
                if total_weight > 0:
                    neighbor_signal /= total_weight
                
                # Combine own signal with neighbor signal
                own_signal = signals.get(ticker, 0.0)
                new_signals[ticker] = own_signal + decay * neighbor_signal
            
            signals = new_signals
        
        return signals
    
    def compute_factor(
        self,
        history_df: pd.DataFrame,
        lookback: int = 5
    ) -> pd.Series:
        """Compute supply chain momentum factor from price history.
        
        Convenience method for integration with factor pipeline.
        
        Args:
            history_df: DataFrame with columns: ticker, date, close.
            lookback: Days to compute price change over.
            
        Returns:
            Series of momentum scores indexed by ticker.
        """
        # Compute recent price changes
        try:
            latest = history_df.groupby('ticker')['close'].last()
            lookback_prices = history_df.groupby('ticker')['close'].nth(-lookback)
            
            # Handle case where nth returns NaN
            valid_mask = ~latest.isna() & ~lookback_prices.isna() & (lookback_prices != 0)
            
            price_changes = pd.Series(0.0, index=latest.index)
            price_changes[valid_mask] = (
                (latest[valid_mask] - lookback_prices[valid_mask]) 
                / lookback_prices[valid_mask] * 100
            )
            
            price_changes_dict = price_changes.to_dict()
            tickers = list(price_changes_dict.keys())
            
            # Compute supply chain momentum
            sc_momentum = self.compute(price_changes_dict, tickers)
            
            return pd.Series(sc_momentum)
            
        except Exception as e:
            logger.warning(f"Supply chain factor computation failed: {e}")
            return pd.Series(dtype=float)
    
    def get_signal_for_ticker(
        self,
        ticker: str,
        price_changes: Dict[str, float]
    ) -> float:
        """Get supply chain momentum for a single ticker.
        
        Args:
            ticker: Company ticker.
            price_changes: Dict of ticker → price change.
            
        Returns:
            Supply chain momentum score (0.0 if not in graph).
        """
        if ticker not in price_changes and ticker not in self.graph._adjacency:
            return 0.0
        
        result = self.compute(price_changes, [ticker])
        return result.get(ticker, 0.0)


@dataclass
class TemporalAttentionWeights:
    """Attention weights across different time horizons.
    
    Used in TemporalAttentionGNN to weight signals from different time periods.
    The weights indicate which historical period is most relevant for a given edge.
    
    Attributes:
        weights_1d: Attention weight for 1-day price changes.
        weights_5d: Attention weight for 5-day price changes.
        weights_21d: Attention weight for 21-day price changes.
    """
    weights_1d: float = 1/3
    weights_5d: float = 1/3
    weights_21d: float = 1/3
    
    def normalize(self) -> 'TemporalAttentionWeights':
        """Normalize weights to sum to 1.0.
        
        Returns:
            New TemporalAttentionWeights with normalized weights.
        """
        total = self.weights_1d + self.weights_5d + self.weights_21d
        if total > 1e-10:
            return TemporalAttentionWeights(
                weights_1d=self.weights_1d / total,
                weights_5d=self.weights_5d / total,
                weights_21d=self.weights_21d / total
            )
        # Default to uniform if all zero
        return TemporalAttentionWeights(1/3, 1/3, 1/3)
    
    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            'weights_1d': self.weights_1d,
            'weights_5d': self.weights_5d,
            'weights_21d': self.weights_21d
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'TemporalAttentionWeights':
        """Deserialize from dictionary."""
        return cls(
            weights_1d=data.get('weights_1d', 1/3),
            weights_5d=data.get('weights_5d', 1/3),
            weights_21d=data.get('weights_21d', 1/3)
        )


class TemporalAttentionGNN(SupplyChainMomentum):
    """Supply chain momentum with temporal attention mechanism.
    
    Extends SupplyChainMomentum to incorporate attention weights across
    multiple time horizons (1-day, 5-day, 21-day). This allows the model
    to dynamically weight signal propagation based on which time horizon
    is most informative for each ticker's supply chain relationships.
    
    The attention mechanism addresses the observation that supply chain
    signal propagation speed varies by industry and relationship type.
    
    Attributes:
        graph: Supply chain relationship graph.
        decay_factor: Base signal decay per step.
        propagation_steps: Number of message passing iterations.
        attention_hidden_dim: Hidden dimension for attention computation.
    """
    
    def __init__(
        self,
        graph: SupplyChainGraph,
        decay_factor: float = 0.5,
        propagation_steps: int = 2,
        attention_hidden_dim: int = 16
    ):
        """Initialize TemporalAttentionGNN.
        
        Args:
            graph: SupplyChainGraph with company relationships.
            decay_factor: Base signal decay factor.
            propagation_steps: Number of hops to propagate signals.
            attention_hidden_dim: Hidden dimension for attention MLP (optional).
        """
        super().__init__(graph, decay_factor, propagation_steps)
        self.attention_hidden_dim = attention_hidden_dim
        
        # Learnable attention parameters (stored per ticker)
        self._attention_weights: Dict[str, TemporalAttentionWeights] = {}
        
        # Running statistics for attention learning
        self._signal_history: Dict[str, Dict[str, List[float]]] = {}
    
    def compute_with_attention(
        self,
        price_changes_1d: Dict[str, float],
        price_changes_5d: Dict[str, float],
        price_changes_21d: Dict[str, float],
        tickers: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, TemporalAttentionWeights]]:
        """Compute momentum with temporal attention weights.
        
        Blends signals from multiple time horizons using learned attention
        weights, then propagates through the supply chain graph.
        
        Args:
            price_changes_1d: Dict of ticker → 1-day price change.
            price_changes_5d: Dict of ticker → 5-day price change.
            price_changes_21d: Dict of ticker → 21-day price change.
            tickers: List of tickers to compute momentum for.
            
        Returns:
            Tuple of:
            - Dict of ticker → supply chain momentum score
            - Dict of ticker → TemporalAttentionWeights used
        """
        attention_weights = {}
        blended_signals = {}
        
        for ticker in tickers:
            # Get or compute attention weights for this ticker
            attn = self._compute_temporal_attention(
                ticker, 
                price_changes_1d, 
                price_changes_5d, 
                price_changes_21d
            )
            attn = attn.normalize()
            attention_weights[ticker] = attn
            self._attention_weights[ticker] = attn
            
            # Blend signals across time horizons
            signal = (
                attn.weights_1d * price_changes_1d.get(ticker, 0.0) +
                attn.weights_5d * price_changes_5d.get(ticker, 0.0) +
                attn.weights_21d * price_changes_21d.get(ticker, 0.0)
            )
            blended_signals[ticker] = signal
        
        # Propagate through graph with attention-weighted decay
        propagated = self._propagate_with_attention(blended_signals, attention_weights)
        
        return propagated, attention_weights
    
    def _compute_temporal_attention(
        self,
        ticker: str,
        price_changes_1d: Dict[str, float],
        price_changes_5d: Dict[str, float],
        price_changes_21d: Dict[str, float]
    ) -> TemporalAttentionWeights:
        """Compute attention weights for a ticker based on signal quality.
        
        Uses a simple heuristic: higher weight to time horizons with larger
        absolute signal magnitude (assuming stronger signals are more informative).
        
        In a production setting, this could be replaced with learned MLP weights.
        
        Args:
            ticker: Company ticker.
            price_changes_1d: 1-day price changes.
            price_changes_5d: 5-day price changes.
            price_changes_21d: 21-day price changes.
            
        Returns:
            Attention weights for the ticker.
        """
        # Use cached weights if available
        if ticker in self._attention_weights:
            return self._attention_weights[ticker]
        
        # Get signals for neighbors (supply chain influence)
        neighbors = self.graph.get_all_neighbors(ticker)
        
        if not neighbors:
            # No supply chain context, use uniform weights
            return TemporalAttentionWeights(1/3, 1/3, 1/3)
        
        # Compute signal magnitude from neighbors for each horizon
        mag_1d = sum(abs(price_changes_1d.get(n, 0.0)) * w for n, w in neighbors)
        mag_5d = sum(abs(price_changes_5d.get(n, 0.0)) * w for n, w in neighbors)
        mag_21d = sum(abs(price_changes_21d.get(n, 0.0)) * w for n, w in neighbors)
        
        # Convert to softmax-like weights
        total = mag_1d + mag_5d + mag_21d + 1e-10
        
        return TemporalAttentionWeights(
            weights_1d=mag_1d / total + 1e-3,  # Small floor for stability
            weights_5d=mag_5d / total + 1e-3,
            weights_21d=mag_21d / total + 1e-3
        )
    
    def _propagate_with_attention(
        self,
        signals: Dict[str, float],
        attention_weights: Dict[str, TemporalAttentionWeights]
    ) -> Dict[str, float]:
        """Propagate signals through graph with attention-weighted decay.
        
        Similar to parent class compute(), but uses attention weights to
        modulate the decay factor for each ticker.
        
        Args:
            signals: Initial signals (blended across time horizons).
            attention_weights: Attention weights per ticker.
            
        Returns:
            Propagated signal scores.
        """
        current_signals = signals.copy()
        
        for step in range(self.propagation_steps):
            new_signals = {}
            base_decay = self.decay_factor ** (step + 1)
            
            for ticker in signals.keys():
                neighbors = self.graph.get_all_neighbors(ticker)
                
                if not neighbors:
                    new_signals[ticker] = current_signals.get(ticker, 0.0)
                    continue
                
                neighbor_signal = 0.0
                total_weight = 0.0
                
                for neighbor, edge_weight in neighbors:
                    if neighbor in current_signals:
                        # Modulate by attention (higher attention = more influence)
                        neighbor_attn = attention_weights.get(
                            neighbor, 
                            TemporalAttentionWeights()
                        )
                        # Use max attention weight as influence multiplier
                        attn_factor = max(
                            neighbor_attn.weights_1d,
                            neighbor_attn.weights_5d,
                            neighbor_attn.weights_21d
                        )
                        neighbor_signal += edge_weight * attn_factor * current_signals[neighbor]
                        total_weight += edge_weight * attn_factor
                
                if total_weight > 0:
                    neighbor_signal /= total_weight
                
                own_signal = current_signals.get(ticker, 0.0)
                new_signals[ticker] = own_signal + base_decay * neighbor_signal
            
            current_signals = new_signals
        
        return current_signals
    
    def get_attention_weights(self) -> Dict[str, TemporalAttentionWeights]:
        """Get all computed attention weights.
        
        Returns:
            Dict of ticker → TemporalAttentionWeights.
        """
        return self._attention_weights.copy()
    
    def export_attention_weights(self) -> Dict[str, Dict[str, float]]:
        """Export attention weights in serializable format.
        
        Returns:
            Dict of ticker → attention weight dict.
        """
        return {
            ticker: attn.to_dict()
            for ticker, attn in self._attention_weights.items()
        }

