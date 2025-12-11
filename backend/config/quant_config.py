"""Quant Configuration Dataclasses.

Centralized configuration for valuation, optimization, and factor analysis.
Supports environment variable overrides via pydantic Settings integration.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)


def _log_default(name: str, value) -> None:
    """Log when a default value is being used."""
    logger.debug(f"[CONFIG] Using default for {name}: {value}")


@dataclass
class ValuationConfig:
    """Configuration for DCF and valuation calculations.
    
    All values can be overridden via environment variables:
    - VALUATION_WACC
    - VALUATION_GROWTH_RATE
    - VALUATION_TERMINAL_GROWTH
    - VALUATION_PROJECTION_YEARS
    - VALUATION_RISK_FREE_RATE
    - VALUATION_MARKET_RISK_PREMIUM
    """
    wacc: float = field(default=0.09)
    growth_rate: float = field(default=0.05)
    terminal_growth: float = field(default=0.02)
    projection_years: int = field(default=5)
    risk_free_rate: float = field(default=0.04)
    market_risk_premium: float = field(default=0.05)
    
    def __post_init__(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'wacc': ('VALUATION_WACC', float),
            'growth_rate': ('VALUATION_GROWTH_RATE', float),
            'terminal_growth': ('VALUATION_TERMINAL_GROWTH', float),
            'projection_years': ('VALUATION_PROJECTION_YEARS', int),
            'risk_free_rate': ('VALUATION_RISK_FREE_RATE', float),
            'market_risk_premium': ('VALUATION_MARKET_RISK_PREMIUM', float),
        }
        
        for attr, (env_var, type_fn) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    setattr(self, attr, type_fn(env_value))
                    logger.info(f"[CONFIG] Override {attr} from env: {getattr(self, attr)}")
                except ValueError:
                    logger.warning(f"[CONFIG] Invalid env value for {env_var}: {env_value}")
            else:
                _log_default(attr, getattr(self, attr))


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization.
    
    All values can be overridden via environment variables:
    - OPTIMIZATION_MAX_WEIGHT
    - OPTIMIZATION_RISK_AVERSION
    - OPTIMIZATION_TOP_N
    - OPTIMIZATION_TARGET_VOL
    - OPTIMIZATION_SECTOR_MAX_WEIGHT
    """
    max_weight: float = field(default=0.10)
    risk_aversion: float = field(default=1.0)
    top_n: int = field(default=50)
    target_vol: Optional[float] = field(default=None)
    sector_max_weight: float = field(default=0.30)
    
    def __post_init__(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'max_weight': ('OPTIMIZATION_MAX_WEIGHT', float),
            'risk_aversion': ('OPTIMIZATION_RISK_AVERSION', float),
            'top_n': ('OPTIMIZATION_TOP_N', int),
            'sector_max_weight': ('OPTIMIZATION_SECTOR_MAX_WEIGHT', float),
        }
        
        for attr, (env_var, type_fn) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    setattr(self, attr, type_fn(env_value))
                    logger.info(f"[CONFIG] Override {attr} from env: {getattr(self, attr)}")
                except ValueError:
                    logger.warning(f"[CONFIG] Invalid env value for {env_var}: {env_value}")
            else:
                _log_default(attr, getattr(self, attr))
        
        # Special handling for optional target_vol
        target_vol_env = os.environ.get('OPTIMIZATION_TARGET_VOL')
        if target_vol_env is not None:
            try:
                self.target_vol = float(target_vol_env)
                logger.info(f"[CONFIG] Override target_vol from env: {self.target_vol}")
            except ValueError:
                logger.warning(f"[CONFIG] Invalid env value for OPTIMIZATION_TARGET_VOL: {target_vol_env}")


@dataclass
class FactorConfig:
    """Configuration for factor weights by market regime.
    
    Bull weights are used during bullish market conditions (e.g., VIX < 20).
    Bear weights favor defensive factors during volatile markets.
    
    Factor key meanings:
    - vsm: Value-Size-Momentum composite
    - bab: Betting Against Beta (low-vol factor)
    - qmj: Quality Minus Junk
    - upside: Upside capture / momentum quality
    - pead: Post-Earnings Announcement Drift
    - sentiment: Analyst sentiment / revisions
    
    Environment variables (JSON string):
    - FACTOR_WEIGHTS_BULL
    - FACTOR_WEIGHTS_BEAR
    """
    bull_weights: Dict[str, float] = field(default_factory=lambda: {
        "vsm": 0.25,
        "bab": 0.08,
        "qmj": 0.17,
        "upside": 0.17,
        "pead": 0.13,
        "sentiment": 0.15,
    })
    bear_weights: Dict[str, float] = field(default_factory=lambda: {
        "vsm": 0.08,
        "bab": 0.22,
        "qmj": 0.30,
        "upside": 0.13,
        "pead": 0.10,
        "sentiment": 0.12,
    })
    
    def __post_init__(self):
        """Apply environment variable overrides."""
        # Bull weights override
        bull_env = os.environ.get('FACTOR_WEIGHTS_BULL')
        if bull_env:
            try:
                self.bull_weights = json.loads(bull_env)
                logger.info(f"[CONFIG] Override bull_weights from env")
            except json.JSONDecodeError:
                logger.warning(f"[CONFIG] Invalid JSON for FACTOR_WEIGHTS_BULL: {bull_env}")
        else:
            _log_default("bull_weights", self.bull_weights)
            
        # Bear weights override
        bear_env = os.environ.get('FACTOR_WEIGHTS_BEAR')
        if bear_env:
            try:
                self.bear_weights = json.loads(bear_env)
                logger.info(f"[CONFIG] Override bear_weights from env")
            except json.JSONDecodeError:
                logger.warning(f"[CONFIG] Invalid JSON for FACTOR_WEIGHTS_BEAR: {bear_env}")
        else:
            _log_default("bear_weights", self.bear_weights)
    
    def get_weights(self, is_bull_market: bool = True) -> Dict[str, float]:
        """Get appropriate factor weights for current market regime.
        
        Args:
            is_bull_market: True for bullish conditions, False for bearish
            
        Returns:
            Dictionary of factor weights summing to ~1.0
        """
        return self.bull_weights if is_bull_market else self.bear_weights


@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting.
    
    Uses token bucket algorithm for smooth rate limiting.
    
    Environment variables:
    - RATE_LIMIT_REQUESTS_PER_SECOND
    - RATE_LIMIT_BURST_SIZE
    """
    requests_per_second: float = field(default=2.0)
    burst_size: int = field(default=5)
    
    def __post_init__(self):
        """Apply environment variable overrides."""
        rps_env = os.environ.get('RATE_LIMIT_REQUESTS_PER_SECOND')
        if rps_env:
            try:
                self.requests_per_second = float(rps_env)
                logger.info(f"[CONFIG] Override requests_per_second from env: {self.requests_per_second}")
            except ValueError:
                logger.warning(f"[CONFIG] Invalid env value for RATE_LIMIT_REQUESTS_PER_SECOND: {rps_env}")
        else:
            _log_default("requests_per_second", self.requests_per_second)
        
        burst_env = os.environ.get('RATE_LIMIT_BURST_SIZE')
        if burst_env:
            try:
                self.burst_size = int(burst_env)
                logger.info(f"[CONFIG] Override burst_size from env: {self.burst_size}")
            except ValueError:
                logger.warning(f"[CONFIG] Invalid env value for RATE_LIMIT_BURST_SIZE: {burst_env}")
        else:
            _log_default("burst_size", self.burst_size)


@dataclass
class LoggingConfig:
    """Configuration for structured logging.
    
    Environment variables:
    - LOG_DEV_MODE: Enable emoji and verbose logging in development
    """
    dev_mode: bool = field(default=False)
    
    def __post_init__(self):
        """Apply environment variable overrides."""
        dev_mode_env = os.environ.get('LOG_DEV_MODE', '').lower()
        if dev_mode_env in ('true', '1', 'yes'):
            self.dev_mode = True
            logger.info("[CONFIG] Development mode enabled for logging")
        elif dev_mode_env in ('false', '0', 'no'):
            self.dev_mode = False
        else:
            _log_default("dev_mode", self.dev_mode)


# Singleton instances with lazy initialization
_valuation_config: Optional[ValuationConfig] = None
_optimization_config: Optional[OptimizationConfig] = None
_factor_config: Optional[FactorConfig] = None
_rate_limit_config: Optional[RateLimitConfig] = None
_logging_config: Optional[LoggingConfig] = None


def get_valuation_config() -> ValuationConfig:
    """Get or create ValuationConfig singleton."""
    global _valuation_config
    if _valuation_config is None:
        _valuation_config = ValuationConfig()
    return _valuation_config


def get_optimization_config() -> OptimizationConfig:
    """Get or create OptimizationConfig singleton."""
    global _optimization_config
    if _optimization_config is None:
        _optimization_config = OptimizationConfig()
    return _optimization_config


def get_factor_config() -> FactorConfig:
    """Get or create FactorConfig singleton."""
    global _factor_config
    if _factor_config is None:
        _factor_config = FactorConfig()
    return _factor_config


def get_rate_limit_config() -> RateLimitConfig:
    """Get or create RateLimitConfig singleton."""
    global _rate_limit_config
    if _rate_limit_config is None:
        _rate_limit_config = RateLimitConfig()
    return _rate_limit_config


def get_logging_config() -> LoggingConfig:
    """Get or create LoggingConfig singleton."""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    return _logging_config


def reset_configs() -> None:
    """Reset all config singletons. Useful for testing."""
    global _valuation_config, _optimization_config, _factor_config
    global _rate_limit_config, _logging_config
    _valuation_config = None
    _optimization_config = None
    _factor_config = None
    _rate_limit_config = None
    _logging_config = None
