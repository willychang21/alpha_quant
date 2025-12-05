import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import logging
from typing import Tuple, Any

logger = logging.getLogger(__name__)

class MetaLabeler:
    def __init__(self, model_params: dict = None):
        self.params = model_params or {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'random_state': 42
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.explainer = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the XGBoost meta-labeler.
        
        Args:
            X_train (pd.DataFrame): Features at the time of the primary signal.
            y_train (pd.Series): Binary label (1 if primary signal succeeded, 0 otherwise).
        """
        logger.info(f"Training Meta-Labeler on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Initialize SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            
    def predict_proba(self, X_test: pd.DataFrame) -> pd.Series:
        """
        Predicts the probability of the primary signal being correct.
        
        Args:
            X_test (pd.DataFrame): Features.
            
        Returns:
            pd.Series: Probability of success (0.0 to 1.0).
        """
        if X_test.empty:
            return pd.Series()
            
        probs = self.model.predict_proba(X_test)[:, 1]
        return pd.Series(probs, index=X_test.index)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns feature importance based on Gain.
        """
        importance = self.model.feature_importances_
        features = self.model.feature_names_in_
        
        df = pd.DataFrame({'feature': features, 'importance': importance})
        return df.sort_values('importance', ascending=False)
        
    def explain(self, X: pd.DataFrame):
        """
        Returns SHAP values for the given data.
        """
        if self.explainer is None:
            return None
        return self.explainer.shap_values(X)

def prepare_meta_features(
    market_data: pd.DataFrame, 
    signals: pd.Series, 
    volatility: pd.Series
) -> pd.DataFrame:
    """
    Constructs the feature set for the Meta-Labeler.
    
    Features typically include:
    - Volatility (Regime)
    - Momentum (Recent returns)
    - Spread / Liquidity (if available)
    - The Primary Signal itself (Magnitude)
    
    Args:
        market_data (pd.DataFrame): OHLCV data.
        signals (pd.Series): The primary signal score (e.g. Ranking Score).
        volatility (pd.Series): Daily volatility.
        
    Returns:
        pd.DataFrame: Feature matrix X.
    """
    X = pd.DataFrame(index=signals.index)
    
    # 1. Volatility (Regime)
    X['volatility'] = volatility
    
    # 2. Signal Magnitude (Confidence of primary model)
    X['signal_score'] = signals
    
    # 3. Momentum (Trend)
    # Using market_data 'Close' aligned to signals
    # Assuming market_data has 'Close' and index is Date
    # We need to align carefully.
    
    # Common features:
    # - Serial Correlation
    # - RSI
    # - Distance from MA
    
    # For now, simple features
    return X.dropna()
