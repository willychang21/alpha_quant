import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Iterator, Tuple

class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation.
    
    This CV scheme is designed for financial time series where labels are derived 
    from overlapping time windows (e.g., 5-day future returns). It prevents leakage by:
    1. Purging: Removing training samples that overlap with the test set.
    2. Embargoing: Removing training samples immediately following the test set 
       to prevent leakage from long-duration serial correlation.
       
    Reference: Marcos Lopez de Prado, "Advances in Financial Machine Learning"
    """
    
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        """
        :param n_splits: Number of folds.
        :param embargo_pct: Percentage of total samples to embargo after each test set.
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, pred_times: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        :param X: Data features (DataFrame). Index must be datetime or sortable.
        :param y: Target variable (Series).
        :param pred_times: Series where index is the observation time (t) and 
                           value is the time when the label is fully realized (t + h).
                           Required for purging.
        :return: Iterator yielding (train_indices, test_indices)
        """
        if pred_times is None:
            raise ValueError("pred_times is required for Purged CV to determine overlaps.")
            
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        embargo = int(n_samples * self.embargo_pct)
        
        # Use KFold to get initial raw splits
        cv = KFold(n_splits=self.n_splits, shuffle=False)
        
        for train_idx, test_idx in cv.split(X):
            # 1. Identify Test Set Time Range
            test_start_time = pred_times.index[test_idx[0]]
            test_end_time = pred_times.iloc[test_idx[-1]] # Use the max prediction time of the test set
            
            # 2. Purge Training Set
            # Remove training samples where the label overlaps with the test set
            # Overlap happens if:
            # Train_Pred_Time > Test_Start_Time  AND  Train_Start_Time < Test_End_Time
            
            # However, for Walk-Forward (expanding window) or K-Fold, we usually just care about
            # not training on data that overlaps with the *future* test set if we are doing OOS.
            # In standard K-Fold, test sets are anywhere.
            
            # Let's implement the logic:
            # A training sample i overlaps with test set if:
            # pred_times[i] > test_start_time  AND  times[i] < test_end_time
            
            train_times = pred_times.index[train_idx] # t
            train_pred_times = pred_times.iloc[train_idx] # t + h
            
            # Find overlaps
            # We want to KEEP samples where:
            # (train_pred_times <= test_start_time)  OR  (train_times >= test_end_time + embargo)
            
            # Logic:
            # 1. Pre-Test Train: Train ends before Test starts. 
            #    Condition: train_pred_times <= test_start_time
            # 2. Post-Test Train: Train starts after Test ends (plus embargo).
            #    Condition: train_times >= test_end_time (plus embargo)
            
            # Note: test_end_time is the time when the LAST label in test set is realized.
            # So we must wait until that info is available before starting next train sample?
            # Actually, standard purging usually just removes the overlap.
            
            # Efficient masking
            # Convert to numeric for faster comparison if needed, but datetime works in pandas
            
            # Mask for Pre-Test
            mask_pre = train_pred_times <= test_start_time
            
            # Mask for Post-Test (with Embargo)
            # We need to find the index in the original series that corresponds to test_end_time + embargo
            # But since we are iterating indices, we can just check times.
            
            # Embargo logic: The training samples immediately following the test set are dropped.
            # We calculate the time when embargo ends.
            # Since we don't have a strict time frequency, we can just drop the next 'embargo' number of samples
            # from the sorted train_idx that appear after test_idx.
            
            # Simplified approach using indices for embargo if time is uniform, 
            # but better to use time if possible. 
            # Let's stick to the definition:
            # "The embargo eliminates training samples that follow a test set."
            
            valid_train_idx = []
            
            for i in train_idx:
                t_start = pred_times.index[i]
                t_end = pred_times.iloc[i]
                
                # Check overlap with Test Range [test_start_time, test_end_time]
                # Actually, the test range for *overlap* purposes is defined by the test samples' start and end times.
                
                # Let's simplify:
                # Overlap if: [t_start, t_end] intersects with [test_start_time, test_end_time]
                # Intersection: max(t_start, test_start_time) < min(t_end, test_end_time)
                
                is_overlap = max(t_start, test_start_time) < min(t_end, test_end_time)
                
                if not is_overlap:
                    # Check Embargo
                    # If this training sample is AFTER the test set, is it within embargo?
                    if t_start >= test_end_time:
                        # This is post-test. Check if it's too close.
                        # For simplicity, we'll assume embargo is handled by dropping indices 
                        # immediately following test_idx max.
                        pass 
                    
                    valid_train_idx.append(i)
            
            # Apply Embargo more efficiently
            # If test set is in the middle, we have Pre-Train and Post-Train.
            # Post-Train starts at test_idx[-1] + 1.
            # We should drop the first 'embargo' samples from Post-Train.
            
            # Re-construct valid_train_idx efficiently
            # 1. Pre-Test: Indices < test_idx.min()
            pre_test_train = train_idx[train_idx < test_idx.min()]
            # Filter Pre-Test for overlap (prediction time > test start)
            pre_test_keep = pre_test_train[pred_times.iloc[pre_test_train].values <= test_start_time]
            
            # 2. Post-Test: Indices > test_idx.max()
            post_test_train = train_idx[train_idx > test_idx.max()]
            
            # Purge Overlaps: Train start must be >= Test end
            # We use the time index to filter
            if len(post_test_train) > 0:
                post_test_times = pred_times.index[post_test_train]
                # Keep only those starting after test_end_time
                # Note: test_end_time is the time the LAST test label is realized.
                # So we can only train on data that starts AFTER that.
                
                # Apply Embargo: The embargo is a time period or number of samples AFTER the test set.
                # If embargo is a number of samples, we can just drop them from the valid set.
                # But strictly, embargo should be a time delta.
                # Here we implement embargo as "number of samples" to skip after the purge.
                
                # First, Purge:
                non_overlapping_mask = post_test_times >= test_end_time
                post_test_keep = post_test_train[non_overlapping_mask]
                
                # Second, Embargo:
                if len(post_test_keep) > embargo:
                    post_test_keep = post_test_keep[embargo:]
                else:
                    post_test_keep = np.array([], dtype=int)
            else:
                post_test_keep = np.array([], dtype=int)
                
            # Combine
            final_train_idx = np.concatenate([pre_test_keep, post_test_keep])
            
            yield final_train_idx, test_idx
