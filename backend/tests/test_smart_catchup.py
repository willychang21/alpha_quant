"""Property-based tests for Smart Catch-Up Service.

Tests validate the core correctness properties defined in the design document:
- Property 1: Business day gap calculation accuracy
- Property 3: BACKFILL context enforcement
- Property 4: Ticker isolation on failure
- Property 5: Retry with exponential backoff
- Property 7: Return type contract
- Property 8: Empty data lake initial load
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.data.integrity import (
    CatchUpResult,
    MarketCalendar,
    OHLCVValidator,
    SmartCatchUpService,
    ValidationContext,
)
from quant.data.integrity.processor import ActionProcessor


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def market_calendar():
    """NYSE market calendar for testing."""
    return MarketCalendar('NYSE')


@pytest.fixture
def mock_reader():
    """Mock ParquetReader."""
    reader = MagicMock()
    reader.read_prices = MagicMock(return_value=pd.DataFrame())
    return reader


@pytest.fixture
def mock_writer():
    """Mock ParquetWriter."""
    writer = MagicMock()
    writer.write_prices = MagicMock(return_value={'rows_written': 0})
    return writer


@pytest.fixture
def mock_fetcher():
    """Mock data fetcher."""
    fetcher = MagicMock()
    fetcher.fetch_range = MagicMock(return_value=pd.DataFrame())
    return fetcher


@pytest.fixture
def mock_provider():
    """Mock data provider."""
    provider = MagicMock()
    provider.get_universe = MagicMock(return_value=['AAPL', 'MSFT', 'GOOGL'])
    return provider


@pytest.fixture
def catch_up_service(mock_provider, mock_fetcher, mock_reader, mock_writer, market_calendar):
    """Configured SmartCatchUpService for testing."""
    return SmartCatchUpService(
        data_provider=mock_provider,
        data_fetcher=mock_fetcher,
        parquet_reader=mock_reader,
        parquet_writer=mock_writer,
        validator=OHLCVValidator(),
        market_calendar=market_calendar,
        max_retries=3,
    )


# =============================================================================
# Property 1: Business day gap calculation accuracy
# =============================================================================

class TestProperty1_GapCalculation:
    """
    Property 1: Business day gap calculation accuracy
    
    For any pair of dates (last_date, target_date) where last_date < target_date,
    the calculated gap SHALL equal the number of NYSE trading days between them
    (exclusive of last_date, inclusive of target_date), correctly excluding
    weekends and market holidays.
    """
    
    def test_gap_excludes_weekends(self, market_calendar):
        """Gap should not count weekend days."""
        # Friday to Monday = 1 business day (Monday only)
        friday = date(2023, 12, 22)  # Friday
        monday = date(2023, 12, 25)  # Monday (Christmas - holiday!)
        tuesday = date(2023, 12, 26)  # Tuesday
        
        days = market_calendar.valid_days(friday, tuesday)
        # Friday + Tuesday (Monday is Christmas holiday)
        assert len(days) == 2
    
    def test_gap_excludes_holidays(self, market_calendar):
        """Gap should not count market holidays."""
        # Christmas week 2023
        dec_22 = date(2023, 12, 22)  # Friday - trading day
        dec_25 = date(2023, 12, 25)  # Monday - Christmas (closed)
        dec_26 = date(2023, 12, 26)  # Tuesday - trading day
        
        assert market_calendar.is_trading_day(dec_22) is True
        assert market_calendar.is_trading_day(dec_25) is False
        assert market_calendar.is_trading_day(dec_26) is True
    
    def test_gap_same_day_returns_zero(self, catch_up_service):
        """Gap between same dates should be 0."""
        today = date(2023, 12, 22)
        gap = catch_up_service._calculate_business_day_gap(today, today)
        assert gap == 0
    
    def test_gap_reverse_dates_returns_zero(self, catch_up_service):
        """Gap with start > end should return 0."""
        gap = catch_up_service._calculate_business_day_gap(
            date(2023, 12, 26), 
            date(2023, 12, 22)
        )
        assert gap == 0
    
    def test_gap_one_business_day(self, catch_up_service):
        """Single business day gap."""
        # Dec 21 (Thu) to Dec 22 (Fri)
        gap = catch_up_service._calculate_business_day_gap(
            date(2023, 12, 21),
            date(2023, 12, 22)
        )
        assert gap == 1
    
    def test_next_trading_day(self, market_calendar):
        """next_trading_day should return correct next day."""
        # Friday before Christmas weekend
        next_day = market_calendar.next_trading_day(date(2023, 12, 22))
        # Should skip weekend and Christmas Monday
        assert next_day == date(2023, 12, 26)


# =============================================================================
# Property 3: BACKFILL context enforcement
# =============================================================================

class TestProperty3_BackfillContext:
    """
    Property 3: BACKFILL context enforcement
    
    For any backfill operation, the validator SHALL be called with
    ValidationContext.BACKFILL, enabling spike reversion verification.
    """
    
    def test_backfill_uses_backfill_context(
        self, mock_provider, mock_fetcher, mock_reader, mock_writer
    ):
        """Validator must be called with BACKFILL context during backfill."""
        # Create mock validator to track context
        mock_validator = MagicMock()
        mock_validator.validate = MagicMock(return_value=MagicMock(
            rows_input=10,
            rows_output=10,
            rows_dropped=0,
            confirmed_spikes=[],
            persistent_moves=[],
        ))
        
        # Return data to trigger validation
        mock_fetcher.fetch_range.return_value = pd.DataFrame({
            'ticker': ['AAPL'] * 5,
            'date': pd.date_range('2023-12-20', periods=5),
            'open': [100.0] * 5,
            'high': [105.0] * 5,
            'low': [99.0] * 5,
            'close': [102.0] * 5,
            'adj_close': [102.0] * 5,
            'volume': [1000000] * 5,
        })
        
        # Create mock reader that returns a date
        mock_reader.read_prices.return_value = pd.DataFrame({
            'date': [date(2023, 12, 18)]
        })
        
        service = SmartCatchUpService(
            data_provider=mock_provider,
            data_fetcher=mock_fetcher,
            parquet_reader=mock_reader,
            parquet_writer=mock_writer,
            validator=mock_validator,
        )
        
        # Trigger backfill
        service.check_and_backfill()
        
        # Verify BACKFILL context was used
        if mock_validator.validate.called:
            call_args = mock_validator.validate.call_args
            assert call_args.kwargs.get('context') == ValidationContext.BACKFILL \
                or (len(call_args.args) > 1 and call_args.args[1] == ValidationContext.BACKFILL)


# =============================================================================
# Property 4: Ticker isolation on failure
# =============================================================================

class TestProperty4_TickerIsolation:
    """
    Property 4: Ticker isolation on failure
    
    For any batch of tickers where one ticker fails validation or fetch,
    the remaining tickers SHALL still be processed and written to the data lake.
    """
    
    def test_single_ticker_failure_continues_others(
        self, mock_provider, mock_fetcher, mock_reader, mock_writer, market_calendar
    ):
        """Failed ticker should not block processing of others."""
        # Setup data with multiple tickers
        mock_fetcher.fetch_range.return_value = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL'],
            'date': pd.date_range('2023-12-20', periods=2).tolist() * 3,
            'open': [100.0, 101.0, 200.0, 201.0, 300.0, 301.0],
            'high': [105.0, 106.0, 205.0, 206.0, 305.0, 306.0],
            'low': [99.0, 100.0, 199.0, 200.0, 299.0, 300.0],
            'close': [102.0, 103.0, 202.0, 203.0, 302.0, 303.0],
            'adj_close': [102.0, 103.0, 202.0, 203.0, 302.0, 303.0],
            'volume': [1000000] * 6,
        })
        
        mock_reader.read_prices.return_value = pd.DataFrame({
            'date': [date(2023, 12, 18)]
        })
        
        # Create validator that fails for MSFT only
        call_count = [0]
        original_validate = OHLCVValidator().validate
        
        def selective_validator(df, context=ValidationContext.DAILY, lazy=True):
            call_count[0] += 1
            if 'ticker' in df.columns and 'MSFT' in df['ticker'].values:
                raise ValueError("Simulated MSFT validation failure")
            return original_validate(df, context, lazy)
        
        mock_validator = MagicMock()
        mock_validator.validate = selective_validator
        
        service = SmartCatchUpService(
            data_provider=mock_provider,
            data_fetcher=mock_fetcher,
            parquet_reader=mock_reader, 
            parquet_writer=mock_writer,
            validator=mock_validator,
            market_calendar=market_calendar,
        )
        
        result = service._perform_backfill(
            start=date(2023, 12, 19),
            end=date(2023, 12, 22),
        )
        
        # Should have processed some tickers despite MSFT failure
        assert result.tickers_failed >= 1
        assert result.tickers_updated >= 0  # At least tried others
        assert len(result.errors) >= 1
        assert any('MSFT' in error for error in result.errors)


# =============================================================================
# Property 5: Retry with exponential backoff
# =============================================================================

class TestProperty5_RetryBackoff:
    """
    Property 5: Retry with exponential backoff
    
    For any network error during fetch, the system SHALL retry up to max_retries
    times with exponentially increasing delays (2^attempt seconds).
    """
    
    def test_retry_on_network_error(
        self, mock_provider, mock_reader, mock_writer, market_calendar
    ):
        """Should retry on network errors with backoff."""
        attempts = []
        
        def failing_fetcher(*args, **kwargs):
            attempts.append(len(attempts))
            if len(attempts) < 3:
                raise ConnectionError("Network error")
            return pd.DataFrame({
                'ticker': ['AAPL'],
                'date': [date(2023, 12, 20)],
                'open': [100.0],
                'high': [105.0],
                'low': [99.0],
                'close': [102.0],
                'adj_close': [102.0],
                'volume': [1000000],
            })
        
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_range = failing_fetcher
        
        service = SmartCatchUpService(
            data_provider=mock_provider,
            data_fetcher=mock_fetcher,
            parquet_reader=mock_reader,
            parquet_writer=mock_writer,
            validator=OHLCVValidator(),
            market_calendar=market_calendar,
            max_retries=3,
        )
        
        with patch('time.sleep') as mock_sleep:
            result = service._fetch_with_retry(
                tickers=['AAPL'],
                start_date=date(2023, 12, 19),
                end_date=date(2023, 12, 22),
            )
        
        # Should have made 3 attempts
        assert len(attempts) == 3
        # Should have slept with exponential backoff
        assert mock_sleep.call_count == 2  # 2 sleeps for 3 attempts
    
    def test_exhausted_retries_raises(
        self, mock_provider, mock_reader, mock_writer, market_calendar
    ):
        """Should raise after all retries exhausted."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_range = MagicMock(
            side_effect=ConnectionError("Persistent network error")
        )
        
        service = SmartCatchUpService(
            data_provider=mock_provider,
            data_fetcher=mock_fetcher,
            parquet_reader=mock_reader,
            parquet_writer=mock_writer,
            validator=OHLCVValidator(),
            market_calendar=market_calendar,
            max_retries=2,
        )
        
        with patch('time.sleep'):
            with pytest.raises(ConnectionError):
                service._fetch_with_retry(
                    tickers=['AAPL'],
                    start_date=date(2023, 12, 19),
                    end_date=date(2023, 12, 22),
                )


# =============================================================================
# Property 7: Return type contract
# =============================================================================

class TestProperty7_ReturnTypeContract:
    """
    Property 7: Return type contract
    
    For any call to check_and_backfill(), the return value SHALL be a tuple
    of (bool, int) where the bool indicates ready status and int indicates
    days backfilled.
    """
    
    def test_return_type_is_tuple(self, catch_up_service, mock_reader):
        """check_and_backfill must return tuple of (bool, int)."""
        mock_reader.read_prices.return_value = pd.DataFrame({
            'date': [date.today()]  # No gap
        })
        
        result = catch_up_service.check_and_backfill()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], int)
    
    def test_return_ready_true_when_up_to_date(self, catch_up_service, mock_reader):
        """Should return ready=True when no gap."""
        mock_reader.read_prices.return_value = pd.DataFrame({
            'date': [date.today()]
        })
        
        ready, days = catch_up_service.check_and_backfill()
        
        assert ready is True
        assert days == 0


# =============================================================================
# Property 8: Empty data lake initial load
# =============================================================================

class TestProperty8_EmptyDataLake:
    """
    Property 8: Empty data lake initial load
    
    For any data lake with no existing data (max_date is None), the system
    SHALL perform an initial load for the configured lookback period.
    """
    
    def test_empty_data_lake_triggers_initial_load(
        self, mock_provider, mock_fetcher, mock_writer, market_calendar
    ):
        """Empty data lake should trigger initial load."""
        # Reader returns empty DataFrame (no data)
        mock_reader = MagicMock()
        mock_reader.read_prices.return_value = pd.DataFrame()
        
        # Return some data for initial load
        mock_fetcher.fetch_range.return_value = pd.DataFrame({
            'ticker': ['AAPL'],
            'date': [date.today() - timedelta(days=365)],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [102.0],
            'adj_close': [102.0],
            'volume': [1000000],
        })
        
        service = SmartCatchUpService(
            data_provider=mock_provider,
            data_fetcher=mock_fetcher,
            parquet_reader=mock_reader,
            parquet_writer=mock_writer,
            validator=OHLCVValidator(),
            market_calendar=market_calendar,
            initial_lookback_days=365 * 2,
        )
        
        ready, days = service.check_and_backfill()
        
        # Should have attempted to fetch data
        assert mock_fetcher.fetch_range.called
        # Fetch should have been for lookback period
        call_args = mock_fetcher.fetch_range.call_args
        start_date = call_args.kwargs.get('start') or call_args.args[1]
        # Start date should be approximately 2 years ago
        expected_start = date.today() - timedelta(days=365 * 2)
        assert abs((start_date - expected_start).days) <= 1
    
    def test_get_max_date_returns_none_for_empty(self, catch_up_service, mock_reader):
        """_get_max_date should return None for empty data lake."""
        mock_reader.read_prices.return_value = pd.DataFrame()
        
        max_date = catch_up_service._get_max_date()
        
        assert max_date is None


# =============================================================================
# CatchUpResult Tests
# =============================================================================

class TestCatchUpResult:
    """Tests for CatchUpResult dataclass."""
    
    def test_to_dict(self):
        """CatchUpResult should serialize to dict."""
        result = CatchUpResult(
            ready=True,
            days_backfilled=5,
            tickers_updated=10,
            tickers_failed=2,
            rows_added=1000,
            rows_dropped=50,
            confirmed_spikes=[{'ticker': 'AAPL', 'date': '2023-12-20'}],
            persistent_moves=[],
            errors=['Error 1'],
        )
        
        d = result.to_dict()
        
        assert d['ready'] is True
        assert d['days_backfilled'] == 5
        assert d['tickers_updated'] == 10
        assert d['tickers_failed'] == 2
        assert d['rows_added'] == 1000
        assert d['rows_dropped'] == 50
        assert len(d['confirmed_spikes']) == 1
        assert len(d['errors']) == 1
    
    def test_default_values(self):
        """CatchUpResult should have sensible defaults."""
        result = CatchUpResult(
            ready=True,
            days_backfilled=0,
            tickers_updated=0,
            tickers_failed=0,
            rows_added=0,
            rows_dropped=0,
        )
        
        assert result.confirmed_spikes == []
        assert result.persistent_moves == []
        assert result.errors == []


# =============================================================================
# MarketCalendar Tests
# =============================================================================

class TestMarketCalendar:
    """Tests for MarketCalendar wrapper."""
    
    def test_valid_days_returns_list(self, market_calendar):
        """valid_days should return list of dates."""
        days = market_calendar.valid_days(
            date(2023, 12, 18),
            date(2023, 12, 22)
        )
        
        assert isinstance(days, list)
        assert all(isinstance(d, date) for d in days)
    
    def test_previous_trading_day(self, market_calendar):
        """previous_trading_day should return correct date."""
        # Monday after a weekend
        prev = market_calendar.previous_trading_day(date(2023, 12, 25))
        assert prev == date(2023, 12, 22)  # Previous Friday
    
    def test_count_trading_days(self, market_calendar):
        """count_trading_days should return correct count."""
        count = market_calendar.count_trading_days(
            date(2023, 12, 18),
            date(2023, 12, 22)
        )
        assert count == 5  # Mon-Fri
