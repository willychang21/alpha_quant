"""MarketCalendar wrapper for business day operations.

Provides a thin wrapper around pandas_market_calendars for accurate
business day calculation in the Smart Catch-Up Service.

Requirements: 1.2, 1.3
"""

from datetime import date, timedelta
from typing import List

import pandas_market_calendars as mcal


class MarketCalendar:
    """
    Wrapper around pandas_market_calendars for business day operations.
    
    Provides methods for calculating valid trading days, checking if a date
    is a trading day, and finding the next trading day.
    
    Requirements: 1.2, 1.3
    
    Example:
        >>> cal = MarketCalendar('NYSE')
        >>> cal.valid_days(date(2023, 12, 22), date(2023, 12, 26))
        [date(2023, 12, 22), date(2023, 12, 26)]  # 12/25 is Christmas
    """
    
    def __init__(self, exchange: str = 'NYSE'):
        """
        Initialize with exchange calendar.
        
        Args:
            exchange: Exchange name (default: NYSE). Other options include
                      'NASDAQ', 'CME', 'LSE', etc.
        """
        self.exchange = exchange
        self.calendar = mcal.get_calendar(exchange)
    
    def valid_days(
        self, 
        start_date: date, 
        end_date: date
    ) -> List[date]:
        """
        Get list of valid trading days in range.
        
        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)
            
        Returns:
            List of valid trading dates (as date objects)
            
        Example:
            >>> cal = MarketCalendar('NYSE')
            >>> days = cal.valid_days(date(2023, 12, 22), date(2023, 12, 26))
            >>> len(days)  # Excludes weekend and Christmas
            2
        """
        if start_date > end_date:
            return []
        
        schedule = self.calendar.schedule(
            start_date=start_date, 
            end_date=end_date
        )
        return [d.date() for d in schedule.index]
    
    def is_trading_day(self, target_date: date) -> bool:
        """
        Check if a date is a valid trading day.
        
        Args:
            target_date: Date to check
            
        Returns:
            True if the date is a valid trading day
            
        Example:
            >>> cal = MarketCalendar('NYSE')
            >>> cal.is_trading_day(date(2023, 12, 25))  # Christmas
            False
        """
        days = self.valid_days(target_date, target_date)
        return len(days) > 0
    
    def next_trading_day(self, target_date: date) -> date:
        """
        Get the next trading day after target_date.
        
        Args:
            target_date: Reference date
            
        Returns:
            The next valid trading day after target_date
            
        Example:
            >>> cal = MarketCalendar('NYSE')
            >>> cal.next_trading_day(date(2023, 12, 22))  # Friday before Christmas
            date(2023, 12, 26)  # Tuesday (Mon is Christmas)
        """
        # Look ahead up to 10 days to handle long weekends/holidays
        end = target_date + timedelta(days=10)
        days = self.valid_days(target_date + timedelta(days=1), end)
        return days[0] if days else target_date + timedelta(days=1)
    
    def previous_trading_day(self, target_date: date) -> date:
        """
        Get the previous trading day before target_date.
        
        Args:
            target_date: Reference date
            
        Returns:
            The previous valid trading day before target_date
        """
        # Look back up to 10 days to handle long weekends/holidays
        start = target_date - timedelta(days=10)
        days = self.valid_days(start, target_date - timedelta(days=1))
        return days[-1] if days else target_date - timedelta(days=1)
    
    def count_trading_days(
        self, 
        start_date: date, 
        end_date: date,
        inclusive: bool = True
    ) -> int:
        """
        Count the number of trading days between two dates.
        
        Args:
            start_date: Start of range
            end_date: End of range
            inclusive: If True, include both start and end dates
            
        Returns:
            Number of trading days in the range
        """
        days = self.valid_days(start_date, end_date)
        return len(days)
