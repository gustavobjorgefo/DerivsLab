"""
derivslab.market_data.providers
--------------------------------
Concrete ``MarketDataProvider`` implementations, one module per data source.

Each provider adapts a specific data feed (ProfitPro RTD, MetaTrader 5,
etc.) to the common ``MarketDataProvider`` interface defined in
``derivslab.market_data.base``. Consumers depend only on that interface
and are therefore agnostic to which provider is in use.
"""
