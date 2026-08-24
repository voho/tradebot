"""Auto-discovered multi-asset strategies (mirrors ``tradebot.strategies``).

Every non-underscore module here is imported by
``tradebot.multi_strategy._discover`` on first use; classes decorated with
``@register_multi_asset`` become available to ``tradebot run``,
``scripts/inference.py`` and any other caller of
``available_multi_asset_strategies()``.
"""
