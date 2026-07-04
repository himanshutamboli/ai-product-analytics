"""Growth-analytics metrics over the `sessions` / `users` tables.

Each module maps to a dashboard section; functions take (filtered) polars frames and return
chartable frames or summary dicts.
"""

from ai_product_analytics.metrics.adoption import active_users_series, stickiness_series
from ai_product_analytics.metrics.economics import (
    cost_series,
    economics_summary,
    spend_by,
    unit_economics_series,
)
from ai_product_analytics.metrics.funnel import activation_funnel, feature_adoption
from ai_product_analytics.metrics.quality import quality_series, quality_summary
from ai_product_analytics.metrics.retention import cohort_retention

__all__ = [
    "active_users_series",
    "stickiness_series",
    "cohort_retention",
    "feature_adoption",
    "activation_funnel",
    "quality_series",
    "quality_summary",
    "cost_series",
    "unit_economics_series",
    "spend_by",
    "economics_summary",
]
