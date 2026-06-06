"""
Lightweight data provider layer for this research project.

The package keeps provider-specific details behind small Python APIs and returns
pandas DataFrames or plain Python dictionaries. It intentionally does not copy
code from external terminal projects; it only follows the same practical ideas:
batch requests, local caching, and narrow wrappers around the upstream libraries.
"""

from . import akshare_cn, yahoo

__all__ = ["akshare_cn", "yahoo"]
