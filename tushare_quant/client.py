from __future__ import annotations

import os

import tushare as ts


DEFAULT_HTTP_URL = "http://8.136.22.187:8011/"


def get_pro_api(token: str | None = None, http_url: str | None = None):
    """
    Build a Tushare Pro client.

    Configuration priority:
      1. Function arguments
      2. Environment variables: TUSHARE_TOKEN, TUSHARE_HTTP_URL
      3. DEFAULT_HTTP_URL for the HTTP endpoint only

    Keep tokens out of source files.
    """
    token = token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Missing TUSHARE_TOKEN environment variable.")

    http_url = http_url or os.environ.get("TUSHARE_HTTP_URL") or DEFAULT_HTTP_URL
    pro = ts.pro_api(token)
    pro._DataApi__http_url = http_url
    return pro
