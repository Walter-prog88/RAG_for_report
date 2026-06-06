import yfinance as yf
import pandas as pd


def get_option_analytics(ticker_symbol):
    """
    获取指定股票的最接近到期日的平值期权链数据
    """
    ticker = yf.Ticker(ticker_symbol)

    # 1. 获取标的价格
    current_price = ticker.fast_info['lastPrice']

    # 2. 获取所有到期日，选择最近的一个
    expirations = ticker.options
    if not expirations:
        return None, "No options found"

    target_date = expirations[0]  # 获取最近的到期日
    opt_chain = ticker.option_chain(target_date)

    # 3. 处理 Call 数据，寻找平值(ATM)期权
    calls = opt_chain.calls
    # 找到行权价最接近当前价格的那行
    atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]

    # 提取关键希腊字母和数据 (yfinance不提供所有希腊字母，主要提取IV和成交量)
    analytics = {
        "ticker": ticker_symbol,
        "underlying_price": round(current_price, 2),
        "expiration": target_date,
        "atm_strike": float(atm_call['strike'].values[0]),
        "atm_iv": round(float(atm_call['impliedVolatility'].values[0]), 4),
        "atm_last_price": float(atm_call['lastPrice'].values[0]),
        "volume": int(atm_call['volume'].values[0])
    }

    return analytics

# --- 示例用法 ---
# data = get_option_analytics("NVDA")
# print(f"Nvidia ATM IV: {data['atm_iv']*100}%")