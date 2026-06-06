import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

# ==========================================
# 1. 配置区：请在此处填入你的 Alpaca 密钥
# ==========================================
API_KEY = 'YOUR_API_KEY_HERE'
SECRET_KEY = 'YOUR_SECRET_KEY_HERE'
SYMBOL = 'TQQQ'  # 你也可以换成 'SOXL'


class QuantitativeBacktester:
    def __init__(self, api_key, secret_key):
        # 初始化客户端
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def fetch_data(self, symbol, days=365):
        """抓取历史日线数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        try:
            bars = self.client.get_stock_bars(request_params)
            if not bars.df.empty:
                df = bars.df.reset_index(level=0, drop=True)
                return df
            else:
                print("未获取到数据，请检查 Symbol 是否正确。")
                return None
        except Exception as e:
            print(f"获取数据失败，错误原因: {e}")
            return None

    def generate_signals(self, df):
        """
        量价策略逻辑：
        - 价格因子：20日均线(短期趋势) 站上 50日均线(中期趋势)
        - 成交量因子：当日成交量 > 5日平均成交量的 1.2倍 (放量确认)
        """
        if df is None: return None

        # 计算技术指标
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['Vol_MA5'] = df['volume'].rolling(window=5).mean()

        # 初始化信号：0 为空仓/观望，1 为持有
        df['signal'] = 0.0

        # 核心逻辑判断
        # 条件：金叉(MA20 > MA50) 且 当前价格在均线上方 且 成交量放量
        condition = (df['MA20'] > df['MA50']) & \
                    (df['close'] > df['MA20']) & \
                    (df['volume'] > df['Vol_MA5'] * 1.2)

        df.loc[condition, 'signal'] = 1.0

        # 计算收益率
        df['market_returns'] = df['close'].pct_change()
        # 信号向后移一天执行 (避免未来函数)
        df['strategy_returns'] = df['market_returns'] * df['signal'].shift(1)

        # 计算累计收益
        df['cum_market'] = (1 + df['market_returns'].fillna(0)).cumprod() - 1
        df['cum_strategy'] = (1 + df['strategy_returns'].fillna(0)).cumprod() - 1

        return df


# ==========================================
# 2. 运行区
# ==========================================
if __name__ == "__main__":
    backtester = QuantitativeBacktester(API_KEY, SECRET_KEY)

    print(f"--- 正在获取 {SYMBOL} 历史量价数据 ---")
    data = backtester.fetch_data(SYMBOL)

    if data is not None:
        print("--- 正在生成交易信号 ---")
        result = backtester.generate_signals(data)

        # 打印最后 10 行结果
        print("\n[ 最近 10 个交易日数据预览 ]")
        cols = ['close', 'volume', 'MA20', 'signal', 'cum_strategy']
        print(result[cols].tail(10))

        final_return = result['cum_strategy'].iloc[-1]
        print(f"\n✅ 策略回测完成！")
        print(f"📈 过去一年的累计收益率: {final_return:.2%}")

        if final_return <= 0:
            print("💡 提示：当前的简单均线逻辑在震荡市可能表现一般，可以考虑加入 RSI 因子避险。")
    else:
        print("❌ 程序终止：无法通过 API 验证，请检查你的 API Key。")