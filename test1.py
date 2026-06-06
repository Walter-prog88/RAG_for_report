import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. 基础配置 (Configuration)
# -----------------------------------------------------------------------------
# 设置中文字体 (防止绘图乱码，Mac常用 Arial Unicode MS 或 Heiti TC)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 配置代理 (你的端口)
PROXY_URL = "http://127.0.0.1:33210"
yf.set_config(proxy=PROXY_URL)


# -----------------------------------------------------------------------------
# 2. 核心类：期权流分析器 (Option Flow Analyzer)
# -----------------------------------------------------------------------------
class OptionResearcher:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.spot_price = None

    def get_realtime_price(self):
        """获取当前正股价格"""
        try:
            # 尝试获取快照数据
            hist = self.ticker.history(period='1d')
            if not hist.empty:
                self.spot_price = hist['Close'].iloc[-1]
                return self.spot_price
        except Exception as e:
            print(f"获取股价失败: {e}")
        return 0

    def fetch_aggregated_flow(self, max_expirations=4):
        """/
        抓取近期期权流数据
        :param max_expirations: 为了速度，默认只抓取最近的 4 个到期日
        """
        print(f"[{self.symbol}] 正在扫描期权链 (前 {max_expirations} 个到期日)...")

        try:
            exp_dates = self.ticker.options
        except Exception as e:
            print(f"网络连接错误，请检查代理: {e}")
            return None

        if not exp_dates:
            print("未找到期权日期。")
            return None

        # 容器：用于存储不同行权价的 Call/Put 总量
        flow_data = []

        # 遍历最近的几个到期日
        for date in exp_dates[:max_expirations]:
            print(f"  ->正在下载 {date} 数据...")
            try:
                opt = self.ticker.option_chain(date)
                calls = opt.calls
                puts = opt.puts

                # 简单清洗
                calls['type'] = 'call'
                puts['type'] = 'put'
                calls['expiration'] = date
                puts['expiration'] = date

                # 合并
                chain = pd.concat([calls, puts], sort=False)
                flow_data.append(chain)
            except Exception as e:
                print(f"  x 跳过 {date}: {e}")
                continue

        if not flow_data:
            return None

        full_df = pd.concat(flow_data)
        return full_df

    def calculate_sentiment_factor(self, df):
        """
        计算核心因子：Sentiment Score (情绪得分)
        """
        # 按类型汇总成交量
        vol_summary = df.groupby('type')['volume'].sum()
        call_vol = vol_summary.get('call', 0)
        put_vol = vol_summary.get('put', 0)
        total_vol = call_vol + put_vol

        # -------------------------------------------------------
        # 🎓 博士论文因子公式：Net Bullish Sentiment (净看涨情绪)
        # 范围 [-1, 1]。接近 1 代表极度看涨，接近 -1 代表极度看跌
        # -------------------------------------------------------
        if total_vol > 0:
            sentiment_score = (call_vol - put_vol) / total_vol
        else:
            sentiment_score = 0

        # 计算 Put/Call Ratio (传统指标)
        pc_ratio = put_vol / call_vol if call_vol > 0 else 0

        return {
            "Symbol": self.symbol,
            "Price": self.spot_price,
            "Total_Vol": total_vol,
            "Call_Vol": call_vol,
            "Put_Vol": put_vol,
            "Sentiment_Score": round(sentiment_score, 4),
            "PC_Ratio": round(pc_ratio, 2),
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

    def visualize_volume_distribution(self, df):
        """
        可视化：绘制行权价(Strike)上的成交量分布
        这是识别‘机构大单’位置的关键图表
        """
        if df is None or df.empty:
            return

        # 筛选在这个价格附近的期权 (例如当前价格 ±20%)
        if self.spot_price:
            lower_bound = self.spot_price * 0.8
            upper_bound = self.spot_price * 1.2
            df = df[(df['strike'] > lower_bound) & (df['strike'] < upper_bound)]

        # 按行权价和类型汇总成交量
        strike_vol = df.groupby(['strike', 'type'])['volume'].sum().unstack().fillna(0)

        # 绘图
        plt.figure(figsize=(12, 6))

        # 绘制 Call (向上) 和 Put (向下，取负值以便对比)
        plt.bar(strike_vol.index, strike_vol['call'], color='green', alpha=0.6, label='Call Vol')
        plt.bar(strike_vol.index, -strike_vol['put'], color='red', alpha=0.6, label='Put Vol')

        # 标记当前股价
        if self.spot_price:
            plt.axvline(x=self.spot_price, color='black', linestyle='--', linewidth=2,
                        label=f'当前股价: {self.spot_price:.2f}')

        plt.title(f'{self.symbol} 期权成交量分布 (Strike Distribution)', fontsize=14)
        plt.xlabel('行权价 (Strike Price)')
        plt.ylabel('成交量 (Volume)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 保存图片作为论文素材
        plt.savefig(f"{self.symbol}_option_structure.png")
        print(f"\n📊 图表已保存为 {self.symbol}_option_structure.png，请在左侧文件栏查看。")
        plt.show()


# -----------------------------------------------------------------------------
# 3. 主程序执行逻辑
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 目标股票：你可以改成 'NVDA', 'AAPL', 'MARA' 等
    target = "TSLA"

    researcher = OptionResearcher(target)

    # 1. 获取股价
    price = researcher.get_realtime_price()
    print(f"当前 {target} 价格: ${price:.2f}")

    # 2. 获取期权数据
    # 注意：这里我们只抓取最近 4 个到期日的数据作为演示
    df_flow = researcher.fetch_aggregated_flow(max_expirations=4)

    if df_flow is not None:
        # 3. 计算因子
        factor = researcher.calculate_sentiment_factor(df_flow)

        print("\n" + "=" * 40)
        print("🧪 策略因子输出 (Alpha Signal)")
        print("=" * 40)
        print(f"时间: {factor['Time']}")
        print(f"看涨成交量 (Calls): {factor['Call_Vol']:.0f}")
        print(f"看跌成交量 (Puts):  {factor['Put_Vol']:.0f}")
        print(f"Put/Call Ratio:    {factor['PC_Ratio']} (值越小越看涨)")
        print("-" * 40)
        print(f"情绪得分 (Sentiment): {factor['Sentiment_Score']}")
        print("(范围 -1 到 1，正数代表看涨，负数代表看跌)")
        print("=" * 40)

        # 4. 可视化分析
        researcher.visualize_volume_distribution(df_flow)

        # 5. (可选) 保存数据用于长期回测
        # 每天运行一次这个脚本，积累数据，就是你论文的数据集
        df_summary = pd.DataFrame([factor])
        # 追加写入 CSV (如果文件不存在会自动创建)
        filename = "thesis_data_collection.csv"
        df_summary.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)
        print(f"\n💾 数据已追加保存到 {filename}")
    else:
        print("未能获取数据。")