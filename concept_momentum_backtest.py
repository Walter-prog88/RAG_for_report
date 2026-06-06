import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from data_providers import akshare_cn

# 设置绘图支持中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统，Windows请改用 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def get_concept_data(concept_name="商业航天", start_date="20240101", limit=20):
    print(f"正在获取【{concept_name}】成分股列表...")
    try:
        stocks_df = akshare_cn.concept_constituents(concept_name)
    except Exception as e:
        print(f"获取概念成分股失败: {e}")
        return pd.DataFrame()
    if stocks_df.empty or '代码' not in stocks_df.columns:
        return pd.DataFrame()

    stock_codes = stocks_df['代码'].tolist()

    # 为了演示速度，这里选取前20只个股。实盘回测时可以去掉限制。
    stock_codes = stock_codes[:limit]

    print(f"正在抓取 {len(stock_codes)} 只成分股的历史行情...")
    histories = akshare_cn.batch_a_share_history(
        stock_codes,
        start_date=start_date,
        adjust="qfq",
        max_workers=4,
    )

    all_prices = {}
    for code in stock_codes:
        df = histories.get(code, pd.DataFrame())
        if df.empty or '日期' not in df.columns or '收盘' not in df.columns:
            print(f"获取 {code} 失败或无有效收盘价")
            continue
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        all_prices[code] = pd.to_numeric(df['收盘'], errors='coerce')

    return pd.DataFrame(all_prices).dropna(how='all')


def backtest_strategy(price_df, top_n=3):
    print("开始计算回测曲线...")
    # 1. 计算日收益率
    returns_df = price_df.pct_change()

    # 2. 计算月度收益率（用于选股信号）
    monthly_prices = price_df.resample('M').last()
    monthly_returns = monthly_prices.pct_change()

    # 3. 策略逻辑：每月月底选出下个月要持有的个股
    dates = monthly_returns.index

    # 初始化资产为1.0
    asset_curve = [1.0]

    for i in range(len(dates) - 1):
        current_month = dates[i]
        next_month = dates[i + 1]

        # 选股信号：在本月最后一天，选出本月表现最好的 N 只股
        signals = monthly_returns.loc[current_month].sort_values(ascending=False)
        selected_stocks = signals.head(top_n).index.tolist()

        # 计算这几只股在下个月的平均日收益率
        month_data = returns_df[selected_stocks][(returns_df.index > current_month) & (returns_df.index <= next_month)]
        daily_avg_return = month_data.mean(axis=1)

        # 记录每日资产变化
        for r in daily_avg_return:
            if not np.isnan(r):
                new_asset = asset_curve[-1] * (1 + r)
                asset_curve.append(new_asset)

    return asset_curve


# --- 执行流程 ---
if __name__ == "__main__":
    # 1. 获取数据
    concept = "商业航天"
    prices = get_concept_data(concept_name=concept, start_date="20240101")

    if prices.empty:
        print("未获取到数据，请检查网络或AkShare版本")
    else:
        # 2. 运行回测
        curve = backtest_strategy(prices, top_n=3)

        # 3. 计算基准（沪深300作为对比）
        hs300 = akshare_cn.china_index_history(symbol="000300", start_date="20240101")
        if not hs300.empty and '日期' in hs300.columns and '收盘' in hs300.columns:
            hs300['日期'] = pd.to_datetime(hs300['日期'])
            hs300.set_index('日期', inplace=True)
            hs300_subset = hs300[hs300.index >= prices.index[0]]
            hs300_curve = (1 + pd.to_numeric(hs300_subset['收盘'], errors='coerce').pct_change().fillna(0)).cumprod()
        else:
            hs300_curve = pd.Series(dtype=float)

        # 4. 可视化
        plt.figure(figsize=(12, 6))
        plt.plot(curve, label=f'策略曲线: {concept}动量轮动', color='red', linewidth=2)
        if not hs300_curve.empty:
            aligned_benchmark = hs300_curve / hs300_curve.iloc[0]
            plt.plot(
                np.linspace(0, len(curve) - 1, len(aligned_benchmark)),
                aligned_benchmark,
                label='沪深300基准',
                color='gray',
                alpha=0.7,
            )
        plt.title(f'量化策略回测：{concept}概念成分股')
        plt.xlabel('交易时间步长')
        plt.ylabel('累计净值')
        plt.legend()
        plt.grid(True)

        # 自动保存到本地 PyCharm 项目目录
        plt.savefig('strategy_result.png')
        print("回测完成！图片已保存为 strategy_result.png")
        plt.show()
