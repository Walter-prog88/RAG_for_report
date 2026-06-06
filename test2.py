import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicTransformer
import warnings

# 忽略版本兼容性产生的 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)


# ==========================================
# 1. 数据模拟与行为数据对齐
# ==========================================
def load_and_align_data():
    print("正在生成模拟数据并进行情绪对齐...")
    # 模拟 QQQ 分钟 K 线 (2000分钟)
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=2000, freq="1min")
    df = pd.DataFrame({
        'time': dates,
        'open': np.random.randn(2000).cumsum() + 400,
        'high': np.random.randn(2000).cumsum() + 405,
        'low': np.random.randn(2000).cumsum() + 395,
        'close': np.random.randn(2000).cumsum() + 400,
        'volume': np.random.randint(1000, 5000, size=2000)
    })

    # 模拟情绪数据 (非连续，模拟新闻发布)
    sent_dates = pd.date_range(start="2025-01-01", periods=500, freq="4min")
    df_sent = pd.DataFrame({
        'time': sent_dates,
        'sentiment': np.random.uniform(-1, 1, size=500)
    })

    # 【行为金融核心】时空对齐：使用 merge_asof 确保无未来函数
    df = pd.merge_asof(df.sort_values('time'), df_sent.sort_values('time'), on='time', direction='backward')
    df['sentiment'] = df['sentiment'].fillna(0)

    # 预处理：平滑情绪并计算目标收益（预测未来5分钟）
    df['sent_ema'] = df['sentiment'].ewm(span=30).mean()
    df['target_return'] = df['close'].shift(-5) / df['close'] - 1

    # 构造一个价格与情绪的交互项，强迫模型考虑行为因素
    df['price_sent_dist'] = df['sent_ema'] * (df['close'] / df['close'].rolling(20).mean() - 1)

    df.dropna(inplace=True)
    return df


# ==========================================
# 2. 非线性因子挖掘 (遗传算法)
# ==========================================
def mine_nonlinear_factors(df):
    print("开始进化非线性 Alpha 因子...")
    # 增加交互项 price_sent_dist 提高行为金融相关度
    features = ['open', 'high', 'low', 'close', 'volume', 'sent_ema', 'price_sent_dist']
    X = df[features].values
    y = df['target_return'].values

    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'sin', 'max', 'min']

    gp = SymbolicTransformer(
        feature_names=features,
        function_set=function_set,
        generations=15,
        population_size=1000,
        n_components=1,
        parsimony_coefficient=0.0005,
        random_state=42,
        verbose=1
    )

    gp.fit(X, y)
    df['best_factor'] = gp.transform(X)
    print(f"\n[进化完成] 最优因子公式: {gp._best_programs[0]}")
    return df


# ==========================================
# 3. 向量化回测系统 (含非对称止损与摩擦成本)
# ==========================================
def backtest_strategy(df, threshold=1.0, cost=0.0002):
    print("开始执行回测逻辑...")
    df = df.copy()

    # A. 信号生成：因子 Z-Score 化
    df['factor_z'] = (df['best_factor'] - df['best_factor'].rolling(60).mean()) / df['best_factor'].rolling(60).std()

    df['signal'] = 0
    df.loc[df['factor_z'] > threshold, 'signal'] = 1  # 多头
    df.loc[df['factor_z'] < -threshold, 'signal'] = -1  # 空头

    # B. 计算收益与非对称止损
    df['market_ret'] = df['close'].pct_change()
    df['strategy_ret'] = df['signal'].shift(1) * df['market_ret']

    # 【行为金融逻辑】非对称止损：情绪悲观时止损线更敏感
    df['stop_loss_limit'] = np.where(df['sent_ema'] < 0, -0.005, -0.015)
    df['is_stop_loss'] = df['strategy_ret'] < df['stop_loss_limit']
    df.loc[df['is_stop_loss'], 'strategy_ret'] = df['stop_loss_limit']

    # C. 摩擦成本与净收益
    df['trades'] = df['signal'].diff().abs()
    df['net_ret'] = df['strategy_ret'] - (df['trades'] * cost)

    # D. 累计收益
    df['cum_market'] = (1 + df['market_ret'].fillna(0)).cumprod()
    df['cum_strategy'] = (1 + df['net_ret'].fillna(0)).cumprod()

    return df


# ==========================================
# 4. 性能展示
# ==========================================
def plot_results(df):
    df_clean = df.dropna(subset=['net_ret'])
    annual_factor = np.sqrt(252 * 390)
    sharpe = annual_factor * df_clean['net_ret'].mean() / df_clean['net_ret'].std()

    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['cum_market'], label='QQQ Benchmark', color='gray', alpha=0.5)
    plt.plot(df['time'], df['cum_strategy'], label='Behavioral Alpha Strategy', color='red', linewidth=1.5)

    plt.title(f"QQQ Alpha Mining Result | Sharpe Ratio: {sharpe:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    try:
        # 第一步：准备数据
        data = load_and_align_data()

        # 第二步：挖掘因子
        data_with_factor = mine_nonlinear_factors(data)

        # 第三步：运行回测 (确保此处函数名与定义一致)
        final_results = backtest_strategy(data_with_factor)

        # 第四步：绘图
        plot_results(final_results)

    except Exception as e:
        print(f"运行出错: {e}")
        print("提示：请确保已执行 pip install 'scikit-learn<1.5.0' gplearn")