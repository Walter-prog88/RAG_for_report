import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

# ---------------- 配置区域 ----------------
# 1. 设置代理 (使用您刚才确认的端口)
PROXY_URL = "http://127.0.0.1:33210"
yf.set_config(proxy=PROXY_URL)

# 2. 定义我们要扫描的股票列表 (模拟全市场扫描，你可以自己添加更多)
# 这里我选了图中前几名比较热门的
target_symbols = ['TSLA', 'NVDA', 'AAPL', 'AMD', 'AMZN', 'META', 'MSFT', 'MARA', 'PLTR']


# ---------------- 工具函数 ----------------

def parse_contract_symbol(contract_name):
    """
    将标准合约代码 (TSLA251219C00500000)
    转换为图中那种简写格式 (251219.C.500)
    """
    # 简单的正则或切片提取
    # 假设格式是标准的：6位日期 + 1位类型 + 8位价格
    try:
        # 提取日期 (跳过股票代码部分，寻找数字开头)
        # 这种硬解析比较依赖格式，这里做个简单处理
        # 实际上 yfinance 返回的 contractSymbol 通常是 "TSLA251219C00500000"
        import re
        match = re.search(r'(\d{6})([CP])(\d{8})', contract_name)
        if match:
            date_str = match.group(1)
            type_str = match.group(2)
            price_str = str(int(match.group(3)) / 1000)  # 去掉前导0并处理小数
            # 如果是整数去掉 .0
            if price_str.endswith(".0"):
                price_str = price_str[:-2]
            return f"{date_str}.{type_str}.{price_str}"
        return contract_name
    except:
        return contract_name


def get_option_stats(symbol):
    print(f"正在分析 {symbol} ...")
    try:
        ticker = yf.Ticker(symbol)
        exp_dates = ticker.options

        if not exp_dates:
            return None

        total_call_vol = 0
        total_put_vol = 0
        leap_call_vol = 0
        leap_put_vol = 0

        # 用于寻找最热合约
        most_active_contract = {"symbol": "", "vol": -1}

        # 确定 LEAP 的界限日期（当前日期 + 3个月）
        today = datetime.now()
        leap_threshold = today + timedelta(days=90)

        # 遍历所有到期日 (这步比较耗时，因为要下载很多数据)
        # 为了演示速度，这里只取前 5 个和最后 2 个到期日作为样本
        # 如果你想全量跑，去掉切片 [:5] 即可
        scan_dates = exp_dates

        for date_str in scan_dates:
            # 获取该日期的期权链
            try:
                opt = ticker.option_chain(date_str)
                calls = opt.calls
                puts = opt.puts

                # 1. 累加总成交量
                c_vol = calls['volume'].fillna(0).sum()
                p_vol = puts['volume'].fillna(0).sum()
                total_call_vol += c_vol
                total_put_vol += p_vol

                # 2. LEAP 计算 (判断到期日是否 > 3个月)
                exp_dt = datetime.strptime(date_str, '%Y-%m-%d')
                if exp_dt > leap_threshold:
                    leap_call_vol += c_vol
                    leap_put_vol += p_vol

                # 3. 寻找全场最热合约 (Call 和 Put 一起比)
                # 找出 Call 中最大的一行
                if not calls.empty:
                    max_c = calls.loc[calls['volume'].idxmax()]
                    if max_c['volume'] > most_active_contract['vol']:
                        most_active_contract = {
                            "symbol": parse_contract_symbol(max_c['contractSymbol']),
                            "vol": max_c['volume']
                        }

                # 找出 Put 中最大的一行
                if not puts.empty:
                    max_p = puts.loc[puts['volume'].idxmax()]
                    if max_p['volume'] > most_active_contract['vol']:
                        most_active_contract = {
                            "symbol": parse_contract_symbol(max_p['contractSymbol']),
                            "vol": max_p['volume']
                        }

            except Exception as e:
                continue  # 跳过错误的日期

        # --- 计算最终指标 ---
        total_vol = total_call_vol + total_put_vol

        # C/P Ratio
        cp_ratio = total_call_vol / total_put_vol if total_put_vol > 0 else 0

        # LEAP Ratio
        leap_ratio = leap_call_vol / leap_put_vol if leap_put_vol > 0 else 0

        return {
            "Symbol": symbol,
            "Vol(万)": round(total_vol / 10000, 2),  # 换算成万张
            "C/P比": round(cp_ratio, 2),
            "LEAP*": round(leap_ratio, 2) if leap_ratio > 0 else "-",
            "最热期权": most_active_contract['symbol']
        }

    except Exception as e:
        print(f"Error {symbol}: {e}")
        return None


# ---------------- 主程序 ----------------

data_list = []

for sym in target_symbols:
    stats = get_option_stats(sym)
    if stats:
        data_list.append(stats)

# 创建 DataFrame 并排序
df = pd.DataFrame(data_list)
# 按成交量降序排列
df = df.sort_values(by="Vol(万)", ascending=False).reset_index(drop=True)
# 添加 Rank 列
df.index = df.index + 1
df.index.name = 'Rank'

print("\n" + "=" * 50)
print("复刻版 - 个股期权成交量 TOP")
print("=" * 50)
print(df)

# 如果想保存成 Excel
# df.to_excel("option_report.xlsx")