import akshare as ak
import pandas as pd
from tqdm import tqdm
import time
import os


def get_csi1000_data(output_dir="data_csi1000"):
    """
    抓取中证1000成分股名单及其历史行情
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("正在获取中证1000成分股最新名单...")
    # 000852 是中证1000的指数代码
    try:
        stock_list_df = ak.index_stock_cons(symbol="000852")
        stock_codes = stock_list_df['品种代码'].tolist()
        # 保存名单备用
        stock_list_df.to_csv(f"{output_dir}/constituent_list.csv", index=False, encoding='utf-8-sig')
        print(f"成功获取 {len(stock_codes)} 只成分股。")
    except Exception as e:
        print(f"获取名单失败: {e}")
        return

    print("开始下载历史日线数据（后复权）...")
    # 我们抓取过去 3 年的数据作为示例
    start_date = "20210101"
    end_date = "20251231"

    # 为了演示，我们只抓取前10只，实际操作时请删除 [:10]
    for code in tqdm(stock_codes[:10]):
        try:
            # adjust="hfq" 代表后复权，这是回测最常用的模式
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"
            )

            if not df.empty:
                # 保存为CSV，以股票代码命名
                df.to_csv(f"{output_dir}/{code}.csv", index=False)

            # 适当休眠，防止请求过快被封IP
            time.sleep(0.2)

        except Exception as e:
            print(f"\n下载股票 {code} 失败: {e}")

    print(f"\n数据下载完成，保存在 '{output_dir}' 目录下。")


if __name__ == "__main__":
    get_csi1000_data()

