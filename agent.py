import os
import requests
import json
from firecrawl import Firecrawl
import akshare as ak

# --- 配置区 ---
FIRECRAWL_KEY = "fc-55132cab79b141c9a071a67bff11d736"
GEMINI_API_KEY = "AIzaSyBr8PHg9n1ms-ZfM1aXVy3auLLFeUpHyRY"
MONITOR_URL = "https://www.federalreserve.gov/newsevents/pressreleases.htm"


def get_available_models():
    """自动获取当前 API Key 拥有的所有模型"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            models = res.json().get('models', [])
            # 过滤出支持生成内容且包含 gemini 的模型
            return [m['name'] for m in models if 'generateContent' in m.get('supportedGenerationMethods', [])]
        return []
    except:
        return []


def start_agent():
    print("\n🚀 贵金属 AI 研究员：任务启动...")
    app = Firecrawl(api_key=FIRECRAWL_KEY)

    # 1. 获取实时行情 (AkShare)
    market_info = "行情获取略过"
    try:
        df = ak.gold_zh_spot()
        if not df.empty:
            market_info = f"当前金价: {df.iloc[0]['最新价']} 元/克"
            print(f"💰 获取成功: {market_info}")
    except:
        pass

    # 2. 抓取美联储公告
    print("📡 正在抓取美联储最新公告...")
    try:
        response = app.scrape(url=MONITOR_URL, formats=['markdown'])
        raw_markdown = response.get('markdown', '') if isinstance(response, dict) else getattr(response, 'markdown', '')
        if not raw_markdown: return
        print(f"📄 成功获取 {len(raw_markdown)} 字符文本")
    except Exception as e:
        print(f"❌ 爬虫故障: {e}");
        return

    # 3. 核心修复：自动探测模型路径
    available_models = get_available_models()

    # 备选列表，如果自动获取失败，则尝试这些硬编码路径
    model_paths = available_models if available_models else [
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-pro",
        "models/gemini-pro"
    ]

    print(f"🔍 账号可用模型探测结果: {model_paths}")

    success = False
    for path in model_paths:
        # 去掉路径中可能重复的 models/ 前缀
        clean_path = path if path.startswith('models/') else f"models/{path}"
        print(f"🧠 尝试调用: {clean_path}...")

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/{clean_path}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"你是一名资深研究员。分析 1/28/2026 美联储公告对金价 {market_info} 的利多/利空逻辑。\n内容：{raw_markdown[:6000]}"
                }]
            }]
        }

        try:
            res = requests.post(endpoint, json=payload, headers={'Content-Type': 'application/json'})
            if res.status_code == 200:
                report = res.json()['candidates'][0]['content']['parts'][0]['text']
                print("\n" + "★" * 15 + " AI 研报产出 " + "★" * 15)
                print(report)
                with open("Gold_Strategy_Report.md", "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"\n💾 研报已保存。")
                success = True;
                break
        except:
            continue

    if not success:
        print("\n❌ 依然无法连接 AI 模型。请检查 API Key 状态。")


if __name__ == "__main__":
    start_agent()