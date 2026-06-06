import openai
import json
import sys


# 彻底移除中文，确保 ASCII 环境下也能运行
def get_ai_trading_strategy(news_text, ticker_data, api_key):
    # 这里我们把 client 初始化放在 try 内部
    try:
        client = openai.OpenAI(api_key=api_key)

        # 使用纯英文 System Prompt
        system_prompt = """
        You are a Quant Derivatives Trader with 10 years experience.
        Task: 
        1. Evaluate news impact on Implied Volatility (IV) (score 0-10).
        2. Recommend a risk-limited option strategy (e.g. Bull Call Spread).
        Output MUST be in JSON format with fields: 
        impact_score, iv_forecast (UP/DOWN/NEUTRAL), reasoning, recommended_strategy.
        """

        user_content = f"News: {news_text}. Price: {ticker_data['underlying_price']}. IV: {ticker_data['atm_iv']}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None


if __name__ == "__main__":
    # 模拟数据全换成英文
    mock_ticker_data = {
        "underlying_price": 185.20,
        "atm_iv": 0.42,
        "expiration": "2024-06-21"
    }
    mock_news = "Nvidia announces deep AI collaboration with Microsoft, beating expectations."

    # --- 填入你的 Key ---
    MY_API_KEY = "sk-xxxx"

    print("Connecting to AI Trader, please wait...")

    result = get_ai_trading_strategy(mock_news, mock_ticker_data, MY_API_KEY)

    if result:
        # 使用 indent=4 打印 JSON
        print(json.dumps(result, indent=4))
    else:
        print("Failed to get result.")