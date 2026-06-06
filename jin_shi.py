import asyncio
from playwright.async_api import async_playwright
from datetime import datetime


async def scrape_jin10_live():
    async with async_playwright() as p:
        # 1. 启动浏览器
        browser = await p.chromium.launch(headless=True)
        # 2. 创建上下文和页面
        context = await browser.new_context()
        page = await context.new_page()

        print(f"正在建立与金十数据的连接... {datetime.now().strftime('%H:%M:%S')}")

        try:
            # 3. 访问首页并等待数据加载
            await page.goto("https://www.jin10.com/", wait_until="domcontentloaded", timeout=60000)

            # 4. 等待快讯列表容器出现
            # 注意：如果以下选择器失效，尝试改为 ".jin-flash-item"
            await page.wait_for_selector(".jin-flash-item-container", timeout=15000)

            # 5. 抓取快讯项
            news_items = await page.query_selector_all(".jin-flash-item-container")

            print(f"\n--- 实时捕获金十快讯 (共 {len(news_items[:5])} 条) ---")

            for item in news_items[:5]:
                # 尝试抓取时间
                time_el = await item.query_selector(".time")
                # 尝试抓取正文内容 (金十通常将正文放在 .content 或 .text 类中)
                content_el = await item.query_selector(".content")

                if content_el:
                    time_val = await time_el.inner_text() if time_el else "未知时间"
                    content_val = await content_el.inner_text()

                    # 格式化输出
                    print(f"【{time_val.strip()}】")
                    print(f"内容: {content_val.strip()}")
                    print("-" * 50)
                else:
                    # 如果 .content 没抓到，尝试打印整个 item 的文本
                    raw_text = await item.inner_text()
                    print(f"捕获原始内容: {raw_text.strip()[:150]}...")
                    print("-" * 50)

        except Exception as e:
            print(f"抓取过程中发生错误: {e}")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(scrape_jin10_live())