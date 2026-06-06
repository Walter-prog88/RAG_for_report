import asyncio
from playwright.async_api import async_playwright
from datetime import datetime, timedelta

# 配置区
SUMMARY_INTERVAL_MINS = 1  # 测试用：1分钟汇总一次
period_cache = []
seen_news = set()
next_summary_time = datetime.now() + timedelta(minutes=SUMMARY_INTERVAL_MINS)


async def scrape_and_summarize():
    global period_cache, next_summary_time, seen_news

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        page = await context.new_page()

        print(f"🚀 系统启动... 正在监听金十数据")

        while True:
            current_time = datetime.now()
            try:
                # 访问首页
                await page.goto("https://www.jin10.com/", wait_until="domcontentloaded", timeout=60000)

                # 等待快讯项加载
                await page.wait_for_selector(".jin-flash-item-container", timeout=15000)
                news_items = await page.query_selector_all(".jin-flash-item-container")

                if not news_items:
                    print(f"[{current_time.strftime('%H:%M:%S')}] 未发现内容项...")
                    continue

                new_found = 0
                # 倒序遍历最新的 15 条
                for item in reversed(news_items[:15]):
                    # --- 改进的提取逻辑 ---
                    # 1. 提取时间
                    time_el = await item.query_selector(".time")
                    t = (await time_el.inner_text()).strip() if time_el else current_time.strftime("%H:%M")

                    # 2. 提取内容：尝试所有可能的类名，如果都失败，直接抓取整个 item 的 inner_text
                    content_el = await item.query_selector(".content") or \
                                 await item.query_selector(".jin-flash-text") or \
                                 await item.query_selector(".right-content")

                    if content_el:
                        c = (await content_el.inner_text()).strip()
                    else:
                        # 兜底：如果找不到特定标签，提取整个容器的文本并清洗
                        raw_c = await item.inner_text()
                        # 过滤掉时间字符串，只留正文
                        c = raw_c.replace(t, "").strip().replace("\n", " ")

                    # 3. 去重并展示
                    if c and c not in seen_news and len(c) > 10:
                        seen_news.add(c)
                        period_cache.append(f"【{t}】 {c}")
                        new_found += 1
                        print(f"\n🔔 [新快讯] {t}")
                        print(f"{c}")
                        print("-" * 40)

                if new_found == 0:
                    print(f"实时监听中... [当前记录: {len(seen_news)}条]", end="\r")

                # 4. 周期汇总逻辑
                if current_time >= next_summary_time:
                    print(f"\n\n{'=' * 20} 🕒 周期汇总时刻 {'=' * 20}")
                    if period_cache:
                        filename = f"Summary_{current_time.strftime('%H%M')}.txt"
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(f"--- 今日要闻汇总 ({current_time.strftime('%Y-%m-%d %H:%M')}) ---\n\n")
                            f.write("\n\n".join(period_cache))
                        print(f"✅ 报告已生成: {filename} (包含 {len(period_cache)} 条内容)")
                        period_cache = []  # 清空缓存开启下一轮
                    else:
                        print("📝 周期内无新消息，跳过生成。")
                    next_summary_time = current_time + timedelta(minutes=SUMMARY_INTERVAL_MINS)

            except Exception as e:
                print(f"\n❌ 连接异常: {e}")

            await asyncio.sleep(45)  # 45秒扫描一次


if __name__ == "__main__":
    asyncio.run(scrape_and_summarize())