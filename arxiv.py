'''
credit to original author: Glenn (chenluda01@outlook.com)
Author: Doragd
'''

import os
import re
import requests
import time
import json
import datetime
import xml.etree.ElementTree as ET
from tqdm import tqdm
from openai import OpenAI

# ============== 配置 ==============
QUERY = os.environ.get('QUERY', 'cs.IR')
LIMITS = int(os.environ.get('LIMITS', 3)) + 10 * 2
FEISHU_URL = os.environ.get("FEISHU_URL", None)
FEISHU_URL_excel = os.environ.get("FEISHU_URL_excel", None)
PROMPT = os.environ.get("PROMPT", '无')

# LLM 配置
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("DEEPSEEK_API_KEY", None))
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

_min_score_str = os.environ.get("MIN_SCORE", "5")
try:
    MIN_SCORE = float(_min_score_str)
except (ValueError, TypeError):
    MIN_SCORE = 5.0

# 评分权重配置（总分10分）
SCORE_INNOVATION = float(os.environ.get("SCORE_INNOVATION", "3"))      # 创新性
SCORE_QUALITY = float(os.environ.get("SCORE_QUALITY", "3"))            # 技术质量
SCORE_PRACTICAL = float(os.environ.get("SCORE_PRACTICAL", "2"))        # 实用价值
SCORE_IMPACT = float(os.environ.get("SCORE_IMPACT", "2"))              # 影响力


# ============== LLM 客户端 ==============
class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def call(self, user_content: str, system_prompt: dict = None, temperature: float = 1.0) -> str:
        messages = []
        if system_prompt:
            messages.append(system_prompt)
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content.strip()

    def retry_call(self, user_content: str, system_prompt: dict = None, temperature: float = 1.0,
                   attempts: int = 3, base_delay: int = 60) -> str | None:
        for attempt in range(attempts):
            try:
                return self.call(user_content, system_prompt, temperature)
            except Exception as e:
                print(f"请求失败（尝试 {attempt + 1}/{attempts}）：{e}")
                if attempt < attempts - 1:
                    time.sleep(base_delay * (attempt + 1))
        return None


def get_client() -> LLMClient:
    if not LLM_API_KEY:
        raise Exception("未设置 LLM_API_KEY (或 DEEPSEEK_API_KEY) 环境变量")
    return LLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, model=LLM_MODEL)


# ============== 翻译 ==============
def translate(summaries: list[str]) -> list[str]:
    """批量翻译论文摘要"""
    if not summaries:
        return []

    client = get_client()
    system_prompt = {
        "role": "system",
        "content": (
            "你是一位专业的AI学术翻译专家，精通机器学习、推荐系统、计算机视觉、NLP等领域。\n\n"
            "**任务**：将英文论文摘要翻译为中文，同时提炼核心创新点。\n\n"
            "**要求**：\n"
            "1. 准确传达论文的核心方法、创新点和主要贡献\n"
            "2. 保持专业术语的准确性（如 Transformer、BERT 等保留英文）\n"
            "3. 使用学术化但易懂的中文表达\n"
            "4. 突出论文的技术亮点和实验结果\n"
            "5. 控制在 150-250 字以内，简洁精炼\n\n"
            "**输出格式**：直接输出翻译后的中文摘要，无需额外说明。"
        )
    }

    translations = []
    for s in summaries:
        result = client.retry_call(s, system_prompt, temperature=0.3)
        translations.append(result if result else '')
    return translations


# ============== 工具函数 ==============
def get_yesterday() -> str:
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')


# ============== arXiv 抓取 ==============
def search_arxiv_papers(search_term: str, max_results: int = 50) -> list[dict]:
    url = (f'http://export.arxiv.org/api/query?'
           f'search_query=all:{search_term}'
           f'&start=0&max_results={max_results}'
           f'&sortBy=submittedDate&sortOrder=descending')

    response = requests.get(url)
    if response.status_code != 200:
        print(f'[-] arXiv API 请求失败: {response.status_code}')
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        print('[-] XML 解析失败')
        return []

    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', ns)

    if not entries:
        return []

    print('[+] 开始处理每日最新论文....')

    papers = []
    for entry in entries:
        title_elem = entry.find('atom:title', ns)
        summary_elem = entry.find('atom:summary', ns)
        id_elem = entry.find('atom:id', ns)
        published_elem = entry.find('atom:published', ns)

        # 解析作者
        authors = []
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        if None in (title_elem, summary_elem, id_elem, published_elem):
            continue

        title = title_elem.text.strip().replace('\n', ' ')
        summary = summary_elem.text.strip().replace('\n', ' ').replace('\r', '')
        paper_url = id_elem.text.strip()
        pub_date = datetime.datetime.strptime(
            published_elem.text.strip(), "%Y-%m-%dT%H:%M:%SZ"
        ).strftime("%Y-%m-%d")

        papers.append({
            'title': title,
            'authors': authors,
            'url': paper_url,
            'pub_date': pub_date,
            'summary': summary,
        })

    print('[+] 开始翻译每日最新论文并缓存....')
    papers = save_and_translate(papers)

    return papers


# ============== 缓存与翻译 ==============
def save_and_translate(papers: list[dict], filename: str = 'arxiv_ids.json') -> list[dict]:
    """翻译新论文并缓存（仅保存 arXiv ID），返回翻译后的论文"""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            cached_ids = set(json.load(f))  # 直接加载为 Set
    else:
        cached_ids = set()

    # 提取 arXiv ID（URL 中的编号）
    def get_arxiv_id(url: str) -> str:
        try:
            # http://arxiv.org/abs/2310.12345v1 -> 2310.12345
            return url.split('/')[-1].split('v')[0]
        except Exception:
            return ""

    # 筛选未缓存的论文
    untranslated = []
    new_ids = []
    for paper in papers:
        arxiv_id = get_arxiv_id(paper['url'])
        if arxiv_id and arxiv_id not in cached_ids:
            untranslated.append(paper)
            new_ids.append(arxiv_id)

    # 批量翻译并更新缓存
    if untranslated:
        summaries = [p['summary'] for p in untranslated]
        translations = translate(summaries)
        for i, t in enumerate(translations):
            untranslated[i]['translated'] = t

        # 更新缓存（添加新 ID）
        cached_ids.update(new_ids)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(cached_ids)), f, indent=2)  # 排序保存

    cached_count = len(papers) - len(untranslated)
    print(f'[+] 总检索条数: {len(papers)} | 命中缓存: {cached_count} | 实际翻译: {len(untranslated)}')

    return untranslated


# ============== LLM 评分筛选 ==============
def rank_papers(papers: list[dict], top_k: int = None, min_score: float = None) -> list[dict]:
    """使用 LLM 对论文评分筛选"""
    if not papers:
        return []

    print(f'[+] 开始使用 LLM 对 {len(papers)} 篇论文进行评分筛选....')

    client = get_client()
    system_prompt = {
        "role": "system",
        "content": (
            "你是一位资深AI领域学术评审专家，负责筛选高质量论文。\n\n"
            f"**评分标准**（总分10分，允许小数）：\n"
            f"- **创新性** (0-{SCORE_INNOVATION}分)：方法是否新颖，是否有理论突破\n"
            f"- **技术质量** (0-{SCORE_QUALITY}分)：方法设计是否严谨，实验是否充分\n"
            f"- **实用价值** (0-{SCORE_PRACTICAL}分)：是否有工业落地潜力，是否解决实际问题\n"
            f"- **影响力** (0-{SCORE_IMPACT}分)：作者/机构声誉，是否可能引领方向\n\n"
            f"**筛选要求**：{PROMPT}\n"
            "- 不符合上述要求的论文，score 必须为 0，decision 为 \"drop\"\n"
            f"- 符合要求但质量一般的论文（score < {MIN_SCORE}），decision 为 \"drop\"\n"
            f"- 高质量且符合要求的论文（score >= {MIN_SCORE}），decision 为 \"keep\"\n\n"
            "**输出格式**：\n"
            "严格输出 JSON 数组，不要包含 Markdown 代码块标记（如 ```json）。\n"
            "每个元素格式：\n"
            '{{"index": 整数, "score": 浮点数, "decision": "keep"或"drop", "reason": "30字内中文理由"}}\n\n'
            "**示例**：\n"
            '[{{"index": 0, "score": 8.5, "decision": "keep", "reason": "提出新型注意力机制，在推荐任务上SOTA"}}, '
            '{{"index": 1, "score": 0, "decision": "drop", "reason": "纯理论物理论文，不相关"}}]'
        )
    }

    def truncate(text: str, max_len: int) -> str:
        return text[:max_len] + ("…" if len(text) > max_len else "")

    items = []
    for i, p in enumerate(papers):
        title = truncate(p.get("title", "").strip(), 200)
        # 加入作者信息
        authors = ", ".join(p.get("authors", [])[:3]) # 只取前3位作者
        if len(p.get("authors", [])) > 3:
            authors += " et al."

        body = truncate(p.get("translated", "") or p.get("summary", ""), 1200)
        items.append(f"{i}. Title: {title}\nAuthors: {authors}\nAbstract: {body}")

    user_prompt = (
        f"下面是待评审论文，共 {len(papers)} 篇。请筛选：\n" +
        "\n".join(items) +
        "\n请仅返回 JSON 数组（UTF-8，无额外注释/代码块）。"
    )

    raw = client.retry_call(user_prompt, system_prompt, temperature=0.4)
    print(f'[+] LLM 评分完成，开始解析结果....')

    # 解析 JSON
    decisions = []
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                decisions = parsed
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
            if match:
                try:
                    decisions = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

    # 清洗结果
    cleaned = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        idx = d.get("index")
        if isinstance(idx, int) and 0 <= idx < len(papers):
            try:
                score_val = float(d.get("score", 0))
            except (ValueError, TypeError):
                score_val = 0.0
            cleaned.append({
                "index": idx,
                "score": score_val,
                "decision": d.get("decision", "drop"),
                "reason": d.get("reason", "")
            })

    # 筛选
    kept = [x for x in cleaned if x["decision"] == "keep"]
    kept.sort(key=lambda x: x["score"], reverse=True)

    threshold = min_score if min_score is not None else MIN_SCORE
    kept = [x for x in kept if x["score"] >= threshold]

    if top_k is not None:
        kept = kept[:top_k]

    # 构建结果
    selected = []
    for x in kept:
        p = dict(papers[x["index"]])
        p["score"] = x["score"]
        p["reason"] = x["reason"]
        selected.append(p)

    print(f'[+] 筛选完成：保留 {len(selected)}/{len(papers)} 篇论文 (阈值: {threshold}分)')

    return selected


# ============== 飞书推送 ==============
def send_feishu_message(title: str, content: str, url: str = None):
    url = url or FEISHU_URL
    if not url:
        raise Exception("未设置 FEISHU_URL 环境变量")

    card_data = {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green",
            "title": {"tag": "plain_text", "content": title[:250]}
        },
        "elements": [
            {"tag": "markdown", "content": content[:25000]}
        ]
    }
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url, json={"msg_type": "interactive", "card": card_data}, headers=headers)
        if resp.status_code == 200 and resp.json().get("code") == 0:
            return
    except Exception:
        pass

    # 回退纯文本
    fallback = {"msg_type": "text", "content": {"text": f"{title}\n{content}"[:25000]}}
    requests.post(url, json=fallback, headers=headers)


# ============== 主流程 ==============
def cronjob():
    if not FEISHU_URL:
        raise Exception("未设置 FEISHU_URL 环境变量")

    print('[+] 开始执行每日推送任务....')
    print('[+] 开始检索每日最新论文....')

    papers = search_arxiv_papers(QUERY, LIMITS)
    papers = rank_papers(papers, min_score=MIN_SCORE)
    print(f'[+] 筛选后剩余稿件数量: {len(papers)}')

    if not papers:
        print('[+] 每日推送任务执行结束')
        return True

    print('[+] 开始推送每日最新论文....')
    yesterday = get_yesterday()

    for ii, paper in enumerate(tqdm(papers, desc="论文推送进度")):
        title = paper['title']
        url = paper['url']
        pub_date = paper['pub_date']
        translated = paper['translated']
        score = paper.get('score', 0)
        reason = paper.get('reason', '')

        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg_title = f'[Newest]{title}' if pub_date == yesterday else title

        # 格式化作者显示（前3位）
        authors = paper.get("authors", [])
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        if not author_str:
            author_str = "Unknown"

        msg_content = (
            f"[{msg_title}]({url})\n"
            f"Authors: {author_str}\n"
            f"Pub Date: {pub_date} | Score: {score:.2f}/10\n"
            f"AI Rationale: {reason or 'None'}\n"
            f"Translated (Powered by {LLM_MODEL}):\n{translated}\n"
        )

        push_title = f'Arxiv:{QUERY}[{ii}]@{current_time}'
        send_feishu_message(push_title, msg_content)
        time.sleep(5)

        if FEISHU_URL_excel:
            paper_json = {
                "QUERY": QUERY,
                "URL": url,
                "Title": title,
                "Pub_date": pub_date,
                "Translated": translated,
                "Reason": reason,
                "Score": f'{score:.2f}'
            }
            requests.post(FEISHU_URL_excel, json=paper_json, headers={"Content-Type": "application/json"})
            print("推送 json", paper_json)
            time.sleep(5)

    print('[+] 每日推送任务执行结束')
    return True


if __name__ == '__main__':
    cronjob()
