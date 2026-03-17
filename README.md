# LLM Paper Recommender

基于大模型的论文筛选与推送系统。自动从 arXiv 获取最新论文，通过 LLM 进行智能筛选和评分，将高质量论文推送到飞书群组。

## 功能特点

- **自动抓取**：定时从 arXiv 获取最新论文（cs.IR、cs.CL、cs.CV、cs.LG）
- **智能筛选**：使用 LLM（默认 DeepSeek，支持任何 OpenAI 兼容接口）对论文进行评分和筛选
- **自动翻译**：摘要自动翻译为中文
- **飞书推送**：筛选后的论文自动推送到飞书群组
- **缓存机制**：已处理论文缓存避免重复翻译

## 快速开始

### 环境变量

**必填：**
```bash
export FEISHU_URL="your_feishu_webhook"
export LLM_API_KEY="your_api_key"  # 或 DEEPSEEK_API_KEY
```

**LLM 配置（可选，支持任何 OpenAI 兼容服务）：**
```bash
export LLM_BASE_URL="https://api.deepseek.com"  # 默认值
export LLM_MODEL="deepseek-chat"                # 默认值
```

**其他可选配置：**
```bash
export FEISHU_URL_excel="your_webhook"    # 推送到飞书表格
export QUERY="cs.IR"                       # arXiv 查询主题
export LIMITS=10                           # 每次获取论文数量
export MIN_SCORE=5                         # 最低分数阈值
export PROMPT="你的筛选要求"               # 论文筛选提示词
```

### 本地运行

```bash
pip install tqdm requests openai
python arxiv.py
```

### GitHub Actions

项目配置了定时任务，每6小时自动执行一次推送：
- `push_arxiv_daily_IR.yml` - 信息检索方向
- `push_arxiv_daily_cl.yml` - 计算语言学方向
- `push_arxiv_daily_cv.yml` - 计算机视觉方向
- `push_arxiv_daily_lg.yml` - 机器学习方向

## 项目结构

```
├── arxiv.py                # 主程序
├── arxiv_ids.json          # 论文 ID 缓存（避免重复评分/翻译）
└── .github/workflows/      # GitHub Actions 配置
```

## License

MIT
