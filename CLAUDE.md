# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-based paper filtering and recommendation system. Fetches papers from arXiv, uses LLM (OpenAI compatible) for intelligent scoring/filtering and translation, then pushes high-quality papers to Feishu groups.

## Architecture

Single file design - all logic in `arxiv.py`:

```
arxiv.py
├── LLMClient             # LLM client (OpenAI compatible) with retry logic
├── translate()           # Batch translate summaries (temperature=0.3)
├── search_arxiv_papers() # Fetch from arXiv API (XML parsing)
├── save_and_translate()  # Cache management (ID-based) + translation
├── rank_papers()         # LLM-based scoring and filtering (temperature=0.4)
├── send_feishu_message() # Push to Feishu webhook
└── cronjob()             # Main entry point
```

## Running Locally

```bash
pip install tqdm requests openai

# Required
export FEISHU_URL="..."
export LLM_API_KEY="..."

# Optional LLM Config (Defaults to DeepSeek)
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"

# Optional App Config
export QUERY="cs.IR"
export PROMPT="筛选要求"

python arxiv.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FEISHU_URL` | Feishu webhook URL | Required |
| `LLM_API_KEY` | LLM API key | Required |
| `LLM_BASE_URL` | LLM API base URL | https://api.deepseek.com |
| `LLM_MODEL` | Model name | deepseek-chat |
| `QUERY` | arXiv category | cs.IR |
| `LIMITS` | Papers to fetch | 10 |
| `PROMPT` | LLM filtering criteria | "无" |
| `MIN_SCORE` | Score threshold | 5.0 |
| `SCORE_INNOVATION` | Innovation weight (0-10) | 3 |
| `SCORE_QUALITY` | Technical quality weight (0-10) | 3 |
| `SCORE_PRACTICAL` | Practical value weight (0-10) | 2 |
| `SCORE_IMPACT` | Impact weight (0-10) | 2 |

## Advanced Configuration

### Scoring Weights

Customize the scoring weights to prioritize different aspects:

```bash
# Academic-oriented (prioritize innovation)
export SCORE_INNOVATION=4
export SCORE_QUALITY=3
export SCORE_PRACTICAL=1
export SCORE_IMPACT=2

# Industry-oriented (prioritize practicality)
export SCORE_INNOVATION=2
export SCORE_QUALITY=3
export SCORE_PRACTICAL=4
export SCORE_IMPACT=1

# Authority-first (prioritize impact)
export SCORE_INNOVATION=2
export SCORE_QUALITY=2
export SCORE_IMPACT=4
export SCORE_PRACTICAL=2
```

### Cache System

The system uses an ID-based cache (`arxiv_ids.json`) to avoid re-translating papers:
- Stores only arXiv IDs (e.g., "2310.12345")
- 90%+ smaller than full-content cache
- O(1) lookup performance
- Automatic deduplication based on arXiv ID

### Prompt Optimization

The system uses optimized prompts for translation and scoring:

**Translation Prompt:**
- Temperature: 0.3 (high consistency)
- Focus on core innovations
- Preserve technical terms (e.g., Transformer, BERT)
- Output length: 150-250 characters

**Scoring Prompt:**
- Temperature: 0.4 (balanced creativity)
- Structured 4-dimension scoring (Innovation, Quality, Practical, Impact)
- Clear keep/drop decision logic
- JSON output with reason (max 30 chars)

## Data Flow

1. `search_arxiv_papers()` → Fetch from arXiv API
2. `save_and_translate()` → Check cache (by arXiv ID), translate new papers
3. `rank_papers()` → LLM scoring and filtering (with progress indicators)
4. `send_feishu_message()` → Push to Feishu

## GitHub Actions

Four scheduled workflows (every 6 hours) for different arXiv topics:

### IR (Information Retrieval)
- **Query**: cs.IR
- **Focus**: Recommendation systems, search, ranking models
- **PROMPT**: Filters for recall/ranking/reranking models, user modeling, cold start, multi-objective optimization, LLM applications in RecSys/Search
- **Priority**: Industry applications (Google/Meta/ByteDance/Alibaba), top conferences (SIGIR/RecSys/KDD/WWW)

### CL (Computation and Language)
- **Query**: cs.CL
- **Focus**: NLP + LLM applications in recommendation/search
- **PROMPT**: Filters for LLM-based recommendation/search, conversational recommendation, intent understanding, query rewriting, RAG applications
- **Priority**: Industrial applications, top conferences (ACL/EMNLP/NAACL/KDD)

### CV (Computer Vision)
- **Query**: cs.CV
- **Focus**: Low-level vision and image restoration
- **PROMPT**: Filters for image de-reflection/denoising/deblurring, super-resolution, all-in-one models, lightweight models
- **Priority**: Industrial deployment, top conferences (CVPR/ICCV/ECCV), SOTA methods

### LG (Machine Learning)
- **Query**: cs.LG
- **Focus**: ML applied to recommendation/search/advertising
- **PROMPT**: Filters for CTR/CVR prediction, multi-task learning, RL for recommendation, GNN, AutoML, federated learning
- **Priority**: Industrial value, top conferences (ICML/NeurIPS/KDD), big tech practices

## Recent Optimizations (2026-01-20)

1. **Prompt Engineering**: Restructured prompts with clear scoring criteria and examples
2. **Cache Optimization**: Switched from full-content to ID-only cache (95% size reduction)
3. **Temperature Tuning**: Translation (0.3) and Scoring (0.4) for optimal stability
4. **Progress Indicators**: Added real-time feedback during LLM calls
5. **Configurable Scoring**: Environment variables for customizing scoring weights
6. **Workflow Updates**: Unified configuration across all GitHub Actions
