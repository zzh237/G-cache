# API Model Options

## Default: qwen-flash ✅

The code now uses **qwen-flash** by default (faster and cheaper than qwen-plus).

## Available Models

| Model | Speed | Cost | Use Case |
|-------|-------|------|----------|
| **qwen-flash** | ⚡ Fastest | 💰 Cheapest | Default (recommended) |
| qwen-plus | 🐢 Medium | 💰💰 Medium | More capable |
| qwen-turbo | ⚡ Fast | 💰 Cheap | Alternative |

## How to Use

### Hybrid Mode (Cache + API)
```bash
# Default: qwen-flash
python run_gsm8k_cache_API.py --llm_name hybrid_cache --use_cache

# Or specify explicitly:
python run_gsm8k_cache_API.py --llm_name hybrid_cache --use_cache --api_model qwen-flash
```

### API-Only Mode (No Cache)
```bash
# qwen-flash (fastest, cheapest)
python run_gsm8k_cache_API.py --llm_name qwen-flash

# qwen-plus (more capable)
python run_gsm8k_cache_API.py --llm_name qwen-plus

# qwen-turbo (alternative)
python run_gsm8k_cache_API.py --llm_name qwen-turbo
```

## What Changed

**Before**: Default was `qwen-plus`
**Now**: Default is `qwen-flash` (faster & cheaper)

Files updated:
- `GDesigner/llm/llm_cache_api.py` - Added qwen-flash support
- `GDesigner/llm/llm_cache_hybrid.py` - Changed default to qwen-flash
- `README.md` - Updated documentation
