# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

# Backend used only for D6 candidate page generation.
# Options:
#   - "gpt": use the OpenAI-compatible API settings below
#   - "ollama": use the local Ollama settings below
D6_MODEL_BACKEND = "gpt"


# ---------------------------------------------------------------------------
# OpenAI-compatible API settings
# ---------------------------------------------------------------------------

# Put one or more API keys here. In parallel mode, run_parallel.py assigns one
# key to each shard in order, reusing keys round-robin if shards > keys.
OPENAI_API_KEYS = [
    "your_api_key",
]

# Use the official OpenAI endpoint or any compatible relay endpoint.
OPENAI_API_BASE = "your_api_base"


# ---------------------------------------------------------------------------
# GPT model choices
# ---------------------------------------------------------------------------

# Generates the D6 candidate/ad page when D6_MODEL_BACKEND = "gpt".
D6_GENERATION_MODEL = "gpt-4o-mini"

# Used for lightweight helper tasks such as ad-theme extraction.
AUXILIARY_MODEL = "gpt-4o-mini"

# Evaluates D6 content quality in the main Pareto objective.
D6_QUALITY_EVAL_MODEL = "gpt-4o-mini"

# Generates cited fusion answers from D1-D6 summaries.
FUSION_MODEL = "gpt-4o-mini"

# Infers feature configs from selected source summaries.
FEATURE_EXTRACTION_MODEL = "gpt-4o-mini"

# Cleans source text only when GPT source cleaning is enabled by caller code.
SOURCE_CLEANING_MODEL = "gpt-4o-mini"

# Sampling temperature for GPT-backed D6 generation.
D6_TEMPERATURE = 0.5

# Number of fusion answers generated per D6 evaluation. Scores are averaged
# across completions. Cached responses are reused as-is.
FUSION_NUM_COMPLETIONS_GPT = 5


# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------

# Ollama is used only by the D6 page generator when D6_MODEL_BACKEND = "ollama".
# Fusion, feature extraction, GEO rewrites, and quality evaluation remain GPT.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "qwen3:1.7b"

# Optional thinking-mode sampling settings for models that support it.
OLLAMA_ENABLE_THINKING = False
OLLAMA_THINKING_TEMPERATURE = 0.6
OLLAMA_THINKING_TOP_P = 0.95

# Default Ollama sampling settings.
OLLAMA_TEMPERATURE = 0.7
OLLAMA_TOP_P = 0.8
OLLAMA_TOP_K = 20
OLLAMA_MIN_P = 0.0


# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------

# False uses the summaries already stored in GEO-Bench. True asks GPT to clean
# or summarize raw source text where that caller path is enabled.
USE_GPT_SUMMARY = False

# Used by SAMPLING_MODE = "continuous". Ignored by "all" and "random".
SAMPLE_LIMIT = 4

# Sampling modes:
#   - "continuous": run samples [0, SAMPLE_LIMIT)
#   - "random": run RANDOM_SAMPLE_SIZE random samples
#   - "all": run the full GEO-Bench test split
SAMPLING_MODE = "all"

# Used only when SAMPLING_MODE = "random".
RANDOM_SAMPLE_SIZE = 50
RANDOM_SAMPLING_SEED = 42


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

# If False, run_parallel.py falls back to a single run_geo_ad.py process.
ENABLE_PARALLEL = True

# Number of shard processes launched by run_parallel.py.
PARALLEL_SHARDS = 2

# Optional manual range, for example (0, 20). When set, run_parallel.py launches
# exactly one shard for that range.
MANUAL_SHARD_RANGE = None


# ---------------------------------------------------------------------------
# Dynamic GA optimization
# ---------------------------------------------------------------------------

# Method 11 in the experiment table.
ENABLE_METHOD_11 = True

# Options:
#   - "pareto": optimize ad visibility and quality with NSGA-II
#   - "single": optimize D6 visibility only
METHOD_11_MODE = "pareto"

# Population size and number of evolutionary generations.
METHOD_11_POPSIZE = 8
METHOD_11_GENERATIONS = 8

# Mutation probability for each feature dimension.
METHOD_11_PMUT = 0.5

# Random seed for GA operations and feature sampling.
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Topic-level query probing
# ---------------------------------------------------------------------------

# Disabled by default because it increases cost. When enabled, related probe
# queries select the most frequently cited D1-D5 pages as feature exemplars.
ENABLE_QUERY_PROBING = False
QUERY_PROBE_COUNT = 5
QUERY_PROBE_TOP_N = 3
QUERY_PROBE_INCLUDE_ORIGINAL = True
QUERY_PROBE_NUM_COMPLETIONS = 1
QUERY_PROBE_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Cache files
# ---------------------------------------------------------------------------

# Reuse GA offspring evaluations within the same shard/cache suffix.
ENABLE_GA_CACHE = True

# Compatibility switches used by geo_functions.py. Keep both False for normal
# main experiments.
STATIC_CACHE = False
SUBJ_STATIC_CACHE = False

# Base cache files. Parallel runs create shard-specific variants automatically.
GLOBAL_CACHE_FILE = "data/global_cache.json"
GEO_CACHE_FILE = "data/geo_optimizations_cache.json"


if D6_MODEL_BACKEND == "ollama":
    D6_MODEL = "ollama-" + OLLAMA_MODEL_NAME.replace(":", "-")
    D6_TEMPERATURE = OLLAMA_TEMPERATURE
else:
    D6_MODEL = D6_GENERATION_MODEL

AD_GEO_CACHE_FILE = f"data/ad_geo_optimizations_cache_{D6_MODEL}.json"

# Print example fusion responses during evaluation. Keep False for large runs.
PRINT_RESPONSES = False
