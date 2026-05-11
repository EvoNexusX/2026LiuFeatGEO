from typing import List, Tuple
import numpy as np
import os
import json

from featgeo import utils
from featgeo.utils import get_answer, extract_citations_new, impression_wordpos_count_simple, impression_word_count_simple, impression_pos_count_simple

_global_cache = None
_global_cache_file = None


def _get_or_load_cache():
    """Return the shared cache object, loading it only when needed."""
    global _global_cache, _global_cache_file
    
    current_cache_file = utils.CACHE_FILE
    
    if _global_cache is None or _global_cache_file != current_cache_file:
        try:
            if os.path.exists(current_cache_file):
                print(f"[geo_runner_wrapper] Preloading cache: {current_cache_file}")
                with open(current_cache_file, 'r', encoding='utf-8') as f:
                    _global_cache = json.load(f)
                _global_cache_file = current_cache_file
                print(f"[geo_runner_wrapper] Cache loaded: {len(_global_cache)} records")
            else:
                _global_cache = {}
                _global_cache_file = current_cache_file
        except (json.JSONDecodeError, IOError, MemoryError) as e:
            print(f"[Warning] Failed to load cache ({current_cache_file}): {e}. Using an empty cache.")
            _global_cache = {}
            _global_cache_file = current_cache_file
    
    return _global_cache


def _get_num_completions():
    """Return the number of GPT fusion completions."""
    try:
        from featgeo import config
        return getattr(config, 'FUSION_NUM_COMPLETIONS_GPT', 5)
    except ImportError:
        return 5


def run_geo_with_d6(query: str, summaries_5: List[str], d6_text: str, print_responses: bool = False, 
                    skip_quality_eval: bool = False) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Run fusion and score the injected D6 source across three metrics."""
    summaries = summaries_5 + [d6_text]
    num_completions = _get_num_completions()  
    cache = _get_or_load_cache()
    
    if cache.get(query) is None:
        cache[query] = [{
            'sources': [{'summary': s, 'source': f'source_{i}'} for i, s in enumerate(summaries)],
            'responses': []
        }]
    
    answers = get_answer(query, summaries=summaries, num_completions=num_completions, n=len(summaries), loaded_cache=cache)
    responses = answers['responses'][-1]

    if print_responses:
        print(f"\n{'='*60}")
        print("Fusion responses (first 2 examples):")
        for i, ans in enumerate(responses[:2]):
            print(f"\n--- Response {i+1} ---\n" + ans[:400] + ('...' if len(ans) > 400 else ''))

    scores_wordpos = np.array([impression_wordpos_count_simple(extract_citations_new(x), len(summaries)) for x in responses])
    
    if len(responses) == 0 or scores_wordpos.size == 0:
        print("  [Warning] [Warning] GPT returned 0 valid responses. Falling back to zero score arrays.")
        num_sources = len(summaries)
        avg_scores_wordpos = np.zeros(num_sources)
        avg_scores_wordcount = np.zeros(num_sources)
        avg_scores_poscount = np.zeros(num_sources)
    else:
        avg_scores_wordpos = scores_wordpos.mean(axis=0)
        
        scores_wordcount = np.array([impression_word_count_simple(extract_citations_new(x), len(summaries)) for x in responses])
        avg_scores_wordcount = scores_wordcount.mean(axis=0)
        
        scores_poscount = np.array([impression_pos_count_simple(extract_citations_new(x), len(summaries)) for x in responses])
        avg_scores_poscount = scores_poscount.mean(axis=0)
    
    if skip_quality_eval:
        d6_quality = 0.0  # Single-objective mode ignores quality.
    else:
        from featgeo.geo_ad.d6_generator import evaluate_d6_quality
        quality_eval_responses = [response for response in responses if response and len(response) > 50]
        if len(quality_eval_responses) > 0:
            quality_scores = [
                evaluate_d6_quality(d6_text, query, fusion_response=response)
                for response in quality_eval_responses
            ]
            d6_quality = sum(quality_scores) / len(quality_scores)
        else:
            d6_quality = evaluate_d6_quality(d6_text, query, fusion_response=None)
    
    avg_scores_wordpos = np.append(avg_scores_wordpos, d6_quality)
    avg_scores_wordcount = np.append(avg_scores_wordcount, d6_quality)
    avg_scores_poscount = np.append(avg_scores_poscount, d6_quality)
    
    ad_score = float(avg_scores_wordpos[5])  # Main objective: D6 visibility at index 5.
    return ad_score, avg_scores_wordpos, avg_scores_wordcount, avg_scores_poscount

