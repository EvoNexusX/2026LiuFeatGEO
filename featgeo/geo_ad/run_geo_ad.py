"""Main FeatGEO experiment worker.

This module runs the shared experiment flow used by the main pipeline:
feature inference from D1-D5, baseline D6 selection, the original GEO
rewriting methods, and the optional dynamic GA optimization step.
"""

import os
import json
import argparse
from typing import List, Dict
from datasets import load_dataset
import numpy as np

from featgeo.geo_ad.geo_methods import GEO_METHODS, extract_sources_from_dataset
from featgeo.geo_ad.feature_schema import FeatureConfig
from featgeo.geo_ad.feature_extractor import infer_config_from_text
from featgeo.geo_ad.d6_population_manager import load_or_generate_d6_population, select_baseline_d6
from featgeo.geo_ad.result_formatter import print_method_result, print_summary_table, print_visibility_matrix, print_average_matrix
from featgeo.geo_ad.geo_runner_wrapper import run_geo_with_d6
from featgeo.geo_ad.ga_optimizer import run_ga_single_scene, run_ga_multi_objective
from featgeo.geo_ad.query_probe import select_probe_feature_summaries
from featgeo import config


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FeatGEO main experiment worker.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--sampling-mode", choices=["continuous", "sparse", "random", "all"], default=None)
    parser.add_argument("--random-indices", default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--cache-suffix", default="")
    parser.add_argument("--global-cache-file", default=None)
    return parser.parse_args()


def apply_geo_methods_to_ad(query: str, summaries_5, baseline_d6: str, baseline_score: float,
                           baseline_scores_vector = None,
                           baseline_scores_wordcount = None,
                           baseline_scores_poscount = None,
                           initial_population: List[FeatureConfig] = None,
                           initial_scores: List[float] = None,
                           initial_scores_vectors: List = None,
                           initial_scores_vectors_wc: List = None,
                           initial_scores_vectors_pc: List = None,
                           initial_d6_pool: List[str] = None,
                           ad_theme: str = None,
                           enable_method_11: bool = False, 
                           m11_generations: int = 3, m11_pmut: float = 0.15,
                           m11_mode: str = 'single',
                           print_responses: bool = False,
                           enable_ga_cache: bool = True,
                           ga_eval_cache: List[Dict] = None):
    """Run the original GEO methods plus the optional dynamic GA method."""
    baseline_content_quality = None
    if baseline_scores_vector is not None:
        from featgeo.geo_ad.multi_objective import calculate_objectives
        _, baseline_content_quality = calculate_objectives(baseline_scores_vector)
    
    print("\nBaseline D6 completed")

    improvements = []
    method_names = []
    all_scores_vectors = []
    all_scores_wordcount = []
    all_scores_poscount = []
    
    print(f"\n{'='*50}")
    print("[Methods 1-10] Rewrite from the baseline D6")
    print(f"{'='*50}")
    
    is_single = (m11_mode == 'single')
    
    for meth_name in GEO_METHODS:
        if meth_name == 'identity':
            ad_score = baseline_score
            final_scores = baseline_scores_vector.copy()
            final_scores_wc = baseline_scores_wordcount.copy()
            final_scores_pc = baseline_scores_poscount.copy()
        else:
            d6_opt = GEO_METHODS[meth_name](baseline_d6)
            ad_score, final_scores, final_scores_wc, final_scores_pc = run_geo_with_d6(
                query, summaries_5, d6_opt, skip_quality_eval=is_single
            )
        
        improvement = ad_score - baseline_score
        improvements.append(improvement)
        method_names.append(meth_name)
        all_scores_vectors.append(final_scores)
        all_scores_wordcount.append(final_scores_wc)
        all_scores_poscount.append(final_scores_pc)

        print_method_result(meth_name, final_scores, improvement, baseline_content_quality, single_objective=is_single)
    
    if enable_method_11:
        print(f"\n{'='*50}")
        if m11_mode == 'pareto':
            print("[Method 11] Pareto multi-objective GA optimization (ad visibility + content quality)")
        else:
            print("[Method 11] Single-objective GA optimization (maximize ad visibility only)")
        print(f"{'='*50}")
        
        if m11_mode == 'pareto':
            ga_result, new_evaluations = run_ga_multi_objective(
                query, summaries_5,
                popsize=len(initial_population) if initial_population else 10,
                generations=m11_generations,
                p_mut=m11_pmut,
                print_progress=True,
                initial_population=initial_population,
                initial_fitness=initial_scores,  # Ignored in Pareto mode because objectives are recomputed.
                initial_scores_vectors=initial_scores_vectors,
                initial_scores_vectors_wc=initial_scores_vectors_wc,
                initial_scores_vectors_pc=initial_scores_vectors_pc,
                initial_d6_pool=initial_d6_pool,
                ad_theme=ad_theme,  # Reuse the already extracted ad theme.
                d6_cache_records=ga_eval_cache if enable_ga_cache else None  # Use the GA evaluation cache when enabled.
            )
            
            if enable_ga_cache and new_evaluations:
                ga_eval_cache.extend(new_evaluations)
                print(f"   [Resume] GA cache: added {len(new_evaluations)} new records")
            
            if ga_result.pareto_front:
                print(f"  [Pareto Front] {len(ga_result.pareto_front)} non-dominated solutions found")
        else:
            ga_result = run_ga_single_scene(
                query, summaries_5,
                generations=m11_generations,
                p_mut=m11_pmut,
                print_progress=True,
                initial_population=initial_population,
                initial_fitness=initial_scores,
                initial_scores_vectors=initial_scores_vectors,
                initial_d6_pool=initial_d6_pool,
                ad_theme=ad_theme  # Reuse the already extracted ad theme.
            )
        
        best_scores_vector = ga_result.best_scores_vector
        best_d6_score = ga_result.best_fitness
        
        best_scores_wc = getattr(ga_result, 'best_scores_wc', None)
        best_scores_pc = getattr(ga_result, 'best_scores_pc', None)
        
        if (best_scores_wc is None or best_scores_pc is None) and ga_result.best_d6:
            _, _, best_scores_wc, best_scores_pc = run_geo_with_d6(
                query, summaries_5, ga_result.best_d6, print_responses=False
            )
        elif best_scores_wc is None or best_scores_pc is None:
            best_scores_wc = best_scores_wc if best_scores_wc is not None else best_scores_vector
            best_scores_pc = best_scores_pc if best_scores_pc is not None else best_scores_vector
        
        improvement_11 = best_d6_score - baseline_score
        improvements.append(improvement_11)
        method_names.append('DynamicGAOptimization')
        all_scores_vectors.append(best_scores_vector)
        all_scores_wordcount.append(best_scores_wc)
        all_scores_poscount.append(best_scores_pc)
        
        from featgeo.geo_ad.multi_objective import calculate_objectives
        _, best_content_quality = calculate_objectives(best_scores_vector)
        print("DynamicGAOptimization completed")
        
        if hasattr(ga_result, 'highest_ad_scores_vector') and ga_result.highest_ad_scores_vector is not None:
            ga_highest_ad_scores = ga_result.highest_ad_scores_vector
            ga_highest_ad_config = ga_result.highest_ad_config if hasattr(ga_result, 'highest_ad_config') else None
            ga_highest_scores_wc = ga_result.highest_ad_scores_wc if hasattr(ga_result, 'highest_ad_scores_wc') else ga_highest_ad_scores
            ga_highest_scores_pc = ga_result.highest_ad_scores_pc if hasattr(ga_result, 'highest_ad_scores_pc') else ga_highest_ad_scores
        else:
            ga_highest_ad_scores = None
            ga_highest_ad_config = None
            ga_highest_scores_wc = None
            ga_highest_scores_pc = None
    else:
        print("\n[Method 11] Disabled (set ENABLE_METHOD_11=True in config.py to enable it)")
        ga_highest_ad_scores = None
        ga_highest_ad_config = None
        ga_highest_scores_wc = None
        ga_highest_scores_pc = None

    return improvements, method_names, all_scores_vectors, ga_highest_ad_scores, all_scores_wordcount, all_scores_poscount, ga_highest_scores_wc, ga_highest_scores_pc, ga_highest_ad_config


def main():
    args = parse_args()
    if args.api_key:
        config.OPENAI_API_KEYS = [args.api_key]

    print_responses = getattr(config, 'PRINT_RESPONSES', False)
    sample_limit = getattr(config, 'SAMPLE_LIMIT', 3)
    use_gpt_summary = getattr(config, 'USE_GPT_SUMMARY', False)
    
    sampling_mode_env = args.sampling_mode or (getattr(config, 'SAMPLING_MODE', 'continuous') if config else 'continuous')
    
    dataset = load_dataset("GEO-Optim/geo-bench", 'test')
    test_split = dataset['test']
    dataset_size = len(test_split)
    
    if sampling_mode_env == 'all':
        random_indices = None
        start_index = args.start_index if args.start_index is not None else 0
        end_index = args.end_index if args.end_index is not None else dataset_size
    elif sampling_mode_env == 'random':
        if args.random_indices is not None:
            random_indices = json.loads(args.random_indices)
        else:
            random_sample_size = getattr(config, 'RANDOM_SAMPLE_SIZE', 50) if config else 50
            random_sampling_seed = getattr(config, 'RANDOM_SAMPLING_SEED', None) if config else None
            
            import random as py_random
            if random_sampling_seed is not None:
                py_random.seed(random_sampling_seed)
            random_indices = sorted(py_random.sample(range(dataset_size), min(random_sample_size, dataset_size)))
            print(f"[Random] [Random Sampling] seed={random_sampling_seed}, selected {random_sample_size} samples")
        
        start_index = 0
        end_index = dataset_size  # Full GEO-bench test size so all sampled indices stay addressable.
    else:
        random_indices = None
        start_index = args.start_index if args.start_index is not None else 0
        end_index = args.end_index if args.end_index is not None else sample_limit
    
    random_seed = getattr(config, 'RANDOM_SEED', None) if config else None
    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        print(f"[Random] Random seed set to {random_seed} (results are reproducible)")
    
    enable_method_11 = getattr(config, 'ENABLE_METHOD_11', False)
    m11_popsize = getattr(config, 'METHOD_11_POPSIZE', 10)
    m11_generations = getattr(config, 'METHOD_11_GENERATIONS', 3)
    m11_pmut = getattr(config, 'METHOD_11_PMUT', 0.15)
    m11_mode = getattr(config, 'METHOD_11_MODE', 'single').lower()  # 'single' or 'pareto'
    enable_ga_cache = getattr(config, 'ENABLE_GA_CACHE', True) if config else True  # GA evaluation cache.
    enable_query_probing = getattr(config, 'ENABLE_QUERY_PROBING', False)
    query_probe_count = getattr(config, 'QUERY_PROBE_COUNT', 5)
    query_probe_top_n = getattr(config, 'QUERY_PROBE_TOP_N', 3)
    query_probe_include_original = getattr(config, 'QUERY_PROBE_INCLUDE_ORIGINAL', True)
    query_probe_num_completions = getattr(config, 'QUERY_PROBE_NUM_COMPLETIONS', 1)
    query_probe_model = config.QUERY_PROBE_MODEL
    experiment_tag = "default_main"
    if enable_query_probing:
        original_flag = "with_original" if query_probe_include_original else "without_original"
        model_tag = str(query_probe_model).replace("/", "_").replace(":", "_")
        experiment_tag = (
            f"query_probe:v1:{original_flag}:"
            f"model{model_tag}:count{query_probe_count}:"
            f"top{query_probe_top_n}:ncomp{query_probe_num_completions}"
        )

    # Each shard writes to its own cache files to avoid concurrent write conflicts.
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Optional shard suffix injected by run_parallel.py.
    # Examples:
    #   - random mode: _random_s42_shard1
    #   - continuous/sparse mode: _shard_0_100
    shard_suffix = args.cache_suffix
    if not shard_suffix and (sampling_mode_env == 'all' or start_index > 0 or end_index != sample_limit):
        # Backward-compatible fallback for older range-based runs.
        shard_suffix = f"_all_{start_index}_{end_index}" if sampling_mode_env == 'all' else f"_shard_{start_index}_{end_index}"
    
    ad_geo_cache_base = os.path.join(data_dir, f'ad_geo_optimizations_cache{shard_suffix}.json')
    
    from featgeo import geo_functions, utils
    geo_functions.set_geo_cache_file(ad_geo_cache_base)
    if args.global_cache_file:
        utils.set_cache_file(args.global_cache_file)
    
    d6_pop_log_path = os.path.join(data_dir, f'ad_d6_population_log{shard_suffix}.json')
    
    if os.path.exists(d6_pop_log_path):
        try:
            with open(d6_pop_log_path, 'r', encoding='utf-8') as f:
                d6_pop_records = json.load(f)
        except:
            d6_pop_records = []
    else:
        d6_pop_records = []
    
    ga_eval_cache_path = os.path.join(data_dir, f'ad_ga_evaluation_cache{shard_suffix}.json')
    
    if enable_ga_cache and os.path.exists(ga_eval_cache_path):
        try:
            with open(ga_eval_cache_path, 'r', encoding='utf-8') as f:
                ga_eval_cache = json.load(f)
            print(f"[Shard] Loaded GA evaluation cache: {len(ga_eval_cache)} records")
        except:
            ga_eval_cache = []
            print("[Warning] GA evaluation cache is corrupted. Starting fresh.")
    else:
        ga_eval_cache = []
        if not enable_ga_cache:
            print("[Info] GA evaluation cache is disabled")

    method_count = 11 if enable_method_11 else 10
    print(f"\n{'='*60}")
    print("FeatGEO Ad Optimization Evaluation")
    if sampling_mode_env == 'random':
        print(f"Sampling mode: random | total samples: {len(random_indices)}")
        print(f"Sample indices: {random_indices[:10]}{'...' if len(random_indices) > 10 else ''}")
    elif sampling_mode_env == 'all':
        print(f"Sampling mode: all | data range: [{start_index}, {end_index}) | total samples: {end_index - start_index}")
    else:
        print(f"Data range: [{start_index}, {end_index}) | total samples: {end_index - start_index}")
    print(f"USE_GPT_SUMMARY={use_gpt_summary}")
    print(f"Method 11 (Dynamic GA): {'enabled' if enable_method_11 else 'disabled'}")
    if enable_method_11:
        mode_desc = 'Pareto multi-objective optimization (ad + quality)' if m11_mode == 'pareto' else 'Single-objective optimization (ad only)'
        print(f"  Optimization mode: {mode_desc}")
        print(f"  GA parameters: popsize={m11_popsize}, generations={m11_generations}, p_mut={m11_pmut}")
    print(f"Query probing: {'enabled' if enable_query_probing else 'disabled'}")
    if enable_query_probing:
        print(
            f"  Probe parameters: count={query_probe_count}, top_n={query_probe_top_n}, "
            f"include_original={query_probe_include_original}, completions={query_probe_num_completions}"
        )
    print(f"Running {method_count} optimization methods")
    if shard_suffix:
        print(f"[Shard] Shard mode: cache suffix = {shard_suffix}")
    print(f"{'='*60}\n")
    
    all_query_results = []  # Each item stores baseline and per-method score vectors for one query.
    
    # Resume support: skip queries that were already fully processed.
    processed_queries = set()
    if shard_suffix:
        query_results_path = os.path.join(data_dir, f'query_results{shard_suffix}.json')
        if os.path.exists(query_results_path):
            try:
                with open(query_results_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                
                valid_results = []
                invalid_count = 0
                for result in existing_results:
                    if not isinstance(result, dict) or 'query' not in result or 'baseline' not in result:
                        invalid_count += 1
                        continue

                    result_tag = result.get('experiment_tag', 'default_main')
                    if result_tag != experiment_tag:
                        invalid_count += 1
                        print(f"   [Resume] Skipping result from a different experiment tag: {result_tag}")
                        continue
                    
                    if 'methods' not in result or len(result.get('methods', {})) < 10:
                        invalid_count += 1
                        print(f"   [Warning] Skipping incomplete result: {result.get('query', 'unknown')[:50]}...")
                        continue
                    
                    valid_results.append(result)
                
                if invalid_count > 0:
                    print(f"[Warning] [Data Repair] Found {invalid_count} incomplete results and filtered them automatically")
                
                for result in valid_results:
                    result['baseline'] = np.array(result['baseline'])
                    for method in result['methods']:
                        result['methods'][method] = np.array(result['methods'][method])
                    for method in result.get('methods_wordcount', {}):
                        result['methods_wordcount'][method] = np.array(result['methods_wordcount'][method])
                    for method in result.get('methods_poscount', {}):
                        result['methods_poscount'][method] = np.array(result['methods_poscount'][method])
                    if result.get('ga_highest_ad') is not None:
                        result['ga_highest_ad'] = np.array(result['ga_highest_ad'])
                    if result.get('ga_highest_ad_wc') is not None:
                        result['ga_highest_ad_wc'] = np.array(result['ga_highest_ad_wc'])
                    if result.get('ga_highest_ad_pc') is not None:
                        result['ga_highest_ad_pc'] = np.array(result['ga_highest_ad_pc'])
                
                all_query_results = valid_results  # Only complete results are resumed.
                processed_queries = {r['query'] for r in valid_results}  # Track processed queries.
                print(f"[Resume] [Resume] Found {len(valid_results)} complete results. These samples will be skipped.")
                print(f"   Processed queries: {list(processed_queries)[:3]}{'...' if len(processed_queries) > 3 else ''}")
            except json.JSONDecodeError as e:
                print(f"[Warning] [Resume] JSON file is corrupted: {e}")
                print("   The file may have been interrupted during writing. It will be backed up and all samples will be reprocessed.")
                # Back up the corrupted file before restarting.
                import shutil
                backup_path = query_results_path + '.corrupted'
                shutil.copy(query_results_path, backup_path)
                print(f"   Corrupted file backed up to: {backup_path}")
                all_query_results = []
                processed_queries = set()
            except Exception as e:
                print(f"[Warning] [Resume] Failed to load existing results: {e}. All samples will be reprocessed.")
                all_query_results = []
                processed_queries = set()
    
    for i, k in enumerate(test_split):
        if sampling_mode_env == 'random':
            if i not in random_indices:
                continue
            sample_position = random_indices.index(i) + 1
        else:
            if i < start_index:
                continue  # Skip samples before this shard's start.
            if i >= end_index:
                print(f"Processed all samples in range [{start_index}, {end_index}).")
                break
            sample_position = i - start_index + 1
        
        if k['query'] in processed_queries:
            print(f"\n{'='*60}\nSample {i+1}/{len(test_split)} (position {sample_position} in this shard) | [Skip] Already processed, skipping")
            print(f"Query: {k['query']}")
            continue
        
        print(f"\n{'='*60}\nSample {i+1}/{len(test_split)} (position {sample_position} in this shard) | Query: {k['query']}")

        sources, summaries_5 = extract_sources_from_dataset(k['sources'], use_gpt_summary=use_gpt_summary)
        
        feature_summaries = summaries_5
        feature_source_tag = "default_d1d5"
        query_probe_info = None

        if enable_query_probing:
            print("\n[Query Probe] Selecting topic-consistent feature exemplars from D1-D5...")
            feature_summaries, query_probe_info = select_probe_feature_summaries(
                query=k['query'],
                summaries=summaries_5,
                top_n=query_probe_top_n,
                probe_count=query_probe_count,
                include_original=query_probe_include_original,
                num_completions=query_probe_num_completions,
                model=query_probe_model,
                print_info=True,
            )
            feature_source_tag = query_probe_info.get('cache_tag', 'query_probe:unknown')

        print("\n[Feature Extraction] Inferring feature configs from selected D1-D5 exemplars...")
        cfg_from_d1d5 = [infer_config_from_text(summary) for summary in feature_summaries]
        print(f"  Extracted feature configs for {len(cfg_from_d1d5)} webpages")
        
        skip_quality = (m11_mode == 'single')
        (initial_population, initial_scores, initial_scores_vectors,
         initial_scores_vectors_wc, initial_scores_vectors_pc,
         initial_d6_pool, cache_hit_count, ad_theme) = load_or_generate_d6_population(
            query=k['query'],
            summaries_5=summaries_5,
            cfg_from_d1d5=cfg_from_d1d5,
            popsize=m11_popsize,
            d6_pop_records=d6_pop_records,
            print_info=True,
            skip_quality_eval=skip_quality,
            feature_source_tag=feature_source_tag
        )
        
        # Append newly generated D6 records to the population log without duplicating cached entries.
        existing_keys = {(r.get('query'), r.get('sample_index'), r.get('individual_index'), r.get('feature_source_tag', 'default_d1d5')) 
                        for r in d6_pop_records}
        
        for idx in range(len(initial_population)):
            key = (k['query'], i, idx, feature_source_tag)
            if key not in existing_keys:
                try:
                    from featgeo.geo_ad.multi_objective import calculate_objectives, calculate_weighted_score
                    scores_vec = initial_scores_vectors[idx]
                    ad_vis, content_qual = calculate_objectives(scores_vec)
                    weighted_score = calculate_weighted_score(ad_vis, content_qual)
                    
                    log_record = {
                        "sample_index": i,
                        "query": k['query'],
                        "feature_source_tag": feature_source_tag,
                        "query_probe": query_probe_info,
                        "ad_theme": ad_theme,
                        "individual_index": idx,
                        "score": float(initial_scores[idx]),
                        "scores_vector": scores_vec.tolist(),
                        "content_quality": float(content_qual),
                        "weighted_score": float(weighted_score),
                        "cfg": initial_population[idx].__dict__,
                        "d6_text": initial_d6_pool[idx],
                    }
                    d6_pop_records.append(log_record)
                    
                    # Flush after each append so resume data is always up to date.
                    with open(d6_pop_log_path, 'w', encoding='utf-8') as f_log:
                        json.dump(d6_pop_records, f_log, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"  [Log Warning] Failed to write the D6 log: {e}")
        
        (baseline_d6, baseline_score, baseline_scores_vector,
         baseline_scores_wordcount, baseline_scores_poscount,
         baseline_ad_vis, baseline_content_qual) = select_baseline_d6(
            initial_population=initial_population,
            initial_scores=initial_scores,
            initial_scores_vectors=initial_scores_vectors,
            initial_scores_vectors_wc=initial_scores_vectors_wc,
            initial_scores_vectors_pc=initial_scores_vectors_pc,
            initial_d6_pool=initial_d6_pool,
            print_info=True,
            single_objective=skip_quality
        )
        
        if cache_hit_count > 0:
            print(f"   [Run] Cache hit rate: {cache_hit_count}/{m11_popsize} ({cache_hit_count/m11_popsize*100:.1f}%)")

        improvements, method_names, all_scores_vectors, ga_highest_ad_scores, all_scores_wordcount, all_scores_poscount, ga_highest_scores_wc, ga_highest_scores_pc, ga_highest_ad_config = apply_geo_methods_to_ad(
            k['query'], summaries_5,
            baseline_d6=baseline_d6,
            baseline_score=baseline_score,
            baseline_scores_vector=baseline_scores_vector,
            baseline_scores_wordcount=baseline_scores_wordcount,
            baseline_scores_poscount=baseline_scores_poscount,
            initial_population=initial_population,
            initial_scores=initial_scores,
            initial_scores_vectors=initial_scores_vectors,
            initial_scores_vectors_wc=initial_scores_vectors_wc,
            initial_scores_vectors_pc=initial_scores_vectors_pc,
            initial_d6_pool=initial_d6_pool,
            ad_theme=ad_theme,
            enable_method_11=enable_method_11,
            m11_generations=m11_generations,
            m11_pmut=m11_pmut,
            m11_mode=m11_mode,
            print_responses=print_responses,
            enable_ga_cache=enable_ga_cache,
            ga_eval_cache=ga_eval_cache
        )
        
        is_single_mode = (m11_mode == 'single')
        print_summary_table(improvements, method_names, all_scores_vectors, baseline_scores_vector,
                           single_objective=is_single_mode)
        
        print_visibility_matrix(method_names, all_scores_vectors, baseline_scores_vector, 
                               ga_highest_ad_scores=ga_highest_ad_scores,
                               all_scores_wordcount=all_scores_wordcount,
                               all_scores_poscount=all_scores_poscount,
                               ga_highest_scores_wc=ga_highest_scores_wc,
                               ga_highest_scores_pc=ga_highest_scores_pc,
                               baseline_scores_wordcount=baseline_scores_wordcount,
                               baseline_scores_poscount=baseline_scores_poscount,
                               single_objective=is_single_mode)
        
        query_result = {
            'query': k['query'],
            'experiment_tag': experiment_tag,
            'feature_source_tag': feature_source_tag,
            'query_probe': query_probe_info,
            'baseline': baseline_scores_vector.copy(),
            'baseline_wordcount': baseline_scores_wordcount.copy(),
            'baseline_poscount': baseline_scores_poscount.copy(),
            'methods': {},
            'methods_wordcount': {},
            'methods_poscount': {},
            'ga_highest_ad': ga_highest_ad_scores.copy() if ga_highest_ad_scores is not None else None,
            'ga_highest_ad_wc': ga_highest_scores_wc.copy() if ga_highest_scores_wc is not None else None,
            'ga_highest_ad_pc': ga_highest_scores_pc.copy() if ga_highest_scores_pc is not None else None,
            'ga_highest_ad_config': ga_highest_ad_config.__dict__ if ga_highest_ad_config is not None else None
        }
        for meth_name, scores_vec, scores_wc, scores_pc in zip(method_names, all_scores_vectors, all_scores_wordcount, all_scores_poscount):
            query_result['methods'][meth_name] = scores_vec.copy()
            query_result['methods_wordcount'][meth_name] = scores_wc.copy()
            query_result['methods_poscount'][meth_name] = scores_pc.copy()
        all_query_results.append(query_result)
        
        # Save after every sample so interrupted runs can resume cleanly.
        try:
            with open(d6_pop_log_path, 'w', encoding='utf-8') as f_log:
                json.dump(d6_pop_records, f_log, ensure_ascii=False, indent=2)
            print(f"[Log] Saved initial-population D6 records ({len(d6_pop_records)} total)")
        except Exception as e:
            print(f"[Log Warning] Failed to write the initial-population D6 JSON log: {e}")
        
        if enable_ga_cache:
            try:
                with open(ga_eval_cache_path, 'w', encoding='utf-8') as f_cache:
                    json.dump(ga_eval_cache, f_cache, ensure_ascii=False, indent=2)
                print(f"[Log] Saved GA evaluation cache ({len(ga_eval_cache)} total)")
            except Exception as e:
                print(f"[Log Warning] Failed to write the GA evaluation cache: {e}")
        
        # Incrementally save query results for resume support.
        if shard_suffix:
            query_results_path = os.path.join(data_dir, f'query_results{shard_suffix}.json')
            try:
                serializable_results = []
                for result in all_query_results:
                    serializable_result = {
                        'query': result['query'],
                        'experiment_tag': result.get('experiment_tag', experiment_tag),
                        'feature_source_tag': result.get('feature_source_tag', 'default_d1d5'),
                        'query_probe': result.get('query_probe'),
                        'baseline': result['baseline'].tolist() if isinstance(result['baseline'], np.ndarray) else result['baseline'],
                        'baseline_wordcount': result['baseline_wordcount'].tolist() if isinstance(result['baseline_wordcount'], np.ndarray) else result['baseline_wordcount'],
                        'baseline_poscount': result['baseline_poscount'].tolist() if isinstance(result['baseline_poscount'], np.ndarray) else result['baseline_poscount'],
                        'methods': {},
                        'methods_wordcount': {},
                        'methods_poscount': {},
                        'ga_highest_ad': result['ga_highest_ad'].tolist() if result.get('ga_highest_ad') is not None and isinstance(result['ga_highest_ad'], np.ndarray) else result.get('ga_highest_ad'),
                        'ga_highest_ad_wc': result['ga_highest_ad_wc'].tolist() if result.get('ga_highest_ad_wc') is not None and isinstance(result['ga_highest_ad_wc'], np.ndarray) else result.get('ga_highest_ad_wc'),
                        'ga_highest_ad_pc': result['ga_highest_ad_pc'].tolist() if result.get('ga_highest_ad_pc') is not None and isinstance(result['ga_highest_ad_pc'], np.ndarray) else result.get('ga_highest_ad_pc'),
                        'ga_highest_ad_config': result.get('ga_highest_ad_config')
                    }
                    for method, scores in result['methods'].items():
                        serializable_result['methods'][method] = scores.tolist() if isinstance(scores, np.ndarray) else scores
                    for method, scores in result.get('methods_wordcount', {}).items():
                        serializable_result['methods_wordcount'][method] = scores.tolist() if isinstance(scores, np.ndarray) else scores
                    for method, scores in result.get('methods_poscount', {}).items():
                        serializable_result['methods_poscount'][method] = scores.tolist() if isinstance(scores, np.ndarray) else scores
                    serializable_results.append(serializable_result)
                
                with open(query_results_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, ensure_ascii=False, indent=2)
                print(f"[Log] Incrementally saved query results ({len(serializable_results)} currently)")
            except Exception as e:
                print(f"[Log Warning] Failed to incrementally save query results: {e}")

    try:
        with open(d6_pop_log_path, 'w', encoding='utf-8') as f_log:
            json.dump(d6_pop_records, f_log, ensure_ascii=False, indent=2)
        print(f"\n[Log] Final save of initial-population D6 records: {d6_pop_log_path} ({len(d6_pop_records)} total)")
    except Exception as e:
        print(f"\n[Log Warning] Final write of the initial-population D6 JSON log failed: {e}")
    
    if enable_ga_cache:
        try:
            with open(ga_eval_cache_path, 'w', encoding='utf-8') as f_cache:
                json.dump(ga_eval_cache, f_cache, ensure_ascii=False, indent=2)
            print(f"[Log] Final save of GA evaluation cache: {ga_eval_cache_path} ({len(ga_eval_cache)} total)")
        except Exception as e:
            print(f"[Log Warning] Final write of the GA evaluation cache failed: {e}")
    
    # Final save of query results for shard merging.
    if shard_suffix:
        query_results_path = os.path.join(data_dir, f'query_results{shard_suffix}.json')
        try:
            serializable_results = []
            for result in all_query_results:
                serializable_result = {
                    'query': result['query'],
                    'experiment_tag': result.get('experiment_tag', experiment_tag),
                    'feature_source_tag': result.get('feature_source_tag', 'default_d1d5'),
                    'query_probe': result.get('query_probe'),
                    'baseline': result['baseline'].tolist() if isinstance(result['baseline'], np.ndarray) else result['baseline'],
                    'baseline_wordcount': result['baseline_wordcount'].tolist() if isinstance(result['baseline_wordcount'], np.ndarray) else result['baseline_wordcount'],
                    'baseline_poscount': result['baseline_poscount'].tolist() if isinstance(result['baseline_poscount'], np.ndarray) else result['baseline_poscount'],
                    'methods': {},
                    'methods_wordcount': {},
                    'methods_poscount': {},
                    'ga_highest_ad': result['ga_highest_ad'].tolist() if result.get('ga_highest_ad') is not None and isinstance(result['ga_highest_ad'], np.ndarray) else result.get('ga_highest_ad'),
                    'ga_highest_ad_wc': result['ga_highest_ad_wc'].tolist() if result.get('ga_highest_ad_wc') is not None and isinstance(result['ga_highest_ad_wc'], np.ndarray) else result.get('ga_highest_ad_wc'),
                    'ga_highest_ad_pc': result['ga_highest_ad_pc'].tolist() if result.get('ga_highest_ad_pc') is not None and isinstance(result['ga_highest_ad_pc'], np.ndarray) else result.get('ga_highest_ad_pc'),
                    'ga_highest_ad_config': result.get('ga_highest_ad_config')
                }
                for method, scores in result['methods'].items():
                    serializable_result['methods'][method] = scores.tolist() if isinstance(scores, np.ndarray) else scores
                # Include the auxiliary wordcount and poscount metrics.
                for method, scores in result.get('methods_wordcount', {}).items():
                    serializable_result['methods_wordcount'][method] = scores.tolist() if isinstance(scores, np.ndarray) else scores
                for method, scores in result.get('methods_poscount', {}).items():
                    serializable_result['methods_poscount'][method] = scores.tolist() if isinstance(scores, np.ndarray) else scores
                serializable_results.append(serializable_result)
            
            with open(query_results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            print(f"[Log] Final query-results save confirmed: {query_results_path} ({len(serializable_results)} total)")
        except Exception as e:
            print(f"[Log Warning] Final save of query results failed: {e}")
    
    print_average_matrix(all_query_results, single_objective=(m11_mode == 'single'))


if __name__ == '__main__':
    main()

