import numpy as np
from typing import List, Tuple
from featgeo.geo_ad.d6_generator import generate_d6, extract_ad_topic
from featgeo.geo_ad.feature_schema import FeatureConfig, random_config, sample_config_from_d1d5
from featgeo.geo_ad.geo_runner_wrapper import run_geo_with_d6
from featgeo.geo_ad.multi_objective import calculate_objectives, calculate_weighted_score


def get_cached_d6_for_query(query: str, d6_pop_records: list, feature_source_tag: str = "default_d1d5") -> list:
    """Return population records that belong to the query and feature source setup."""
    if feature_source_tag == "default_d1d5":
        return [
            r for r in d6_pop_records
            if r.get('query') == query and r.get('feature_source_tag', 'default_d1d5') == feature_source_tag
        ]

    return [
        r for r in d6_pop_records
        if r.get('query') == query and r.get('feature_source_tag') == feature_source_tag
    ]


def get_or_extract_ad_theme(query: str, summaries_5: List[str], 
                           d6_pop_records: list, print_info: bool = True,
                           feature_source_tag: str = "default_d1d5") -> str:
    """Reuse a cached ad theme when available, otherwise extract one."""
    cached_records = get_cached_d6_for_query(query, d6_pop_records, feature_source_tag)
    if cached_records:
        ad_theme = cached_records[0].get('ad_theme')
        if ad_theme:
            if print_info:
                print("  Ad theme loaded from cache")
            return ad_theme
    
    if print_info:
        print("  Extracting ad theme...")
    
    ad_theme = extract_ad_topic(summaries_5, temperature=0.3, verbose=False)
    
    if print_info:
        print("  Ad theme extraction completed")
    
    return ad_theme


def print_d6_individual(idx: int, total: int, scores_vec: np.ndarray,
                       score: float, tag: str = "", single_objective: bool = False) -> None:
    """Print one compact initial-population progress line."""
    print(f"  Candidate {idx}/{total} completed")


def load_or_generate_d6_population(
    query: str,
    summaries_5: list,
    cfg_from_d1d5: list,
    popsize: int,
    d6_pop_records: list,
    print_info: bool = True,
    skip_quality_eval: bool = False,
    feature_source_tag: str = "default_d1d5"
) -> Tuple[List[FeatureConfig], List[float], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str], int, str]:
    """Load or build the shared initial D6 population for one query."""
    if print_info:
        print("\nInitial D6 population")
        print("  Checking cache...")

    cached_d6_records = get_cached_d6_for_query(query, d6_pop_records, feature_source_tag)

    ad_theme = get_or_extract_ad_theme(query, summaries_5, d6_pop_records, print_info, feature_source_tag)

    initial_population = []
    initial_scores = []
    initial_scores_vectors = []
    initial_scores_vectors_wc = []
    initial_scores_vectors_pc = []
    initial_d6_pool = []
    cache_hit_count = 0

    if len(cached_d6_records) >= popsize:
        if print_info:
            print(f"  [OK] Sufficient cache. Found {len(cached_d6_records)} records and will reuse the top {popsize}.")

        sorted_records = sorted(
            cached_d6_records,
            key=lambda x: x.get('weighted_score', x.get('score', 0)),
            reverse=True
        )
        selected_records = sorted_records[:popsize]
        cache_hit_count = popsize

        for idx, record in enumerate(selected_records):
            cfg_dict = record.get('cfg', {})
            cfg = FeatureConfig(**cfg_dict)
            initial_population.append(cfg)

            d6_text = record.get('d6_text', '')
            initial_d6_pool.append(d6_text)

            score = record.get('score', 0)
            initial_scores.append(score)

            scores_vector = record.get('scores_vector', None)
            scores_vector_wc = record.get('scores_vector_wc', None)
            scores_vector_pc = record.get('scores_vector_pc', None)

            if scores_vector is not None:
                initial_scores_vectors.append(np.array(scores_vector))
                if scores_vector_wc is None or scores_vector_pc is None:
                    _, _, scores_wc, scores_pc = run_geo_with_d6(query, summaries_5, d6_text, print_responses=False, skip_quality_eval=skip_quality_eval)
                    initial_scores_vectors_wc.append(scores_wc)
                    initial_scores_vectors_pc.append(scores_pc)
                    if print_info:
                        print_d6_individual(idx+1, popsize, np.array(scores_vector), score, "cache + wc/pc fill", single_objective=skip_quality_eval)
                else:
                    initial_scores_vectors_wc.append(np.array(scores_vector_wc))
                    initial_scores_vectors_pc.append(np.array(scores_vector_pc))
                    if print_info:
                        print_d6_individual(idx+1, popsize, np.array(scores_vector), score, "cache", single_objective=skip_quality_eval)
            else:
                _, scores_vec, scores_wc, scores_pc = run_geo_with_d6(query, summaries_5, d6_text, print_responses=False, skip_quality_eval=skip_quality_eval)
                initial_scores_vectors.append(scores_vec)
                initial_scores_vectors_wc.append(scores_wc)
                initial_scores_vectors_pc.append(scores_pc)
                if print_info:
                    print_d6_individual(idx+1, popsize, scores_vec, score, "cache backfilled", single_objective=skip_quality_eval)

    elif len(cached_d6_records) > 0:
        if print_info:
            print(f"  [Warning] Partial cache. Found {len(cached_d6_records)} records and need {popsize - len(cached_d6_records)} more.")

        cache_hit_count = len(cached_d6_records)

        for idx, record in enumerate(cached_d6_records):
            cfg_dict = record.get('cfg', {})
            cfg = FeatureConfig(**cfg_dict)
            initial_population.append(cfg)

            d6_text = record.get('d6_text', '')
            initial_d6_pool.append(d6_text)

            score = record.get('score', 0)
            initial_scores.append(score)

            scores_vector = record.get('scores_vector', None)
            scores_vector_wc = record.get('scores_vector_wc', None)
            scores_vector_pc = record.get('scores_vector_pc', None)

            if scores_vector is not None:
                initial_scores_vectors.append(np.array(scores_vector))
                if scores_vector_wc is None or scores_vector_pc is None:
                    _, _, scores_wc, scores_pc = run_geo_with_d6(query, summaries_5, d6_text, print_responses=False, skip_quality_eval=skip_quality_eval)
                    initial_scores_vectors_wc.append(scores_wc)
                    initial_scores_vectors_pc.append(scores_pc)
                    if print_info:
                        print_d6_individual(idx+1, popsize, np.array(scores_vector), score, "cache + wc/pc fill", single_objective=skip_quality_eval)
                else:
                    initial_scores_vectors_wc.append(np.array(scores_vector_wc))
                    initial_scores_vectors_pc.append(np.array(scores_vector_pc))
                    if print_info:
                        print_d6_individual(idx+1, popsize, np.array(scores_vector), score, "cache", single_objective=skip_quality_eval)
            else:
                _, scores_vec, scores_wc, scores_pc = run_geo_with_d6(query, summaries_5, d6_text, print_responses=False, skip_quality_eval=skip_quality_eval)
                initial_scores_vectors.append(scores_vec)
                initial_scores_vectors_wc.append(scores_wc)
                initial_scores_vectors_pc.append(scores_pc)
                if print_info:
                    print_d6_individual(idx+1, popsize, scores_vec, score, "cache backfilled", single_objective=skip_quality_eval)

        num_to_generate = popsize - len(cached_d6_records)
        if print_info:
            print(f"  Generating {num_to_generate} new D6 candidates...")

        for new_idx in range(num_to_generate):
            if cfg_from_d1d5:
                cfg = sample_config_from_d1d5(cfg_from_d1d5)
            else:
                cfg = random_config()
            d6 = generate_d6(summaries_5, cfg, ad_theme=ad_theme, verbose=False)
            score, scores_vec, scores_wc, scores_pc = run_geo_with_d6(query, summaries_5, d6, print_responses=False, skip_quality_eval=skip_quality_eval)

            initial_population.append(cfg)
            initial_d6_pool.append(d6)
            initial_scores.append(score)
            initial_scores_vectors.append(scores_vec)
            initial_scores_vectors_wc.append(scores_wc)
            initial_scores_vectors_pc.append(scores_pc)

            idx = len(cached_d6_records) + new_idx + 1
            if print_info:
                print_d6_individual(idx, popsize, scores_vec, score, "new", single_objective=skip_quality_eval)

    else:
        if print_info:
            print(f"  Generating {popsize} initial D6 candidates...")

        for idx in range(popsize):
            if cfg_from_d1d5:
                cfg = sample_config_from_d1d5(cfg_from_d1d5)
            else:
                cfg = random_config()
            d6 = generate_d6(summaries_5, cfg, ad_theme=ad_theme, verbose=False)
            score, scores_vec, scores_wc, scores_pc = run_geo_with_d6(query, summaries_5, d6, print_responses=False, skip_quality_eval=skip_quality_eval)

            initial_population.append(cfg)
            initial_d6_pool.append(d6)
            initial_scores.append(score)
            initial_scores_vectors.append(scores_vec)
            initial_scores_vectors_wc.append(scores_wc)
            initial_scores_vectors_pc.append(scores_pc)

            if print_info:
                print_d6_individual(idx+1, popsize, scores_vec, score, "", single_objective=skip_quality_eval)

    return (initial_population, initial_scores, initial_scores_vectors,
            initial_scores_vectors_wc, initial_scores_vectors_pc,
            initial_d6_pool, cache_hit_count, ad_theme)


def select_baseline_d6(
    initial_population: List[FeatureConfig],
    initial_scores: List[float],
    initial_scores_vectors: List[np.ndarray],
    initial_scores_vectors_wc: List[np.ndarray],
    initial_scores_vectors_pc: List[np.ndarray],
    initial_d6_pool: List[str],
    print_info: bool = True,
    single_objective: bool = False
) -> Tuple[str, float, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Select the baseline D6 candidate from the initial population."""
    if single_objective:
        best_init_idx = np.argmax(initial_scores)
        baseline_ad_vis = initial_scores[best_init_idx]
        baseline_content_qual = 0.0
    else:
        multi_objectives = []
        for scores_vec in initial_scores_vectors:
            ad_vis, content_qual = calculate_objectives(scores_vec)
            multi_objectives.append((ad_vis, content_qual))

        weighted_scores = [calculate_weighted_score(ad_vis, content_qual)
                          for ad_vis, content_qual in multi_objectives]
        best_init_idx = np.argmax(weighted_scores)
        baseline_ad_vis, baseline_content_qual = multi_objectives[best_init_idx]

    baseline_d6 = initial_d6_pool[best_init_idx]
    baseline_score = initial_scores[best_init_idx]
    baseline_scores_wordpos = initial_scores_vectors[best_init_idx]
    baseline_scores_wordcount = initial_scores_vectors_wc[best_init_idx] if initial_scores_vectors_wc else np.array([])
    baseline_scores_poscount = initial_scores_vectors_pc[best_init_idx] if initial_scores_vectors_pc else np.array([])

    if print_info:
        print(f"\nBaseline D6 selected: candidate {best_init_idx+1}/{len(initial_d6_pool)}")

    return (baseline_d6, baseline_score, baseline_scores_wordpos,
            baseline_scores_wordcount, baseline_scores_poscount,
            baseline_ad_vis, baseline_content_qual)

