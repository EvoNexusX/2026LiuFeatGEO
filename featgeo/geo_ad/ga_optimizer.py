from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np

from featgeo.geo_ad.feature_schema import FeatureConfig, random_config
from featgeo.geo_ad.d6_generator import generate_d6, extract_ad_topic
from featgeo.geo_ad.geo_runner_wrapper import run_geo_with_d6
from featgeo.geo_ad.multi_objective import (
    calculate_objectives, 
    calculate_weighted_score,
    select_by_nsga2, 
    get_pareto_front,
    non_dominated_sort,
    crowding_distance
)


USE_PARETO_FRONT_SELECTION = True


def _format_ga_best_matrix(solutions: List[Dict[str, Any]]) -> str:
    """Format the weighted-best Pareto solution for compact GA progress logs."""
    if not solutions:
        return "best=N/A"

    best = max(
        solutions,
        key=lambda sol: calculate_weighted_score(sol['ad_visibility'], sol['content_quality'])
    )
    scores = np.array(best['scores_vector'])
    d_scores = " ".join(f"D{i+1}={scores[i]:.4f}" for i in range(min(6, len(scores))))
    weighted = calculate_weighted_score(best['ad_visibility'], best['content_quality'])
    return f"best [{d_scores}] Q={best['content_quality']:.4f} W={weighted:.4f}"


def select_parents_from_fronts(objectives: List[Tuple[float, float]], 
                                population: List[FeatureConfig],
                                num_parents: int = 2) -> List[FeatureConfig]:
    """Select parents with Pareto front ranking and crowding distance."""
    fronts = non_dominated_sort(objectives)
    
    selection_pool = []
    
    for front_rank, front in enumerate(fronts):
        if len(front) > 0:
            distances = crowding_distance(objectives, front)
            for i, idx in enumerate(front):
                selection_pool.append((idx, front_rank, distances[i]))
    
    selection_pool.sort(key=lambda x: (x[1], -x[2]))
    
    parents = []
    pool_size = len(selection_pool)
    
    for _ in range(num_parents):
        i, j = random.randrange(pool_size), random.randrange(pool_size)
        candidate_i = selection_pool[i]
        candidate_j = selection_pool[j]
        
        if candidate_i[1] < candidate_j[1]:
            winner_idx = candidate_i[0]
        elif candidate_j[1] < candidate_i[1]:
            winner_idx = candidate_j[0]
        else:
            winner_idx = candidate_i[0] if candidate_i[2] >= candidate_j[2] else candidate_j[0]
        
        parents.append(population[winner_idx])
    
    return parents


class GAResult:
    def __init__(self, best_config: FeatureConfig, best_fitness: float, history: List[float], 
                 best_d6: Optional[str] = None, best_scores_vector: Optional[List[float]] = None,
                 pareto_front: Optional[List[Dict[str, Any]]] = None,
                 best_scores_wc: Optional[List[float]] = None,
                 best_scores_pc: Optional[List[float]] = None,
                 highest_ad_config: Optional[FeatureConfig] = None,
                 highest_ad_fitness: Optional[float] = None,
                 highest_ad_d6: Optional[str] = None,
                 highest_ad_scores_vector: Optional[List[float]] = None,
                 highest_ad_scores_wc: Optional[List[float]] = None,
                 highest_ad_scores_pc: Optional[List[float]] = None):
        self.best_config = best_config
        self.best_fitness = best_fitness
        self.history = history
        self.best_d6 = best_d6
        self.best_scores_vector = best_scores_vector
        self.pareto_front = pareto_front
        self.best_scores_wc = best_scores_wc
        self.best_scores_pc = best_scores_pc
        self.highest_ad_scores_wc = highest_ad_scores_wc
        self.highest_ad_scores_pc = highest_ad_scores_pc
        self.highest_ad_config = highest_ad_config
        self.highest_ad_fitness = highest_ad_fitness
        self.highest_ad_d6 = highest_ad_d6
        self.highest_ad_scores_vector = highest_ad_scores_vector


def crossover(a: FeatureConfig, b: FeatureConfig) -> FeatureConfig:
    """Arithmetic crossover with random convex weights."""
    values = {}
    for field in a.__dataclass_fields__.keys():
        wa = random.random()
        wb = 1.0 - wa
        values[field] = wa * getattr(a, field) + wb * getattr(b, field)
    return FeatureConfig(**values)


def mutate(cfg: FeatureConfig, p_mut: float = 0.1, sigma: float = 0.2) -> FeatureConfig:
    """Gaussian mutation with clipping to each feature range."""
    from .feature_schema import RANGES
    
    values = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()}
    for k in values.keys():
        if random.random() < p_mut:
            lo, hi = RANGES[k]
            step = random.gauss(0.0, sigma)
            values[k] = min(hi, max(lo, values[k] + step))
    return FeatureConfig(**values)


def evaluate_individual(cfg: FeatureConfig, scenes: List[Dict[str, Any]], print_responses: bool = False) -> float:
    scores = []
    for scene in scenes:
        query = scene['query']
        summaries_5 = scene['summaries_5']
        d6 = generate_d6(summaries_5, cfg)
        ad_score, _, _, _ = run_geo_with_d6(query, summaries_5, d6, print_responses=print_responses)
        scores.append(ad_score)
    return sum(scores) / max(1, len(scores))


def run_ga(scenes: List[Dict[str, Any]], popsize: int = 20, generations: int = 10, p_mut: float = 0.1, print_responses: bool = False) -> GAResult:
    """Multi-scene GA optimization over shared feature configs."""
    population = [random_config() for _ in range(popsize)]
    fitness = [evaluate_individual(cfg, scenes, print_responses=print_responses) for cfg in population]
    history = [max(fitness)]

    for _ in range(generations):
        parents = []
        for _ in range(popsize):
            i, j = random.randrange(popsize), random.randrange(popsize)
            parents.append(population[i] if fitness[i] >= fitness[j] else population[j])

        new_population = []
        for _ in range(popsize):
            a, b = random.choice(parents), random.choice(parents)
            child = crossover(a, b)
            child = mutate(child, p_mut=p_mut)
            new_population.append(child)

        population = new_population
        fitness = [evaluate_individual(cfg, scenes, print_responses=print_responses) for cfg in population]
        history.append(max(fitness))

    best_idx = max(range(popsize), key=lambda i: fitness[i])
    return GAResult(population[best_idx], fitness[best_idx], history, best_d6=None, best_scores_vector=None)


def run_ga_single_scene(query: str, summaries_5: List[str], popsize: int = 10, generations: int = 3, p_mut: float = 0.15, 
                       print_progress: bool = True, initial_population: List[FeatureConfig] = None,
                       initial_fitness: Optional[List[float]] = None,
                       initial_scores_vectors: Optional[List] = None,
                       initial_d6_pool: Optional[List[str]] = None,
                       ad_theme: Optional[str] = None) -> GAResult:
    """Single-scene GA optimization for one query and its five summaries.
    
    Args:
        query: Query text
        summaries_5: Five webpage summaries
        popsize: Population size
        generations: Number of generations
        p_mut: Mutation probability
        print_progress: Whether to print GA progress
        initial_population: Optional starting population
        initial_fitness: Optional cached fitness values aligned with initial_population
        initial_scores_vectors: Optional cached full score vectors aligned with initial_population
        initial_d6_pool: Optional cached D6 texts aligned with initial_population
        ad_theme: Optional ad theme. If omitted, it is extracted automatically.
        
    Returns:
        GAResult with the best config, fitness, and optimization history
    """
    if ad_theme is None:
        if print_progress:
            print("  [GA] Extracting ad theme...")
        ad_theme = extract_ad_topic(summaries_5, verbose=False)
        if print_progress:
            print("  [GA] Ad theme completed")
    
    if print_progress:
        if initial_population:
            print(f"  [GA] Using the provided initial population (popsize={len(initial_population)}). Starting evolution...")
        else:
            print(f"  [GA] Initializing a new population (popsize={popsize})...")
    
    if initial_population:
        population = initial_population
        popsize = len(population)
    else:
        population = [random_config() for _ in range(popsize)]
    
    if initial_fitness is not None and len(initial_fitness) == popsize:
        fitness = list(initial_fitness)
        best_idx_0 = max(range(popsize), key=lambda i: fitness[i])
        
        if initial_scores_vectors is not None and len(initial_scores_vectors) == popsize:
            import numpy as np
            best_scores_vector = initial_scores_vectors[best_idx_0]
            if not isinstance(best_scores_vector, np.ndarray):
                best_scores_vector = np.array(best_scores_vector)
            if print_progress:
                print("  [GA] Generation 0 reuses cached full score vectors (no re-evaluation)")
        else:
            d6_best = generate_d6(summaries_5, population[best_idx_0], ad_theme=ad_theme)
            _, best_scores_vector, _, _ = run_geo_with_d6(query, summaries_5, d6_best, print_responses=False, skip_quality_eval=True)
            if print_progress:
                print("  [GA] [Warning] Generation 0 is missing cached score vectors, so it was re-evaluated (scores may differ)")
    else:
        fitness = []
        scores_vectors = []
        for cfg in population:
            d6 = generate_d6(summaries_5, cfg, ad_theme=ad_theme)
            ad_score, scores_vec, _, _ = run_geo_with_d6(query, summaries_5, d6, print_responses=False, skip_quality_eval=True)
            fitness.append(ad_score)
            scores_vectors.append(scores_vec)
        best_idx_0 = max(range(popsize), key=lambda i: fitness[i])
        best_scores_vector = scores_vectors[best_idx_0]
    
    global_best_fitness = max(fitness)
    global_best_idx = max(range(popsize), key=lambda i: fitness[i])
    global_best_config = population[global_best_idx]
    global_best_scores_vector = best_scores_vector
    
    if initial_d6_pool is not None and len(initial_d6_pool) == popsize:
        global_best_d6 = initial_d6_pool[global_best_idx]
    else:
        global_best_d6 = generate_d6(summaries_5, global_best_config, ad_theme=ad_theme)
    
    history = [global_best_fitness]
    if print_progress:
        print(f"  [GA] Generation 0 completed | best={history[0]:.4f}")
    
    for gen in range(generations):
        if print_progress:
            print(f"  [GA] Generation {gen+1}/{generations} started")

        parents = []
        for _ in range(popsize):
            i, j = random.randrange(popsize), random.randrange(popsize)
            parents.append(population[i] if fitness[i] >= fitness[j] else population[j])
        
        new_population = []
        for _ in range(popsize):
            a, b = random.choice(parents), random.choice(parents)
            child = crossover(a, b)
            child = mutate(child, p_mut=p_mut)
            new_population.append(child)
        
        population = new_population
        fitness = []
        scores_vectors = []
        d6_texts = []
        for i, cfg in enumerate(population):
            d6 = generate_d6(summaries_5, cfg, ad_theme=ad_theme)
            d6_texts.append(d6)
            ad_score, scores_vec, _, _ = run_geo_with_d6(query, summaries_5, d6, print_responses=False, skip_quality_eval=True)
            fitness.append(ad_score)
            scores_vectors.append(scores_vec)
            

        gen_best_idx = max(range(popsize), key=lambda i: fitness[i])
        gen_best_fitness = fitness[gen_best_idx]
        if gen_best_fitness > global_best_fitness:
            global_best_fitness = gen_best_fitness
            global_best_config = population[gen_best_idx]
            global_best_scores_vector = scores_vectors[gen_best_idx]
            global_best_d6 = d6_texts[gen_best_idx]
        
        history.append(max(fitness))
        if print_progress:
            print(f"  [GA] Generation {gen+1}/{generations} completed | best={history[-1]:.4f}")
    
    return GAResult(global_best_config, global_best_fitness, history, 
                    best_d6=global_best_d6, best_scores_vector=global_best_scores_vector)


def run_ga_multi_objective(query: str, summaries_5: List[str], 
                           popsize: int = 10, generations: int = 3, p_mut: float = 0.15,
                           print_progress: bool = True,
                           initial_population: List[FeatureConfig] = None,
                           initial_fitness: Optional[List[float]] = None,
                           initial_scores_vectors: Optional[List] = None,
                           initial_scores_vectors_wc: Optional[List] = None,
                           initial_scores_vectors_pc: Optional[List] = None,
                           initial_d6_pool: Optional[List[str]] = None,
                           ad_theme: Optional[str] = None,
                           d6_cache_records: Optional[List[Dict]] = None) -> Tuple[GAResult, List[Dict]]:
    """
    Multi-objective GA optimization (NSGA-II) for ad visibility and content quality.
    
    Objective 1: maximize ad visibility
    Objective 2: maximize content quality
    
    Args:
        query: Query text
        summaries_5: Five webpage summaries
        popsize: Population size
        generations: Number of generations
        p_mut: Mutation probability
        print_progress: Whether to print progress
        initial_population: Optional starting population
        initial_fitness: Optional cached fitness values (ignored here)
        initial_scores_vectors: Optional wordpos score vectors
        initial_scores_vectors_wc: Optional wordcount score vectors
        initial_scores_vectors_pc: Optional poscount score vectors
        initial_d6_pool: Optional cached D6 texts
        ad_theme: Optional ad theme. If omitted, it is extracted automatically.
        d6_cache_records: Optional D6 evaluation cache records
        
    Returns:
        (GAResult, new_evaluations):
            - GAResult with the Pareto front and representative solutions
            - new_evaluations with newly evaluated records for cache updates
    """
    if ad_theme is None:
        if print_progress:
            print("  [Multi-Objective GA] Extracting ad theme...")
        ad_theme = extract_ad_topic(summaries_5, verbose=False)
        if print_progress:
            print("  [Multi-Objective GA] Ad theme completed")
    
    if print_progress:
        print("  [Multi-Objective GA] Running NSGA-II on objectives: (ad visibility, content quality)")
        if initial_population:
            print(f"  [Multi-Objective GA] Using the provided initial population (popsize={len(initial_population)})")
    
    if d6_cache_records is None:
        d6_cache_records = []
    new_evaluations = []
    cache_hit_count = 0
    cache_miss_count = 0
    
    if initial_population:
        population = initial_population
        popsize = len(population)
    else:
        population = [random_config() for _ in range(popsize)]
    
    d6_texts = []
    scores_wc = []
    scores_pc = []
    
    has_full_cache = (initial_scores_vectors is not None and len(initial_scores_vectors) == popsize and
                      initial_scores_vectors_wc is not None and len(initial_scores_vectors_wc) == popsize and
                      initial_scores_vectors_pc is not None and len(initial_scores_vectors_pc) == popsize and
                      initial_d6_pool is not None and len(initial_d6_pool) == popsize)
    
    if has_full_cache:
        scores_vectors = [np.array(vec) if not isinstance(vec, np.ndarray) else vec 
                         for vec in initial_scores_vectors]
        scores_wc = [np.array(vec) if not isinstance(vec, np.ndarray) else vec 
                    for vec in initial_scores_vectors_wc]
        scores_pc = [np.array(vec) if not isinstance(vec, np.ndarray) else vec 
                    for vec in initial_scores_vectors_pc]
        d6_texts = initial_d6_pool
        if print_progress:
            print("  [Multi-Objective GA] Generation 0 reused cached scores")
    elif initial_scores_vectors is not None and len(initial_scores_vectors) == popsize:
        scores_vectors = [np.array(vec) if not isinstance(vec, np.ndarray) else vec 
                         for vec in initial_scores_vectors]
        if initial_d6_pool is not None and len(initial_d6_pool) == popsize:
            d6_texts = initial_d6_pool
            for d6 in d6_texts:
                _, _, wc, pc = run_geo_with_d6(query, summaries_5, d6, print_responses=False)
                scores_wc.append(wc)
                scores_pc.append(pc)
        else:
            for cfg in population:
                d6 = generate_d6(summaries_5, cfg, ad_theme=ad_theme)
                d6_texts.append(d6)
                _, _, wc, pc = run_geo_with_d6(query, summaries_5, d6, print_responses=False)
                scores_wc.append(wc)
                scores_pc.append(pc)
        if print_progress:
            print("  [Multi-Objective GA] Generation 0 reused cache and completed metric backfill")
    else:
        scores_vectors = []
        for cfg in population:
            d6 = generate_d6(summaries_5, cfg, ad_theme=ad_theme)
            d6_texts.append(d6)
            _, scores_vec, wc, pc = run_geo_with_d6(query, summaries_5, d6, print_responses=False)
            scores_vectors.append(scores_vec)
            scores_wc.append(wc)
            scores_pc.append(pc)
    
    objectives = [calculate_objectives(vec) for vec in scores_vectors]
    
    pareto_indices = get_pareto_front(objectives)
    
    global_pareto_solutions = []
    for idx in pareto_indices:
        ad_vis, quality = objectives[idx]
        global_pareto_solutions.append({
            'config': population[idx],
            'ad_visibility': ad_vis,
            'content_quality': quality,
            'scores_vector': scores_vectors[idx],
            'scores_wc': scores_wc[idx],
            'scores_pc': scores_pc[idx],
            'd6_text': d6_texts[idx] if idx < len(d6_texts) else generate_d6(summaries_5, population[idx], ad_theme=ad_theme, verbose=False),
            'generation': 0
        })
    
    if print_progress:
        print(
            f"  [Multi-Objective GA] Generation 0 completed | "
            f"Pareto={len(global_pareto_solutions)} | {_format_ga_best_matrix(global_pareto_solutions)}"
        )
    
    for gen in range(generations):
        offspring_population = []
        offspring_scores = []
        offspring_scores_wc = []
        offspring_scores_pc = []
        offspring_objectives = []
        
        for offspring_idx in range(popsize):
            parent1, parent2 = select_parents_from_fronts(objectives, population, num_parents=2)
            
            child = crossover(parent1, parent2)
            child = mutate(child, p_mut=p_mut)
            offspring_population.append(child)
        
        offspring_d6_texts = []
        if print_progress:
            print(f"  [Multi-Objective GA] Generation {gen+1}/{generations} started")
        for i, cfg in enumerate(offspring_population):
            cached_d6, cached_scores_vec = None, None
            
            if d6_cache_records:
                cfg_dict = cfg.__dict__
                for record in d6_cache_records:
                    if record.get('query') == query and record.get('ad_theme') == ad_theme:
                        cached_cfg = record.get('cfg', {})
                        if all(abs(cfg_dict.get(k, 0) - cached_cfg.get(k, -999)) < 1e-6 for k in cfg_dict.keys()):
                            cached_d6 = record.get('d6_text')
                            cached_scores_vec = record.get('scores_vector')
                            if cached_d6 and cached_scores_vec:
                                cache_hit_count += 1
                                break
            
            if cached_d6 and cached_scores_vec:
                d6 = cached_d6
                scores_vec = np.array(cached_scores_vec) if not isinstance(cached_scores_vec, np.ndarray) else cached_scores_vec
                _, _, child_wc, child_pc = run_geo_with_d6(query, summaries_5, d6, print_responses=False)
            else:
                cache_miss_count += 1
                d6 = generate_d6(summaries_5, cfg, ad_theme=ad_theme, verbose=False)
                _, scores_vec, child_wc, child_pc = run_geo_with_d6(query, summaries_5, d6, print_responses=False)
                
                new_evaluations.append({
                    'query': query,
                    'cfg': cfg.__dict__,
                    'ad_theme': ad_theme,
                    'd6_text': d6,
                    'score': float(scores_vec[-1]),
                    'scores_vector': scores_vec.tolist() if isinstance(scores_vec, np.ndarray) else list(scores_vec)
                })
            
            offspring_population[i] = cfg
            offspring_d6_texts.append(d6)
            offspring_scores.append(scores_vec)
            
            offspring_scores_wc.append(child_wc)
            offspring_scores_pc.append(child_pc)
            ad_vis, content_qual = calculate_objectives(scores_vec)
            offspring_objectives.append((ad_vis, content_qual))
            

        combined_population = population + offspring_population
        combined_objectives = objectives + offspring_objectives
        combined_scores = scores_vectors + offspring_scores
        combined_scores_wc = scores_wc + offspring_scores_wc
        combined_scores_pc = scores_pc + offspring_scores_pc
        combined_d6_texts = d6_texts + offspring_d6_texts
        
        selected_indices = select_by_nsga2(combined_objectives, popsize)
        
        population = [combined_population[i] for i in selected_indices]
        objectives = [combined_objectives[i] for i in selected_indices]
        scores_vectors = [combined_scores[i] for i in selected_indices]
        scores_wc = [combined_scores_wc[i] for i in selected_indices]
        scores_pc = [combined_scores_pc[i] for i in selected_indices]
        d6_texts = [combined_d6_texts[i] for i in selected_indices]
        
        current_gen_solutions = []
        for idx in range(len(population)):
            ad_vis, quality = objectives[idx]
            current_gen_solutions.append({
                'config': population[idx],
                'ad_visibility': ad_vis,
                'content_quality': quality,
                'scores_vector': scores_vectors[idx],
                'scores_wc': scores_wc[idx],
                'scores_pc': scores_pc[idx],
                'd6_text': d6_texts[idx],  # Reuse the exact D6 text that was actually evaluated.
                'generation': gen + 1
            })
        
        def config_to_key(cfg):
            """Convert a config into a hashable key."""
            return tuple(sorted(cfg.__dict__.items()))
        
        existing_configs = {config_to_key(sol['config']) for sol in global_pareto_solutions}
        for sol in current_gen_solutions:
            key = config_to_key(sol['config'])
            if key not in existing_configs:
                global_pareto_solutions.append(sol)
                existing_configs.add(key)
        
        combined_objectives = [(sol['ad_visibility'], sol['content_quality']) for sol in global_pareto_solutions]
        
        global_pareto_indices = get_pareto_front(combined_objectives)
        global_pareto_solutions = [global_pareto_solutions[i] for i in global_pareto_indices]
        
        if print_progress:
            print(
                f"  [Multi-Objective GA] Generation {gen+1}/{generations} completed | "
                f"offspring={popsize} | Pareto={len(global_pareto_solutions)} | "
                f"{_format_ga_best_matrix(global_pareto_solutions)}"
            )
    
    best_solution = None
    best_weighted_score = -float('inf')
    
    highest_ad_solution = None
    highest_ad_visibility = -float('inf')
    
    for sol in global_pareto_solutions:
        weighted = calculate_weighted_score(sol['ad_visibility'], sol['content_quality'])
        if weighted > best_weighted_score:
            best_weighted_score = weighted
            best_solution = sol
        
        should_update = False
        if sol['ad_visibility'] > highest_ad_visibility:
            should_update = True
        elif sol['ad_visibility'] == highest_ad_visibility and highest_ad_solution is not None:
            if sol['content_quality'] > highest_ad_solution['content_quality']:
                should_update = True
        elif highest_ad_solution is None:
            should_update = True
        
        if should_update:
            highest_ad_visibility = sol['ad_visibility']
            highest_ad_solution = sol
    
    best_config = best_solution['config']
    best_fitness = best_solution['ad_visibility']
    best_d6 = best_solution['d6_text']
    best_scores_vector = best_solution['scores_vector']
    best_scores_wc = best_solution['scores_wc']
    best_scores_pc = best_solution['scores_pc']
    
    highest_ad_config = highest_ad_solution['config']
    highest_ad_fitness = highest_ad_solution['ad_visibility']
    highest_ad_d6 = highest_ad_solution['d6_text']
    highest_ad_scores_vector = highest_ad_solution['scores_vector']
    highest_ad_scores_wc = highest_ad_solution['scores_wc']
    highest_ad_scores_pc = highest_ad_solution['scores_pc']
    
    history = []
    
    if print_progress:
        print(
            f"  [Multi-Objective GA] Optimization completed | "
            f"Pareto={len(global_pareto_solutions)} | {_format_ga_best_matrix(global_pareto_solutions)}"
        )
    
    ga_result = GAResult(
        best_config=best_config,
        best_fitness=best_fitness,
        history=history,
        best_d6=best_d6,
        best_scores_vector=best_scores_vector,
        pareto_front=global_pareto_solutions,
        best_scores_wc=best_scores_wc,
        best_scores_pc=best_scores_pc,
        highest_ad_config=highest_ad_config,
        highest_ad_fitness=highest_ad_fitness,
        highest_ad_d6=highest_ad_d6,
        highest_ad_scores_vector=highest_ad_scores_vector,
        highest_ad_scores_wc=highest_ad_scores_wc,
        highest_ad_scores_pc=highest_ad_scores_pc
    )
    
    return ga_result, new_evaluations

