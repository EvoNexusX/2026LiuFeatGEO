import numpy as np
from typing import List, Tuple


WEIGHTED_SCORE_AD_WEIGHT = 0.8
WEIGHTED_SCORE_QUALITY_WEIGHT = 0.2


def pareto_dominates(obj1: Tuple[float, float], obj2: Tuple[float, float]) -> bool:
    """Return whether ``obj1`` Pareto-dominates ``obj2``."""
    ad1, quality1 = obj1
    ad2, quality2 = obj2
    
    better_in_at_least_one = (ad1 > ad2) or (quality1 > quality2)
    
    not_worse_in_all = (ad1 >= ad2) and (quality1 >= quality2)
    
    return better_in_at_least_one and not_worse_in_all


def calculate_objectives(scores_vec: np.ndarray) -> Tuple[float, float]:
    """Extract ad visibility and quality from a 7D score vector."""
    if len(scores_vec) < 7:
        raise ValueError(f"Expected a 7-dimensional score vector, got {len(scores_vec)} dimensions")
    
    ad_visibility = float(scores_vec[5])
    quality_score = float(scores_vec[6])
    
    return ad_visibility, quality_score


def calculate_weighted_score(ad_visibility: float, content_quality: float) -> float:
    """Compute the weighted score."""
    return WEIGHTED_SCORE_AD_WEIGHT * ad_visibility + WEIGHTED_SCORE_QUALITY_WEIGHT * content_quality


def non_dominated_sort(objectives: List[Tuple[float, float]]) -> List[List[int]]:
    """Partition candidates into Pareto fronts."""
    n = len(objectives)
    domination_counts = [0] * n
    dominated_by = [[] for _ in range(n)]
    dominates = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if pareto_dominates(objectives[i], objectives[j]):
                dominates[i].append(j)
                dominated_by[j].append(i)
                domination_counts[j] += 1
            elif pareto_dominates(objectives[j], objectives[i]):
                dominates[j].append(i)
                dominated_by[i].append(j)
                domination_counts[i] += 1
    
    fronts = []
    current_front = [i for i in range(n) if domination_counts[i] == 0]
    
    while current_front:
        fronts.append(current_front)
        next_front = []
        for i in current_front:
            for j in dominates[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front = next_front
    
    return fronts


def crowding_distance(objectives: List[Tuple[float, float]], front: List[int]) -> List[float]:
    """Compute crowding distance for one Pareto front."""
    if len(front) <= 2:
        return [float('inf')] * len(front)
    
    distances = [0.0] * len(front)
    
    for obj_idx in range(2):
        sorted_indices = sorted(range(len(front)), 
                               key=lambda i: objectives[front[i]][obj_idx])
        
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        obj_min = objectives[front[sorted_indices[0]]][obj_idx]
        obj_max = objectives[front[sorted_indices[-1]]][obj_idx]
        obj_range = obj_max - obj_min
        
        if obj_range > 0:
            for i in range(1, len(front) - 1):
                distances[sorted_indices[i]] += (
                    objectives[front[sorted_indices[i + 1]]][obj_idx] -
                    objectives[front[sorted_indices[i - 1]]][obj_idx]
                ) / obj_range
    
    return distances


def select_by_nsga2(objectives: List[Tuple[float, float]], 
                    population_size: int) -> List[int]:
    """Select candidate indices with NSGA-II."""
    fronts = non_dominated_sort(objectives)
    selected = []
    
    for front in fronts:
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
        else:
            remaining = population_size - len(selected)
            distances = crowding_distance(objectives, front)
            sorted_front = sorted(range(len(front)), 
                                 key=lambda i: distances[i], 
                                 reverse=True)
            selected.extend([front[i] for i in sorted_front[:remaining]])
            break
        
        if len(selected) >= population_size:
            break
    
    return selected


def get_pareto_front(objectives: List[Tuple[float, float]]) -> List[int]:
    """Return the first Pareto front."""
    fronts = non_dominated_sort(objectives)
    return fronts[0] if fronts else []

