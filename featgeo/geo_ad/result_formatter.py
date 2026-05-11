import numpy as np
from typing import List, Dict, Any
from featgeo.geo_ad.multi_objective import calculate_objectives, calculate_weighted_score


STATUS_EPSILON = 5e-5


def format_status(delta: float, baseline: bool = False) -> str:
    """Return a compact result status for table output."""
    if baseline:
        return "="
    if delta > STATUS_EPSILON:
        return "+"
    if delta < -STATUS_EPSILON:
        return "-"
    return "="


def print_method_result(
    meth_name: str,
    final_scores: np.ndarray,
    improvement: float,
    baseline_content_quality: float = None,
    single_objective: bool = False
) -> None:
    """Print a compact completion line for one optimization method."""
    print(f"{meth_name} completed")


def print_summary_table(
    improvements: List[float],
    method_names: List[str],
    all_scores_vectors: List[np.ndarray],
    baseline_scores_vector: np.ndarray,
    single_objective: bool = False
) -> None:
    """Print the improvement summary table."""
    baseline_ad, baseline_qual = calculate_objectives(baseline_scores_vector)
    baseline_weighted = calculate_weighted_score(baseline_ad, baseline_qual)
    
    print(f"\n{'='*80}")
    if single_objective:
        print(f"Improvement Summary ({len(method_names)} methods, based on ad visibility)")
        print(f"{'='*80}")
        print(f"{'Method':<30} {'Ad Visibility':<16} {'Ad Delta':<16} {'Status'}")
    else:
        print(f"Improvement Summary ({len(method_names)} methods, based on weighted score)")
        print(f"{'='*80}")
        print(f"{'Method':<30} {'Ad Delta':<16} {'Quality Delta':<16} {'Weighted Delta':<16} {'Status'}")
    print(f"{'-'*80}")
    
    success_count = 0
    best_improvement = -float('inf')
    best_method = ""
    
    for i, (meth_name, scores_vec) in enumerate(zip(method_names, all_scores_vectors)):
        ad_vis, content_qual = calculate_objectives(scores_vec)
        weighted_score = calculate_weighted_score(ad_vis, content_qual)
        
        ad_change = ad_vis - baseline_ad
        qual_change = content_qual - baseline_qual
        weighted_change = weighted_score - baseline_weighted
        
        if single_objective:
            status = format_status(ad_change, baseline=(meth_name == 'identity'))
            if ad_change > STATUS_EPSILON:
                success_count += 1
            if ad_change > best_improvement:
                best_improvement = ad_change
                best_method = meth_name
        else:
            status = format_status(weighted_change, baseline=(meth_name == 'identity'))
            if weighted_change > STATUS_EPSILON:
                success_count += 1
            if weighted_change > best_improvement:
                best_improvement = weighted_change
                best_method = meth_name
        
        method_label = meth_name if i < 10 else f"{meth_name}(Method 11)"
        
        if single_objective:
            print(f"{method_label:<30} {ad_vis:.4f}          {ad_change:+.4f}          {status}")
        else:
            print(f"{method_label:<30} {ad_change:+.4f}      {qual_change:+.4f}      {weighted_change:+.4f}      {status}")
    
    print(f"{'-'*80}")
    if single_objective:
        print(f"Improved methods: {success_count}/{len(method_names)} (based on ad visibility)")
    else:
        print(f"Improved methods: {success_count}/{len(method_names)} (based on weighted score)")
    print(f"Best improvement: {best_improvement:+.4f} ({best_method})")


def print_visibility_matrix(
    method_names: List[str],
    all_scores_vectors: List[np.ndarray],
    baseline_scores_vector: np.ndarray,
    ga_highest_ad_scores: np.ndarray = None,
    all_scores_wordcount: List[np.ndarray] = None,
    all_scores_poscount: List[np.ndarray] = None,
    ga_highest_scores_wc: np.ndarray = None,
    ga_highest_scores_pc: np.ndarray = None,
    baseline_scores_wordcount: np.ndarray = None,
    baseline_scores_poscount: np.ndarray = None,
    single_objective: bool = False
) -> None:
    """Print the detailed visibility matrices across supported metrics."""
    baseline_ad, baseline_qual = calculate_objectives(baseline_scores_vector)
    baseline_weighted = calculate_weighted_score(baseline_ad, baseline_qual)
    
    if single_objective:
        print(f"\n{'='*100}")
        print("Matrix 1: word-count x position-weight (main optimization metric)")
        print(f"{'='*100}")
        print(f"{'Method':<30} {'D1':<10} {'D2':<10} {'D3':<10} {'D4':<10} {'D5':<10} {'D6':<10} {'Status'}")
        print(f"{'-'*100}")
    else:
        print(f"\n{'='*115}")
        print("Matrix 1: word-count x position-weight (main optimization metric)")
        print(f"{'='*115}")
        print(f"{'Method':<30} {'D1':<10} {'D2':<10} {'D3':<10} {'D4':<10} {'D5':<10} {'D6':<10} {'Quality':<8} {'Weighted':<8} {'Status'}")
        print(f"{'-'*115}")
    
    print(f"{'identity':<30} ", end='')
    for i in range(6):
        print(f"{baseline_scores_vector[i]:<10.6f} ", end='')
    if single_objective:
        print(format_status(0.0, baseline=True))
    else:
        print(f"{baseline_qual:<8.4f} {baseline_weighted:<8.4f} {format_status(0.0, baseline=True)}")
    
    for idx, (meth_name, scores_vec) in enumerate(zip(method_names, all_scores_vectors)):
        if meth_name == 'identity':
            continue
        
        ad_vis, content_qual = calculate_objectives(scores_vec)
        weighted_score = calculate_weighted_score(ad_vis, content_qual)
        
        if single_objective:
            ad_change = ad_vis - baseline_ad
            status = format_status(ad_change)
        else:
            weighted_change = weighted_score - baseline_weighted
            status = format_status(weighted_change)
        
        method_label = meth_name if idx < 10 else f"{meth_name}(Method 11)"
        
        print(f"{method_label:<30} ", end='')
        for i in range(6):
            print(f"{scores_vec[i]:<10.6f} ", end='')
        if single_objective:
            print(f"{status}")
        else:
            print(f"{content_qual:<8.4f} {weighted_score:<8.4f} {status}")
    
    if not single_objective and ga_highest_ad_scores is not None:
        ad_vis_highest, content_qual_highest = calculate_objectives(ga_highest_ad_scores)
        weighted_score_highest = calculate_weighted_score(ad_vis_highest, content_qual_highest)
        weighted_change_highest = weighted_score_highest - baseline_weighted
        status_highest = format_status(weighted_change_highest)
        
        print(f"{'  -> GA_HighestD6':<30} ", end='')
        for i in range(6):
            print(f"{ga_highest_ad_scores[i]:<10.6f} ", end='')
        print(f"{content_qual_highest:<8.4f} {weighted_score_highest:<8.4f} {status_highest}")
    
    if single_objective:
        print(f"{'-'*100}")
    else:
        print(f"{'-'*115}")
    
    if all_scores_wordcount:
        print(f"\n{'='*100}")
        print("Matrix 2: pure word count")
        print(f"{'='*100}")
        print(f"{'Method':<30} {'D1':<10} {'D2':<10} {'D3':<10} {'D4':<10} {'D5':<10} {'D6':<10}")
        print(f"{'-'*100}")
        
        baseline_wc = baseline_scores_wordcount if baseline_scores_wordcount is not None else baseline_scores_vector
        print(f"{'identity':<30} ", end='')
        for i in range(6):
            print(f"{baseline_wc[i]:<10.6f} ", end='')
        print()
        
        for idx, (meth_name, scores_wc) in enumerate(zip(method_names, all_scores_wordcount)):
            if meth_name == 'identity':
                continue
            
            method_label = meth_name if idx < 10 else f"{meth_name}(Method 11)"
            print(f"{method_label:<30} ", end='')
            for i in range(6):
                print(f"{scores_wc[i]:<10.6f} ", end='')
            print()
        
        if not single_objective:
            if ga_highest_scores_wc is not None:
                print(f"{'  -> GA_HighestD6':<30} ", end='')
                for i in range(6):
                    print(f"{ga_highest_scores_wc[i]:<10.6f} ", end='')
                print()
            elif ga_highest_ad_scores is not None:
                print(f"{'  -> GA_HighestD6':<30} ", end='')
                for i in range(6):
                    print(f"{ga_highest_ad_scores[i]:<10.6f} ", end='')
                print()
        
        print(f"{'-'*100}")
    
    if all_scores_poscount:
        print(f"\n{'='*100}")
        print("Matrix 3: pure position weight")
        print(f"{'='*100}")
        print(f"{'Method':<30} {'D1':<10} {'D2':<10} {'D3':<10} {'D4':<10} {'D5':<10} {'D6':<10}")
        print(f"{'-'*100}")
        
        baseline_pc = baseline_scores_poscount if baseline_scores_poscount is not None else baseline_scores_vector
        print(f"{'identity':<30} ", end='')
        for i in range(6):
            print(f"{baseline_pc[i]:<10.6f} ", end='')
        print()
        
        for idx, (meth_name, scores_pc) in enumerate(zip(method_names, all_scores_poscount)):
            if meth_name == 'identity':
                continue
            
            method_label = meth_name if idx < 10 else f"{meth_name}(Method 11)"
            print(f"{method_label:<30} ", end='')
            for i in range(6):
                print(f"{scores_pc[i]:<10.6f} ", end='')
            print()
        
        if not single_objective:
            if ga_highest_scores_pc is not None:
                print(f"{'  -> GA_HighestD6':<30} ", end='')
                for i in range(6):
                    print(f"{ga_highest_scores_pc[i]:<10.6f} ", end='')
                print()
            elif ga_highest_ad_scores is not None:
                print(f"{'  -> GA_HighestD6':<30} ", end='')
                for i in range(6):
                    print(f"{ga_highest_ad_scores[i]:<10.6f} ", end='')
                print()
        
        print(f"{'-'*100}")


def print_average_matrix(all_query_results: List[dict], single_objective: bool = False) -> None:
    """Print average matrices across multiple queries, including variances."""
    if len(all_query_results) <= 1:
        if len(all_query_results) == 1:
            print("\n[Note] Only 1 query was processed. Average and variance matrices are not needed.")
        return
    
    baseline_avg = np.mean([r['baseline'] for r in all_query_results], axis=0)
    baseline_var = np.var([r['baseline'] for r in all_query_results], axis=0)
    baseline_ad_avg, baseline_qual_avg = calculate_objectives(baseline_avg)
    baseline_weighted_avg = calculate_weighted_score(baseline_ad_avg, baseline_qual_avg)
    
    all_methods = list(all_query_results[0]['methods'].keys())
    
    if single_objective:
        print(f"\n{'='*294}")
        print(f"Average Matrix 1 across {len(all_query_results)} queries: word-count x position-weight")
        print(f"{'='*294}")
        print(f"{'Method':<30}{'D1 Mean':>11}{'D1 Var':>11}{'D2 Mean':>11}{'D2 Var':>11}{'D3 Mean':>11}{'D3 Var':>11}{'D4 Mean':>11}{'D4 Var':>11}{'D5 Mean':>11}{'D5 Var':>11}{'D6 Mean':>11}{'D6 Var':>11}{'Status':<8}")
        print(f"{'-'*294}")
    else:
        print(f"\n{'='*338}")
        print(f"Average Matrix 1 across {len(all_query_results)} queries: word-count x position-weight")
        print(f"{'='*338}")
        print(f"{'Method':<30}{'D1 Mean':>11}{'D1 Var':>11}{'D2 Mean':>11}{'D2 Var':>11}{'D3 Mean':>11}{'D3 Var':>11}{'D4 Mean':>11}{'D4 Var':>11}{'D5 Mean':>11}{'D5 Var':>11}{'D6 Mean':>11}{'D6 Var':>11}{'Quality Mean':>13}{'Quality Var':>12}{'Weighted Mean':>13}{'Weighted Var':>12} {'Status':<8}")
        print(f"{'-'*338}")
    
    baseline_ad_scores = [calculate_objectives(r['baseline'])[0] for r in all_query_results]
    baseline_qual_scores = [calculate_objectives(r['baseline'])[1] for r in all_query_results]
    baseline_weighted_scores = [calculate_weighted_score(ad, q) for ad, q in zip(baseline_ad_scores, baseline_qual_scores)]
    baseline_qual_var = np.var(baseline_qual_scores)
    baseline_weighted_var = np.var(baseline_weighted_scores)
    
    print(f"{'identity':<30}", end='')
    for i in range(6):
        print(f"{baseline_avg[i]:>11.6f}{baseline_var[i]:>11.6f}", end='')
    if single_objective:
        print(f"{format_status(0.0, baseline=True):<8}")
    else:
        print(f"{baseline_qual_avg:>13.6f}{baseline_qual_var:>12.6f}{baseline_weighted_avg:>13.6f}{baseline_weighted_var:>12.6f} {format_status(0.0, baseline=True):<8}")
    
    for idx, meth_name in enumerate(all_methods):
        if meth_name == 'identity':
            continue
        
        method_scores = [r['methods'][meth_name] for r in all_query_results]
        scores_avg = np.mean(method_scores, axis=0)
        scores_var = np.var(method_scores, axis=0)
        
        method_ad_scores = [calculate_objectives(scores)[0] for scores in method_scores]
        method_qual_scores = [calculate_objectives(scores)[1] for scores in method_scores]
        method_weighted_scores = [calculate_weighted_score(ad, q) for ad, q in zip(method_ad_scores, method_qual_scores)]
        
        ad_vis_avg = np.mean(method_ad_scores)
        content_qual_avg = np.mean(method_qual_scores)
        content_qual_var = np.var(method_qual_scores)
        weighted_avg = np.mean(method_weighted_scores)
        weighted_var = np.var(method_weighted_scores)
        
        if single_objective:
            ad_change = ad_vis_avg - baseline_ad_avg
            status = format_status(ad_change)
        else:
            weighted_change = weighted_avg - baseline_weighted_avg
            status = format_status(weighted_change)
        
        method_label = meth_name if idx < 10 else f"{meth_name}(Method 11)"
        
        print(f"{method_label:<30}", end='')
        for i in range(6):
            print(f"{scores_avg[i]:>11.6f}{scores_var[i]:>11.6f}", end='')
        if single_objective:
            print(f"{status:<8}")
        else:
            print(f"{content_qual_avg:>13.6f}{content_qual_var:>12.6f}{weighted_avg:>13.6f}{weighted_var:>12.6f} {status:<8}")
    
    if not single_objective:
        ga_highest_scores = [r['ga_highest_ad'] for r in all_query_results if r.get('ga_highest_ad') is not None]
        if ga_highest_scores:
            ga_highest_avg = np.mean(ga_highest_scores, axis=0)
            ga_highest_var = np.var(ga_highest_scores, axis=0)
            
            ga_ad_scores = [calculate_objectives(scores)[0] for scores in ga_highest_scores]
            ga_qual_scores = [calculate_objectives(scores)[1] for scores in ga_highest_scores]
            ga_weighted_scores = [calculate_weighted_score(ad, q) for ad, q in zip(ga_ad_scores, ga_qual_scores)]
            
            content_qual_highest_avg = np.mean(ga_qual_scores)
            content_qual_highest_var = np.var(ga_qual_scores)
            weighted_highest_avg = np.mean(ga_weighted_scores)
            weighted_highest_var = np.var(ga_weighted_scores)
            
            weighted_change_highest = weighted_highest_avg - baseline_weighted_avg
            status_highest = format_status(weighted_change_highest)
            
            print(f"{'  -> GA_HighestD6':<30}", end='')
            for i in range(6):
                print(f"{ga_highest_avg[i]:>11.6f}{ga_highest_var[i]:>11.6f}", end='')
            print(f"{content_qual_highest_avg:>13.6f}{content_qual_highest_var:>12.6f}{weighted_highest_avg:>13.6f}{weighted_highest_var:>12.6f} {status_highest:<8}")
    
    print(f"{'-'*338}")
    
    if 'methods_wordcount' in all_query_results[0]:
        print(f"\n{'='*294}")
        print(f"Average Matrix 2 across {len(all_query_results)} queries: pure word count")
        print(f"{'='*294}")
        print(f"{'Method':<30}{'D1 Mean':>11}{'D1 Var':>11}{'D2 Mean':>11}{'D2 Var':>11}{'D3 Mean':>11}{'D3 Var':>11}{'D4 Mean':>11}{'D4 Var':>11}{'D5 Mean':>11}{'D5 Var':>11}{'D6 Mean':>11}{'D6 Var':>11}")
        print(f"{'-'*294}")
        
        baseline_wc_data = [r.get('baseline_wordcount', r['baseline']) for r in all_query_results]
        baseline_wc_avg = np.mean(baseline_wc_data, axis=0)
        baseline_wc_var = np.var(baseline_wc_data, axis=0)
        print(f"{'identity':<30}", end='')
        for i in range(6):
            print(f"{baseline_wc_avg[i]:>11.6f}{baseline_wc_var[i]:>11.6f}", end='')
        print()
        
        for idx, meth_name in enumerate(all_methods):
            if meth_name == 'identity':
                continue
            
            method_scores_wc = [r['methods_wordcount'][meth_name] for r in all_query_results]
            scores_wc_avg = np.mean(method_scores_wc, axis=0)
            scores_wc_var = np.var(method_scores_wc, axis=0)
            method_label = meth_name if idx < 10 else f"{meth_name}(Method 11)"
            
            print(f"{method_label:<30}", end='')
            for i in range(6):
                print(f"{scores_wc_avg[i]:>11.6f}{scores_wc_var[i]:>11.6f}", end='')
            print()
        
        if not single_objective:
            ga_highest_scores_wc = [r['ga_highest_ad_wc'] for r in all_query_results if r.get('ga_highest_ad_wc') is not None]
            if ga_highest_scores_wc:
                ga_highest_wc_avg = np.mean(ga_highest_scores_wc, axis=0)
                ga_highest_wc_var = np.var(ga_highest_scores_wc, axis=0)
                print(f"{'  -> GA_HighestD6':<30}", end='')
                for i in range(6):
                    print(f"{ga_highest_wc_avg[i]:>11.6f}{ga_highest_wc_var[i]:>11.6f}", end='')
                print()
        
        print(f"{'-'*294}")
    
    if 'methods_poscount' in all_query_results[0]:
        print(f"\n{'='*294}")
        print(f"Average Matrix 3 across {len(all_query_results)} queries: pure position weight")
        print(f"{'='*294}")
        print(f"{'Method':<30}{'D1 Mean':>11}{'D1 Var':>11}{'D2 Mean':>11}{'D2 Var':>11}{'D3 Mean':>11}{'D3 Var':>11}{'D4 Mean':>11}{'D4 Var':>11}{'D5 Mean':>11}{'D5 Var':>11}{'D6 Mean':>11}{'D6 Var':>11}")
        print(f"{'-'*294}")
        
        baseline_pc_data = [r.get('baseline_poscount', r['baseline']) for r in all_query_results]
        baseline_pc_avg = np.mean(baseline_pc_data, axis=0)
        baseline_pc_var = np.var(baseline_pc_data, axis=0)
        print(f"{'identity':<30}", end='')
        for i in range(6):
            print(f"{baseline_pc_avg[i]:>11.6f}{baseline_pc_var[i]:>11.6f}", end='')
        print()
        
        for idx, meth_name in enumerate(all_methods):
            if meth_name == 'identity':
                continue
            
            method_scores_pc = [r['methods_poscount'][meth_name] for r in all_query_results]
            scores_pc_avg = np.mean(method_scores_pc, axis=0)
            scores_pc_var = np.var(method_scores_pc, axis=0)
            method_label = meth_name if idx < 10 else f"{meth_name}(Method 11)"
            
            print(f"{method_label:<30}", end='')
            for i in range(6):
                print(f"{scores_pc_avg[i]:>11.6f}{scores_pc_var[i]:>11.6f}", end='')
            print()
        
        if not single_objective:
            ga_highest_scores_pc = [r['ga_highest_ad_pc'] for r in all_query_results if r.get('ga_highest_ad_pc') is not None]
            if ga_highest_scores_pc:
                ga_highest_pc_avg = np.mean(ga_highest_scores_pc, axis=0)
                ga_highest_pc_var = np.var(ga_highest_scores_pc, axis=0)
                print(f"{'  -> GA_HighestD6':<30}", end='')
                for i in range(6):
                    print(f"{ga_highest_pc_avg[i]:>11.6f}{ga_highest_pc_var[i]:>11.6f}", end='')
                print()
        
        print(f"{'-'*294}")
    
    print(f"\nNote: the tables above show averages across {len(all_query_results)} queries")
    
    if not single_objective:
        _print_ga_highest_features(all_query_results)


def _print_ga_highest_features(all_query_results: List[Dict[str, Any]]) -> None:
    """Print summary statistics for GA_HighestD6 feature configs."""
    configs = []
    for r in all_query_results:
        config_dict = r.get('ga_highest_ad_config')
        if config_dict is not None:
            configs.append(config_dict)
    
    if not configs:
        print("\n[Feature Analysis] No GA_HighestD6 feature configs were found")
        return
    
    print(f"\n{'='*100}")
    print(f"GA_HighestD6 Feature Configuration Analysis ({len(configs)} queries)")
    print(f"{'='*100}")
    
    feature_names = list(configs[0].keys())
    
    print(f"\n{'Feature':<25} {'Mean':<12} {'Min':<12} {'Max':<12} {'Std':<12}")
    print(f"{'-'*73}")
    
    for feat_name in sorted(feature_names):
        values = [c[feat_name] for c in configs]
        avg_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        std_val = np.std(values)
        
        print(f"{feat_name:<25} {avg_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f} {std_val:<12.4f}")
    
    print(f"{'-'*73}")
    

