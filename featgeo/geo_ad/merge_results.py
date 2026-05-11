#!/usr/bin/env python3
import json
import os
import sys
import glob
import numpy as np
from datetime import datetime

from featgeo.geo_ad.result_formatter import print_average_matrix
from featgeo.geo_ad.multi_objective import calculate_objectives, calculate_weighted_score

def get_m11_mode():
    """Return the GA optimization mode."""
    try:
        from featgeo import config
        return getattr(config, 'METHOD_11_MODE', 'single').lower()
    except ImportError:
        return 'single'

def merge_shard_files(pattern, output_file, fallback_patterns=None):
    """Merge shard files."""
    shard_files = sorted(glob.glob(pattern))
    
    checked_patterns = []
    if fallback_patterns:
        if isinstance(fallback_patterns, str):
            fallback_patterns = [fallback_patterns]
        checked_patterns.extend(fallback_patterns)
        for fallback_pattern in fallback_patterns:
            if shard_files:
                break
            shard_files = sorted(glob.glob(fallback_pattern))
            if shard_files:
                print(f"Using matching files: {fallback_pattern}")
    
    if not shard_files:
        print(f"[Warning] No matching files found: {pattern}")
        for checked_pattern in checked_patterns:
            print(f"   Also checked: {checked_pattern}")
        return []
    
    print(f"Found {len(shard_files)} shard files:")
    for f in shard_files:
        print(f"  - {f}")
    
    all_records = []
    for shard_file in shard_files:
        try:
            with open(shard_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
                all_records.extend(records)
                print(f"  [OK] {shard_file}: {len(records)} records")
        except Exception as e:
            print(f"  [Error] {shard_file}: failed to read - {e}")
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"[OK] Merged into: {output_file} ({len(all_records)} records)\n")
    else:
        print(f"[OK] Merged {len(all_records)} records\n")
    
    return all_records


def merge_query_results(pattern, output_file, fallback_patterns=None):
    """Merge query result files and restore numpy arrays."""
    shard_files = sorted(glob.glob(pattern))
    
    checked_patterns = []
    if fallback_patterns:
        if isinstance(fallback_patterns, str):
            fallback_patterns = [fallback_patterns]
        checked_patterns.extend(fallback_patterns)
        for fallback_pattern in fallback_patterns:
            if shard_files:
                break
            shard_files = sorted(glob.glob(fallback_pattern))
            if shard_files:
                print(f"Using matching files: {fallback_pattern}")
    
    if not shard_files:
        print(f"[Warning] No matching files found: {pattern}")
        for checked_pattern in checked_patterns:
            print(f"   Also checked: {checked_pattern}")
        return []
    
    print(f"Found {len(shard_files)} query-result files:")
    for f in shard_files:
        print(f"  - {f}")
    
    all_query_results = []
    for shard_file in shard_files:
        try:
            with open(shard_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                for result in results:
                    result['baseline'] = np.array(result['baseline'])
                    for method in result['methods']:
                        result['methods'][method] = np.array(result['methods'][method])
                    if 'methods_wordcount' in result:
                        for method in result['methods_wordcount']:
                            result['methods_wordcount'][method] = np.array(result['methods_wordcount'][method])
                    if 'methods_poscount' in result:
                        for method in result['methods_poscount']:
                            result['methods_poscount'][method] = np.array(result['methods_poscount'][method])
                    if result.get('ga_highest_ad') is not None:
                        result['ga_highest_ad'] = np.array(result['ga_highest_ad'])
                    if result.get('ga_highest_ad_wc') is not None:
                        result['ga_highest_ad_wc'] = np.array(result['ga_highest_ad_wc'])
                    if result.get('ga_highest_ad_pc') is not None:
                        result['ga_highest_ad_pc'] = np.array(result['ga_highest_ad_pc'])
                all_query_results.extend(results)
                print(f"  [OK] {shard_file}: {len(results)} results")
        except Exception as e:
            print(f"  [Error] {shard_file}: failed to read - {e}")
    
    print(f"[OK] Loaded all query results ({len(all_query_results)} total)\n")
    return all_query_results


def save_summary_to_json(all_query_results, output_path):
    """Write aggregate statistics to JSON."""
    if not all_query_results:
        return
    
    baseline_avg = np.mean([r['baseline'] for r in all_query_results], axis=0)
    baseline_var = np.var([r['baseline'] for r in all_query_results], axis=0)
    all_methods = list(all_query_results[0]['methods'].keys())
    baseline_ad_vis, baseline_content_qual = calculate_objectives(baseline_avg)
    
    summary = {
        'meta': {
            'total_queries': len(all_query_results),
            'timestamp': datetime.now().isoformat(),
            'methods_count': len(all_methods)
        },
        'baseline': {
            'scores': baseline_avg.tolist(),
            'variance': baseline_var.tolist(),
            'ad_visibility': float(baseline_ad_vis),
            'ad_visibility_variance': float(baseline_var[5]),
            'organic_total': float(np.sum(baseline_avg[:5])),
            'organic_total_variance': float(np.sum(baseline_var[:5]))
        },
        'methods': {}
    }
    
    for method_name in all_methods:
        method_scores = [r['methods'][method_name] for r in all_query_results]
        scores_avg = np.mean(method_scores, axis=0)
        scores_var = np.var(method_scores, axis=0)
        
        ad_vis, content_qual = calculate_objectives(scores_avg)
        weighted = calculate_weighted_score(ad_vis, content_qual)
        
        summary['methods'][method_name] = {
            'scores': scores_avg.tolist(),
            'variance': scores_var.tolist(),
            'ad_visibility': float(ad_vis),
            'ad_visibility_variance': float(scores_var[5]),
            'content_quality': float(content_qual),
            'content_quality_variance': float(scores_var[6]),
            'weighted_score': float(weighted)
        }
    
    ga_highest_scores = [r['ga_highest_ad'] for r in all_query_results if r.get('ga_highest_ad') is not None]
    if ga_highest_scores:
        ga_highest_avg = np.mean(ga_highest_scores, axis=0)
        ga_highest_var = np.var(ga_highest_scores, axis=0)
        ad_vis_ga, content_qual_ga = calculate_objectives(ga_highest_avg)
        weighted_ga = calculate_weighted_score(ad_vis_ga, content_qual_ga)
        
        summary['ga_highest_d6'] = {
            'scores': ga_highest_avg.tolist(),
            'variance': ga_highest_var.tolist(),
            'ad_visibility': float(ad_vis_ga),
            'ad_visibility_variance': float(ga_highest_var[5]),
            'content_quality': float(content_qual_ga),
            'content_quality_variance': float(ga_highest_var[6]),
            'weighted_score': float(weighted_ga)
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Saved summary JSON: {output_path}")


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'result'))
    os.makedirs(result_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report_path = os.path.join(result_dir, f'merge_report_{timestamp}.txt')
    summary_json_path = os.path.join(result_dir, f'summary_{timestamp}.json')
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    with open(report_path, 'w', encoding='utf-8') as report_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, report_file)
        
        try:
            print("="*60)
            print("Merging parallel shard outputs")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60 + "\n")
            
            print("1. Merging D6 initial-population logs...")
            d6_pattern = os.path.join(data_dir, 'ad_d6_population_log_shard_*.json')
            d6_random_pattern = os.path.join(data_dir, 'ad_d6_population_log_random_s*_shard*.json')
            d6_all_pattern = os.path.join(data_dir, 'ad_d6_population_log_all_*.json')
            d6_output = os.path.join(data_dir, 'ad_d6_population_log_merged.json')
            d6_records = merge_shard_files(d6_pattern, d6_output, [d6_random_pattern, d6_all_pattern])
            
            print("2. Merging GA evaluation caches...")
            ga_pattern = os.path.join(data_dir, 'ad_ga_evaluation_cache_shard_*.json')
            ga_random_pattern = os.path.join(data_dir, 'ad_ga_evaluation_cache_random_s*_shard*.json')
            ga_all_pattern = os.path.join(data_dir, 'ad_ga_evaluation_cache_all_*.json')
            ga_output = os.path.join(data_dir, 'ad_ga_evaluation_cache_merged.json')
            ga_records = merge_shard_files(ga_pattern, ga_output, [ga_random_pattern, ga_all_pattern])
            
            print("="*60)
            print("Merge statistics")
            print("="*60)
            print(f"D6 initial-population records: {len(d6_records)}")
            print(f"GA evaluation cache records: {len(ga_records)}")
            
            print("3. Merging query results and computing the average matrix...")
            results_pattern = os.path.join(data_dir, 'query_results_shard_*.json')
            results_random_pattern = os.path.join(data_dir, 'query_results_random_s*_shard*.json')
            results_all_pattern = os.path.join(data_dir, 'query_results_all_*.json')
            all_query_results = merge_query_results(results_pattern, None, [results_random_pattern, results_all_pattern])
            
            if all_query_results:
                print("\n" + "="*60)
                print("Average visibility matrix")
                print("="*60)
                m11_mode = get_m11_mode()
                print_average_matrix(all_query_results, single_objective=(m11_mode == 'single'))
                
                save_summary_to_json(all_query_results, summary_json_path)
                
                print("\n" + "="*60)
                print("Merge complete.")
                print("="*60)
                print("\n[Output] Results saved to:")
                print(f"  - Report file: {report_path}")
                print(f"  - Summary JSON: {summary_json_path}")
                print(f"  - Merged data: {data_dir}/")
            else:
                print("\n[Warning] No query-result files found. The average matrix cannot be computed.")
                print("Make sure the run produced shard query-result files.")
        
        finally:
            sys.stdout = original_stdout


if __name__ == '__main__':
    main()

