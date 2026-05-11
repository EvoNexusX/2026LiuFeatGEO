#!/usr/bin/env python3
"""Launch parallel GEO optimization shards from the shared config settings."""
import os
import sys
import time
import subprocess
import random
import json
from pathlib import Path

try:
    from featgeo import config
except ImportError:
    print("[Error] Unable to import config.py. Please make sure the file exists.")
    sys.exit(1)


def project_root():
    return Path(__file__).resolve().parents[2]


def create_logs_dir():
    """Create the log directory used by shard workers."""
    logs_dir = Path(__file__).parent / 'logs'
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def calculate_shard_ranges(total_samples, num_shards, sampling_mode='continuous', random_indices=None):
    """Split work into shard ranges for continuous, sparse, or random sampling."""
    if sampling_mode == 'sparse':
        ranges = []
        for i in range(num_shards):
            start = i * 100
            end = start + 10
            ranges.append((start, end))
        return ranges
    elif sampling_mode == 'random':
        if random_indices is None:
            raise ValueError("random_indices must be provided for random sampling mode")
        
        shard_size = len(random_indices) // num_shards
        
        ranges = []
        start = 0
        for i in range(num_shards):
            end = start + shard_size
            if i == num_shards - 1:
                end = len(random_indices)
            ranges.append(random_indices[start:end])
            start = end
        
        return ranges
    else:
        shard_size = total_samples // num_shards
        
        ranges = []
        start = 0
        for i in range(num_shards):
            end = start + shard_size
            if i == num_shards - 1:
                end = total_samples
            ranges.append((start, end))
            start = end
        
        return ranges


def start_shard(shard_id, shard_range, logs_dir, api_key=None, sampling_mode='continuous', random_seed=None):
    """Launch one shard process with its own cache and log files."""
    if sampling_mode == 'random':
        indices_list = shard_range
        start_idx = min(indices_list)
        end_idx = max(indices_list) + 1
        seed_str = f"s{random_seed}" if random_seed is not None else "noseed"
        log_suffix = f"shard{shard_id}_random_{seed_str}_{len(indices_list)}samples"
    else:
        start_idx, end_idx = shard_range
        indices_list = None
        prefix = "all" if sampling_mode == "all" else "shard"
        log_suffix = f"{prefix}_{start_idx}_{end_idx}"
    
    log_file = logs_dir / f"{log_suffix}.log"
    pid_file = logs_dir / f"{log_suffix}.pid"
    
    cmd = [sys.executable, '-u', '-m', 'featgeo.geo_ad.run_geo_ad']
    
    if sampling_mode == 'random':
        seed_str = f"s{random_seed}" if random_seed is not None else "noseed"
        cache_suffix = f"_random_{seed_str}_shard{shard_id}"
        cmd.extend([
            '--sampling-mode', 'random',
            '--random-indices', json.dumps(indices_list),
            '--cache-suffix', cache_suffix,
        ])
    else:
        if sampling_mode == 'sparse':
            cache_start = (start_idx // 100) * 100
            cache_end = cache_start + 100
        elif sampling_mode == 'all':
            cache_start = start_idx
            cache_end = end_idx
        else:
            cache_start = start_idx
            cache_end = end_idx
        
        cache_prefix = "all" if sampling_mode == "all" else "shard"
        cache_suffix = f"_{cache_prefix}_{cache_start}_{cache_end}"
        cmd.extend([
            '--sampling-mode', sampling_mode,
            '--start-index', str(start_idx),
            '--end-index', str(end_idx),
            '--cache-suffix', cache_suffix,
        ])
    
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    data_dir.mkdir(exist_ok=True)
    cache_file = data_dir / f'global_cache{cache_suffix}.json'
    cmd.extend(['--global-cache-file', str(cache_file)])
    
    if api_key:
        cmd.extend(['--api-key', api_key])
        key_hint = f" | API_KEY=...{api_key[-8:]}" if len(api_key) > 8 else ""
    else:
        key_hint = ""
    
    log = open(log_file, 'w', buffering=1)
    process = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        cwd=project_root()
    )
    
    with open(pid_file, 'w') as f:
        f.write(str(process.pid))
    
    if sampling_mode == 'random':
        print(f"  [OK] Shard {shard_id}: {len(indices_list)} random samples | PID={process.pid}{key_hint} | Log={log_file.name}")
    else:
        print(f"  [OK] Shard {shard_id}: samples [{start_idx}, {end_idx}) | PID={process.pid}{key_hint} | Log={log_file.name}")
    return process


def wait_for_all_shards(processes):
    """Wait for all shard processes and stop them cleanly on interrupt."""
    print("\n[Wait] Waiting for all shards to finish...\n")
    
    try:
        for i, process in enumerate(processes, 1):
            process.wait()
            print(f"  [OK] Shard {i}/{len(processes)} finished (PID={process.pid})")
        
        print("\n" + "="*60)
        print("[OK] All shards completed.")
        print("="*60)
        return True
    
    except KeyboardInterrupt:
        print("\n\n[Warning] Interrupt detected. Stopping all shards...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
                print(f"  [Stop] Terminated PID={process.pid}")
        print("All shard processes were stopped.")
        return False


def main():
    print("="*60)
    print("FeatGEO Parallel Experiment Launcher")
    print("="*60)
    
    enable_parallel = getattr(config, 'ENABLE_PARALLEL', False)
    num_shards = getattr(config, 'PARALLEL_SHARDS', 4)
    manual_range = getattr(config, 'MANUAL_SHARD_RANGE', None)
    sample_limit = getattr(config, 'SAMPLE_LIMIT', 50)
    d6_model_backend = getattr(config, 'D6_MODEL_BACKEND', 'gpt')
    sampling_mode = getattr(config, 'SAMPLING_MODE', 'continuous')
    
    random_sample_size = getattr(config, 'RANDOM_SAMPLE_SIZE', 50)
    random_seed = getattr(config, 'RANDOM_SAMPLING_SEED', 42)
    
    api_keys = getattr(config, 'OPENAI_API_KEYS', None)
    if api_keys and isinstance(api_keys, list) and len(api_keys) > 0:
        use_multi_keys = True
    else:
        use_multi_keys = False
    
    mode_desc = {
        'continuous': 'continuous sampling',
        'sparse': 'sparse sampling (10 every 100)',
        'random': f'random sampling ({random_sample_size} samples)',
        'all': 'all samples'
    }
    
    print("\n[Config] Current configuration:")
    print(f"  - D6 page-generator backend: {d6_model_backend.upper()}")
    print(f"  - Fusion answer model: {getattr(config, 'FUSION_MODEL', 'gpt-4o-mini')}")
    dataset_total_size = 1000
    total_display = dataset_total_size if sampling_mode == 'all' else sample_limit
    print(f"  - Total samples: {total_display}")
    print(f"  - Sampling mode: {mode_desc.get(sampling_mode, sampling_mode)}")
    if sampling_mode == 'random':
        print(f"  - Random seed: {random_seed if random_seed is not None else 'true random'}")
    print(f"  - Parallel execution: {'enabled' if enable_parallel else 'disabled'}")
    if use_multi_keys:
        print(f"  - Multiple API keys: enabled ({len(api_keys)} keys)")
        print(f"  - Key assignment: {'1:1 mapping' if len(api_keys) >= num_shards else f'round-robin reuse (cycle every {len(api_keys)} shards)'}")
    else:
        print("  - Multiple API keys: disabled (all shards use the same key)")
    
    if not enable_parallel:
        print("\n[Warning] Parallel execution is disabled (config.ENABLE_PARALLEL=False)")
        print("Tip: set ENABLE_PARALLEL=True in config.py to enable shard execution")
        print("\nFalling back to serial execution...")
        print("="*60 + "\n")
        
        subprocess.run([sys.executable, '-m', 'featgeo.geo_ad.run_geo_ad'], cwd=project_root())
        return
    
    if d6_model_backend.lower() == 'ollama':
        print("\n[Warning] Warning: the D6 page generator is Ollama, and parallel execution is not recommended")
        print("  Reason: local page generation is constrained by GPU memory, and parallel jobs may exhaust VRAM")
        print("  Recommendation: set ENABLE_PARALLEL=False in config.py")
        response = input("\nContinue with parallel execution anyway? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    print(f"  - Shard count: {num_shards}")
    
    logs_dir = create_logs_dir()
    
    random_indices = None
    if sampling_mode == 'random':
        if random_seed is not None:
            random.seed(random_seed)
        all_indices = list(range(dataset_total_size))
        random_indices = random.sample(all_indices, min(random_sample_size, dataset_total_size))
        random_indices.sort()
        print(f"  - Randomly selected {len(random_indices)} samples from {dataset_total_size}")
        print(f"  - Sample indices: {random_indices[:5]}...{random_indices[-5:] if len(random_indices) > 10 else ''}")
    
    if manual_range:
        start_idx, end_idx = manual_range
        print(f"  - Manual range: [{start_idx}, {end_idx})")
        print("\n[Warning] Note: a manual range runs only one shard")
        print("="*60 + "\n")
        
        api_key = api_keys[0] if use_multi_keys else None
        process = start_shard(1, (start_idx, end_idx), logs_dir, api_key=api_key, sampling_mode=sampling_mode, random_seed=random_seed)
        success = wait_for_all_shards([process])
    else:
        if sampling_mode == 'random':
            total_samples = len(random_indices)
        elif sampling_mode == 'all':
            total_samples = dataset_total_size
        else:
            total_samples = sample_limit
        ranges = calculate_shard_ranges(total_samples, num_shards, sampling_mode, random_indices)
        
        print("\n[Shard] Shard allocation:")
        if sampling_mode == 'random':
            for i, indices in enumerate(ranges, 1):
                print(f"  Shard {i}: {len(indices)} random samples (indices: {indices[0]}...{indices[-1]})")
        else:
            for i, (start, end) in enumerate(ranges, 1):
                print(f"  Shard {i}: samples [{start}, {end}) - {end-start} samples")
        
        print("\n[Run] Launching all shards...")
        print("="*60)
        
        processes = []
        for i, shard_range in enumerate(ranges, 1):
            if use_multi_keys:
                api_key = api_keys[(i - 1) % len(api_keys)]
            else:
                api_key = None
            
            process = start_shard(i, shard_range, logs_dir, api_key=api_key, sampling_mode=sampling_mode, random_seed=random_seed)
            processes.append(process)
            time.sleep(2)
        
        print("\n[Tip] Monitoring tips:")
        print(f"  - View all logs: tail -f {logs_dir}/shard_*.log")
        print(f"  - Check progress: watch -n 5 'grep \"Processed\" {logs_dir}/shard_*.log'")
        print("  - Stop execution: press Ctrl+C")
        
        success = wait_for_all_shards(processes)
    
    if success:
        print("\n[Next] Next step:")
        print("  Run the merge script: python -m featgeo.geo_ad.merge_results")
        print("="*60)


if __name__ == '__main__':
    main()

