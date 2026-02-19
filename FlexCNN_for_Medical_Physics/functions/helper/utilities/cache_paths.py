import hashlib
import json
import os
import shutil
from typing import Dict, Iterable

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def cache_dataset_paths(path_map: Dict[str, str], settings: dict, exclude_keys: Iterable[str] = ()): 
    """
    Optionally copy dataset files into a local cache directory and return updated paths.

    Controlled by settings:
      - use_cache: bool (default False)
      - cache_dir: str (required when use_cache True)
      - cache_max_gb: int/float (default 40)
    
    NOTE: This cache is most effective for:
      - Many small files (<100 MB each) where network latency per file adds up
      - Random access patterns across multiple files
      - Full file loads (not memory-mapped reads)
    
    For large monolithic memory-mapped .np files (>1 GB) on Colab Drive,
    the cache may actually SLOW DOWN loading due to Drive's FUSE optimizations
    for large sequential reads. In such cases, set use_cache=False.
    """
    if not settings.get('use_cache', False):
        return path_map

    cache_dir = settings.get('cache_dir')
    if cache_dir is None:
        return path_map

    exclude_keys = set(exclude_keys)
    os.makedirs(cache_dir, exist_ok=True)

    index_path = os.path.join(cache_dir, 'cache_index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            cache_index = json.load(f)
    else:
        cache_index = {}

    max_bytes = int(settings.get('cache_max_gb', 40) * (1024 ** 3))
    cache_dir_abs = os.path.abspath(cache_dir)

    def _cached_name(src_path: str) -> str:
        digest = hashlib.sha1(src_path.encode('utf-8')).hexdigest()
        return f"{digest}_{os.path.basename(src_path)}"

    def _is_in_cache(src_path: str) -> bool:
        try:
            return os.path.abspath(src_path).startswith(cache_dir_abs + os.sep)
        except Exception:
            return False

    updated = dict(path_map)
    total_to_copy = 0
    copy_plan = []

    for key, src_path in path_map.items():
        if key in exclude_keys or src_path is None:
            continue
        if _is_in_cache(src_path):
            continue
        if not os.path.exists(src_path):
            continue

        src_size = os.path.getsize(src_path)
        src_mtime = os.path.getmtime(src_path)
        cached_path = os.path.join(cache_dir, _cached_name(src_path))
        cached_entry = cache_index.get(src_path)

        if cached_entry:
            cached_ok = (
                cached_entry.get('cached_path') == cached_path
                and cached_entry.get('size') == src_size
                and cached_entry.get('mtime') == src_mtime
                and os.path.exists(cached_path)
            )
            if cached_ok:
                updated[key] = cached_path
                continue

        if os.path.exists(cached_path):
            cached_size = os.path.getsize(cached_path)
            cached_ok = cached_size == src_size
            if cached_ok:
                cache_index[src_path] = {
                    'cached_path': cached_path,
                    'size': src_size,
                    'mtime': src_mtime,
                }
                updated[key] = cached_path
                continue

        total_to_copy += src_size
        copy_plan.append((key, src_path, cached_path, src_size, src_mtime))

    if total_to_copy > max_bytes:
        print(
            f"Cache size cap exceeded: {total_to_copy / (1024 ** 3):.2f} GB > "
            f"{max_bytes / (1024 ** 3):.2f} GB. Skipping new cache copies."
        )
        if cache_index:
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(cache_index, f, indent=2)
        return updated

    if copy_plan:
        print(f"\n📦 Caching {len(copy_plan)} file(s) to {cache_dir} ({total_to_copy / (1024 ** 3):.2f} GB total)")
        print("This may take 5-20 minutes depending on file sizes and network speed...")
    
    iterator = enumerate(copy_plan, 1)
    if HAS_TQDM:
        iterator = tqdm(iterator, total=len(copy_plan), desc="Caching files", unit="file")
    
    for idx, (key, src_path, cached_path, src_size, src_mtime) in iterator:
        file_desc = f"{os.path.basename(src_path)} ({src_size / (1024 ** 3):.2f} GB)"
        if not HAS_TQDM:
            print(f"  [{idx}/{len(copy_plan)}] Copying {file_desc}...", end='', flush=True)
        else:
            iterator.set_postfix_str(file_desc)
        
        os.makedirs(os.path.dirname(cached_path), exist_ok=True)
        
        # Copy to temp file first (atomic write to prevent corruption)
        temp_path = cached_path + '.tmp'
        try:
            shutil.copy2(src_path, temp_path)
            # Validate size after copy
            if os.path.getsize(temp_path) != src_size:
                raise IOError(f"Size mismatch after copy: expected {src_size}, got {os.path.getsize(temp_path)}")
            # Atomic rename (replaces any partial file)
            if os.path.exists(cached_path):
                os.remove(cached_path)
            os.rename(temp_path, cached_path)
            # Update index only after successful copy
            cache_index[src_path] = {
                'cached_path': cached_path,
                'size': src_size,
                'mtime': src_mtime,
            }
            updated[key] = cached_path
            if not HAS_TQDM:
                print(" ✓")
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if not HAS_TQDM:
                print(f" ✗ FAILED: {e}")
            else:
                print(f"\n  ✗ FAILED copying {os.path.basename(src_path)}: {e}")
            print(f"  Skipping cache for {key}, will use original path")
            # Don't update path; keep original
    
    if copy_plan:
        print(f"✅ Caching complete! All successful files ready in {cache_dir}\n")

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(cache_index, f, indent=2)

    return updated
