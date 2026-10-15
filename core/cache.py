import hashlib
import json
import os
import pickle


def _params_hash(params):
    blob = json.dumps(params, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha1(blob).hexdigest()[:10]


def cached_run(cache_dir, name, params, compute_fn):
    """Return `compute_fn()`, cached on disk under `cache_dir/<name>_<hash>.pkl`.

    `params` is hashed to key the cache file, so changing any run parameter
    (geometry, seeds, gamma, ...) misses the cache and recomputes rather than
    silently returning a stale result.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{name}_{_params_hash(params)}.pkl')

    if os.path.exists(path):
        with open(path, 'rb') as f:
            cached = pickle.load(f)
        print(f'[cache] loaded {name} from {path}')
        return cached['result']

    result = compute_fn()
    with open(path, 'wb') as f:
        pickle.dump({'params': params, 'result': result}, f)
    print(f'[cache] saved {name} to {path}')
    return result
