from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

__META_FILE_NAME = 'meta.yaml'


def meta_exists(dir_path: Path) -> bool:
    return (dir_path / __META_FILE_NAME).exists()


def save_meta_info(meta: Dict[str, Any]):
    with open(Path(meta['results_dir']) / __META_FILE_NAME, 'w+') as f:
        yaml.safe_dump(meta, f)


def load_meta_infos(base_dir: Path) -> Iterable[Dict[str, Any]]:
    metas = []
    for file_path in base_dir.rglob(__META_FILE_NAME):
        with open(file_path) as f:
            meta = yaml.safe_load(f)
        metas.append(meta)
    return metas


def filter_by(dicts: Iterable[Dict[str, Any]], **filters: Any) -> Iterable[Dict[str, Any]]:
    return [d for d in dicts if all(str(d.get(k, '').lower()) == str(v).lower() for k, v in filters.items())]


def collect_by(dicts: Iterable[Dict[str, Any]], *keys: str) -> Dict[str, Iterable[Dict[str, Any]]]:
    result = defaultdict(list)
    for d in dicts:
        result[','.join((str(d.get(k, '')) for k in keys))].append(d)
    result.default_factory = None
    return result
