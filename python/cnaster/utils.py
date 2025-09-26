import logging

logger = logging.getLogger(__name__)


def merge_dicts(first, second):
    merged = first.copy()
    
    for k, v in second.items():
        if k in merged:
            logger.warning(
                f"Key clash on '{k}': overwriting value {merged[k]} with {v}"
            )
        merged[k] = v
    return merged
