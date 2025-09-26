import logging

logger = logging.getLogger(__name__)


def merge_dicts(first, second):
    merged = first.copy()
    collision = False
    
    for k, v in second.items():
        if k in merged:
            collision = True
            logger.warning(
                f"Key clash on '{k}': overwriting value {merged[k]} with {v}"
            )
        merged[k] = v

    if not collision:
        logger.info(f"Safely merged dictionaries with no collisions.")
        
    return merged
