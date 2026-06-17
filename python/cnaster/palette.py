import seaborn as sns

def get_ordered_acn(mode="joint"):
    """
    Returns an immutable tuple of ordered Allele Copy Number states.
    
    Args:
        mode (str): "joint" for (A, B) tuples, "independent" for single integers.
    """
    if mode == "independent":
        return (0, 1, 2, 3, 4, 5, 6, "7+")
    
    # Default: joint
    return (
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (2, 2),
        (3, 1),
        (4, 0),
        (3, 2),
        (4, 1),
        (5, 0),
        (3, 3),
        (4, 2),
        (5, 1),
        (6, 0),
    )


def get_full_palette(palette_name="chisel_joint"):
    """
    Returns a dictionary mapping allele copy numbers to colors, 
    and the ordered list of states.
    
    Available custom palettes:
    - "chisel_joint": Maps (A, B) tuples.
    - "chisel_independent": Maps individual integer copies.
    """
    palette = {}

    if palette_name == "chisel_joint":
        ordered_acn = get_ordered_acn(mode="joint")
        palette.update({(0, 0): "darkblue"})
        palette.update({(1, 0): "lightblue"})
        palette.update({(1, 1): "lightgray", (2, 0): "dimgray"})
        palette.update({(2, 1): "lightgoldenrodyellow", (3, 0): "gold"})
        palette.update({(2, 2): "navajowhite", (3, 1): "orange", (4, 0): "darkorange"})
        palette.update({(3, 2): "salmon", (4, 1): "red", (5, 0): "darkred"})
        palette.update({(3, 3): "plum", (4, 2): "orchid", (5, 1): "purple", (6, 0): "indigo"})
        
    elif palette_name == "chisel_independent":
        ordered_acn = get_ordered_acn(mode="independent")
        # Aligned to visually match the dominant colors of the joint palette
        palette.update({
            0: "darkblue",              # Matches (0,0) deletion
            1: "lightgray",             # Matches (1,1) normal diploid
            2: "lightgoldenrodyellow",  # Matches baseline amplification
            3: "orange",                
            4: "red",                   
            5: "darkred",               
            6: "purple",                
            "7+": "indigo"              
        })
        
    else:
        # Dynamic fallback to seaborn categorical palettes (e.g., "tab20b")
        ordered_acn = get_ordered_acn(mode="joint")
        colors = sns.color_palette(palette_name, len(ordered_acn)).as_hex()
        palette = dict(zip(ordered_acn, colors))

    # Safely inject the default fallback color to prevent KeyErrors
    if "default" not in palette:
        palette["default"] = "lightgray"

    return palette, ordered_acn