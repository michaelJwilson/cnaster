import seaborn as sns


def get_ordered_acn(mode="joint"):
    """
    Returns an immutable tuple of ordered Allele Copy Number states.

    Args:
        mode (str): "joint" for (A, B) tuples, "independent" for single integers.
    """
    if mode == "single":
        return (0, 1, 2, 3, 4, 5, 6, "7+")

    # NB fallback to joint as default
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
    Returns a dictionary mapping copy states to colors,
    according to an encoding scheme, and the ordered list
    of states.

    Available custom palettes:
    - "chisel_joint": maps (A, B) tuples to colors.
    - "chisel_independent": maps integer copies, i.e. A, to colors.
    """
    palette = {}

    # NB default joint encoding.
    if palette_name in ("chisel", "chisel_joint"):
        ordered_acn = get_ordered_acn(mode="joint")

        palette.update({(0, 0): "darkblue"})
        palette.update({(1, 0): "lightblue"})
        palette.update({(1, 1): "lightgray"})
        palette.update({(2, 0): "dimgray"})
        palette.update({(2, 1): "lightgoldenrodyellow"})
        palette.update({(3, 0): "gold"})
        palette.update({(2, 2): "navajowhite"})
        palette.update({(3, 1): "orange"})
        palette.update({(4, 0): "darkorange"})
        palette.update({(3, 2): "salmon"})
        palette.update({(4, 1): "red"})
        palette.update({(5, 0): "darkred"})
        palette.update({(3, 3): "plum"})
        palette.update({(4, 2): "orchid"})
        palette.update({(5, 1): "purple"})
        palette.update({(6, 0): "indigo"})

    elif palette_name == "chisel_single":
        ordered_acn = get_ordered_acn(mode="chisel_single")

        '''
        palette.update(
            {
                0: "white", 
                1: "lightblue",
                2: "lightgoldenrodyellow",
                3: "yellow",
                4: "salmon",
                5: "orange",
                6: "red",
                "7+": "darkred",
            }
        )
        '''

        palette.update({
            0: "white",        # Negative space for complete loss
            1: "#ADD8E6",    # Very pale yellow (Baseline)
            2: "#fed976",    # Light yellow-orange
            3: "#feb24c",    # Orange
            4: "#fd8d3c",    # Dark orange
            5: "#f03b20",    # Red
            6: "#bd0026",    # Dark red
            "7+": "#660013", # Deep burgundy
        })

        palette["default"] = "lightblue"

    else:
        ordered_acn = get_ordered_acn(mode="joint")
        colors = sns.color_palette(palette_name, len(ordered_acn)).as_hex()
        palette = dict(zip(ordered_acn, colors))

    if "default" not in palette:
        palette["default"] = "lightgray"

    return palette, ordered_acn
