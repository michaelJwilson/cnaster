import seaborn as sns

# TODO immutable?
def get_ordered_acn():
    return [
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
    ]


def get_full_palette(palette="tab20b"):
    colors = [
        "darkblue",
        "lightblue",
        "lightgray",
        "dimgray",
        "lightgoldenrodyellow",
        "gold",
        "navajowhite",
        "orange",
        "darkorange",
        "salmon",
        "red",
        "darkred",
        "plum",
        "orchid",
        "purple",
        "indigo",
    ]

    ordered_acn = get_ordered_acn()
    # TODO HACK
    colors = sns.color_palette("tab20b", len(ordered_acn)).as_hex()
    palette = dict(zip(ordered_acn, colors))

    """
    # TODO
    palette = {}
    palette.update({(0, 0): "darkblue"})
    palette.update({(1, 0): "lightblue"})
    palette.update({(1, 1): "lightgray", (2, 0): "dimgray"})
    palette.update({(2, 1): "lightgoldenrodyellow", (3, 0): "gold"})
    palette.update({(2, 2): "navajowhite", (3, 1): "orange", (4, 0): "darkorange"})
    palette.update({(3, 2): "salmon", (4, 1): "red", (5, 0): "darkred"})
    palette.update(
        {(3, 3): "plum", (4, 2): "orchid", (5, 1): "purple", (6, 0): "indigo"}
    )

    assert palette == new_palette
    """
    return palette, ordered_acn