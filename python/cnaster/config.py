import json


class JSONConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = JSONConfig(v)

            setattr(self, k, v)

    def __iter__(self):
        return iter(xx for xx in dir(self) if (xx != "from_file") and not xx.startswith("_") )

    def __str__(self):
        return json.dumps(self, indent=4)

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as f:
            # Remove comments if present (JSON standard does not allow them)
            lines = [line for line in f if not line.strip().startswith("//")]
            d = json.loads("".join(lines))
        return cls(d)
