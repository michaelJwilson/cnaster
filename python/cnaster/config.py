import json

class JSONConfig:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return json.dumps(self, indent=4)

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as f:
            # Remove comments if present (JSON standard does not allow them)
            lines = [line for line in f if not line.strip().startswith("//")]
            d = json.loads("".join(lines))
        return cls(d)