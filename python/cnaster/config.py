import json

class JSONConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, JSONConfig(v))
            else:
                setattr(self, k, v)

    def __str__(self):
        def to_dict(obj):
            if isinstance(obj, JSONConfig):
                return {k: to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        return json.dumps(to_dict(self), indent=4)

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as f:
            # Remove comments if present (JSON standard does not allow them)
            lines = [line for line in f if not line.strip().startswith("//")]
            d = json.loads("".join(lines))
        return cls(d)