import json
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ModelParameters:
    cna_rate: float

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelParameters":
        return ModelParameters(cna_rate=d["cna_rate"])


@dataclass
class CloneInitialization:
    min_clone_read_coverage: int
    min_clone_snp_coverage: int

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CloneInitialization":
        return CloneInitialization(
            min_clone_read_coverage=d["min_clone_read_coverage"],
            min_clone_snp_coverage=d["min_clone_snp_coverage"],
        )


@dataclass
class Config:
    model_parameters: ModelParameters
    clone_initialization: CloneInitialization

    @staticmethod
    def from_file(path: str) -> "Config":
        with open(path, "r") as f:
            d = json.load(f)
        return Config(
            model_parameters=ModelParameters.from_dict(d["model_parameters"]),
            clone_initialization=CloneInitialization.from_dict(
                d["clone_initialization"]
            ),
        )


@dataclass
class PhasingConfig:
    switch_rate: float

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PhasingConfig":
        return PhasingConfig(switch_rate=d["switch_rate"])


@dataclass
class SimConfig:
    segment_kbp: int
    trans_rate: float
    copy_num_states: List[str]
    rdr_dispersion: float
    baf_dispersion: float
    phasing: PhasingConfig

    @staticmethod
    def from_file(path: str) -> "SimConfig":
        with open(path, "r") as f:
            d = json.load(f)
        return SimConfig(
            segment_kbp=d["segment_kbp"],
            trans_rate=d["trans_rate"],
            copy_num_states=d["copy_num_states"],
            rdr_dispersion=d["rdr_dispersion"],
            baf_dispersion=d["baf_dispersion"],
            phasing=PhasingConfig.from_dict(d["phasing"]),
        )
