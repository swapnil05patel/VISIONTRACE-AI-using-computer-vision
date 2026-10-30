from .base import DatasetSample, ValidationReport, validate_and_list
from .market1501 import parse_market1501
from .mars import parse_mars
from .msmt17 import parse_msmt17

PARSERS = {
    "market1501": parse_market1501,
    "mars": parse_mars,
    "msmt17": parse_msmt17,
}

__all__ = ["DatasetSample", "ValidationReport", "validate_and_list", "PARSERS"]
