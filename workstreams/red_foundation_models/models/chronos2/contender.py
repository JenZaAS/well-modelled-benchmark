"""Station wrapper for Amazon Chronos-2."""
from ...depth_model import DepthForecaster
from .backend import load


class Chronos2(DepthForecaster):
    name = "red_chronos2"
    licence_name = "Apache-2.0"
    licence_url = "https://huggingface.co/amazon/chronos-2"
    licence_restrictions = "Retain the licence and applicable notices; mark modified files."
    load_backend = staticmethod(load)
