"""Station wrapper for Mistral Timestral-5M."""
from ...depth_model import DepthForecaster
from .backend import load


class Timestral(DepthForecaster):
    name = "red_timestral"
    licence_name = "Modified MIT (weights); Apache-2.0 (inference code)"
    licence_url = (
        "https://huggingface.co/mistralai/Timestral-5M/blob/"
        "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE"
    )
    licence_restrictions = (
        "Retain attribution and permission notice; separate authorization required if company "
        "or employer global consolidated revenue exceeded USD 20 million in the preceding month."
    )
    load_backend = staticmethod(load)
