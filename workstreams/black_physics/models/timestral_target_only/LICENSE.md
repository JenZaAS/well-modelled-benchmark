# Timestral-5M target-only control

The [upstream inference code](https://github.com/mistralai/timestral) is Apache-2.0. The
[checkpoint](https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE)
uses **Modified MIT**: retain the attribution and permission notice; you may not exercise the licence if the
global consolidated monthly revenue of your company or employer exceeded USD 20 million in the preceding month.
This wrapper (MIT) reuses red's adapter and does not bundle weights; fetch them into the cache as described in
red's [backend.py](../../../red_foundation_models/models/timestral/backend.py).
