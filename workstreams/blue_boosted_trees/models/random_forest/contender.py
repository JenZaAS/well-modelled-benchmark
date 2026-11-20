"""Random forest from the ML study, registered for the benchmark (base features only)."""
from workstreams.blue_boosted_trees.ml_study.registered import RegisteredML


class BlueMLRf(RegisteredML):
    """Random forest (60 trees) on the other logs + depth. Point predictions only. Only the base feature set is
    registered: with trend features a single fit takes about 7 minutes."""
    name = "blue_ml_rf"
    kind = "rf"
    licence_restrictions = "none for this code; uses scikit-learn (BSD-3-Clause)"
