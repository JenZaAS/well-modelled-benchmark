"""Contenders owned by the tabular foundation models workstream."""

from .models.kumo_tabular.contender import KumoTabularContender
from .models.tabum_v11.contender import TabUMContender

CONTENDERS = [KumoTabularContender, TabUMContender]
