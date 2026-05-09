"""Reglas de negocio del dominio macro: taxonomía + enriquecimiento + importance.

Compartido por ``application`` (orquestadores) e ``infrastructure`` (cuando
se necesita acceso a patterns para retrieval).
"""

from . import enrichment, importance_rules, taxonomy

__all__ = ["taxonomy", "enrichment", "importance_rules"]
