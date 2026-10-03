"""Combine the graph adapter with an explicitly activated decision audit."""
from decision_audit import DecisionAuditWorker
from uniform_graphs import UniformGraphWorker


class ClosureGraphWorker(UniformGraphWorker, DecisionAuditWorker):
    pass
