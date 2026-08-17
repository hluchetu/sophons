"""Builder and static validation for Sophons graphs."""

from __future__ import annotations

from sophons.multiagent.graph.types import (
    EdgeCondition,
    GraphEdge,
    GraphExecutor,
    GraphNode,
    InputMapper,
    JoinPolicy,
)


class GraphBuilder:
    """Build a validated directed graph."""

    def __init__(self, *, id: str = "default_graph") -> None:
        if not id.strip():
            raise ValueError("Graph ID cannot be empty.")
        self._id = id
        self.nodes: dict[str, GraphNode] = {}
        self.edges: set[GraphEdge] = set()
        self.entry_points: set[str] = set()
        self._max_node_executions: int | None = None
        self._execution_timeout: float | None = None
        self._node_timeout: float | None = None
        self._max_concurrency: int | None = None

    def add_node(
        self,
        executor: GraphExecutor,
        node_id: str | None = None,
        *,
        join: JoinPolicy = "all",
        input_mapper: InputMapper | None = None,
        timeout: float | None = None,
    ) -> GraphNode:
        if any(node.executor is executor for node in self.nodes.values()):
            raise ValueError("Each graph node must use a unique executor instance.")
        if join not in ("all", "any"):
            raise ValueError("Join policy must be 'all' or 'any'.")
        if timeout is not None and timeout <= 0:
            raise ValueError("Node timeout must be greater than zero.")

        resolved_id = node_id or getattr(executor, "id", None)
        resolved_id = resolved_id or getattr(executor, "name", None)
        resolved_id = resolved_id or f"node_{len(self.nodes)}"
        if not isinstance(resolved_id, str) or not resolved_id.strip():
            raise ValueError("Node ID must be a non-empty string.")
        if resolved_id in self.nodes:
            raise ValueError(f"Node {resolved_id!r} already exists.")

        node = GraphNode(
            node_id=resolved_id,
            executor=executor,
            join=join,
            input_mapper=input_mapper,
            timeout=timeout,
        )
        self.nodes[resolved_id] = node
        return node

    def add_edge(
        self,
        source: str | GraphNode,
        target: str | GraphNode,
        condition: EdgeCondition | None = None,
        *,
        label: str | None = None,
    ) -> GraphEdge:
        source_id = self._resolve_node_id(source, "Source")
        target_id = self._resolve_node_id(target, "Target")
        edge = GraphEdge(source_id, target_id, condition, label)
        if edge in self.edges:
            raise ValueError(f"Edge {source_id!r} -> {target_id!r} already exists.")
        self.edges.add(edge)
        return edge

    def set_entry_point(self, node: str | GraphNode) -> GraphBuilder:
        self.entry_points.add(self._resolve_node_id(node, "Entry point"))
        return self

    def set_graph_id(self, graph_id: str) -> GraphBuilder:
        if not graph_id.strip():
            raise ValueError("Graph ID cannot be empty.")
        self._id = graph_id
        return self

    def set_max_node_executions(self, maximum: int | None) -> GraphBuilder:
        if maximum is not None and maximum < 1:
            raise ValueError("Maximum node executions must be at least 1.")
        self._max_node_executions = maximum
        return self

    def set_execution_timeout(self, seconds: float | None) -> GraphBuilder:
        if seconds is not None and seconds <= 0:
            raise ValueError("Execution timeout must be greater than zero.")
        self._execution_timeout = seconds
        return self

    def set_node_timeout(self, seconds: float | None) -> GraphBuilder:
        if seconds is not None and seconds <= 0:
            raise ValueError("Node timeout must be greater than zero.")
        self._node_timeout = seconds
        return self

    def set_max_concurrency(self, maximum: int | None) -> GraphBuilder:
        if maximum is not None and maximum < 1:
            raise ValueError("Maximum concurrency must be at least 1.")
        self._max_concurrency = maximum
        return self

    def build(self) -> "Graph":
        from sophons.multiagent.graph.runtime import Graph

        if not self.nodes:
            raise ValueError("Graph must contain at least one node.")

        entries = set(self.entry_points)
        if not entries:
            targets = {edge.target for edge in self.edges}
            entries = set(self.nodes) - targets
        if not entries:
            raise ValueError(
                "No entry point could be detected. Cyclic graphs need an explicit entry point."
            )

        if self._has_cycle() and (
            self._max_node_executions is None
            and self._execution_timeout is None
        ):
            raise ValueError(
                "Cyclic graphs require max node executions or an execution timeout."
            )

        return Graph(
            id=self._id,
            nodes=dict(self.nodes),
            edges=frozenset(self.edges),
            entry_points=frozenset(entries),
            max_node_executions=self._max_node_executions,
            execution_timeout=self._execution_timeout,
            node_timeout=self._node_timeout,
            max_concurrency=self._max_concurrency,
        )

    def _resolve_node_id(self, node: str | GraphNode, label: str) -> str:
        node_id = node if isinstance(node, str) else node.node_id
        if node_id not in self.nodes:
            raise ValueError(f"{label} node {node_id!r} was not found.")
        if isinstance(node, GraphNode) and self.nodes[node_id] is not node:
            raise ValueError(f"{label} node belongs to a different builder.")
        return node_id

    def _has_cycle(self) -> bool:
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            outgoing[edge.source].append(edge.target)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False
            visiting.add(node_id)
            if any(visit(target) for target in outgoing[node_id]):
                return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        return any(visit(node_id) for node_id in self.nodes)
