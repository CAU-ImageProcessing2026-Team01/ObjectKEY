# Maximum flow and minimum cut

[Back to README](../README.md#how-it-works)

[GrabCut](grabcut.md) expresses segmentation as a graph with a source, a sink and nonnegative edge capacities. A minimum cut separates foreground from background at minimum energy.

## Principle

For a partition $(S,T)$ with source $s\in S$ and sink $t\in T$, the cut capacity is

$$
C(S,T)=\sum_{u\in S,\ v\in T}c(u,v).
$$

The maximum-flow/minimum-cut theorem states

$$
\max_f \lvert f\rvert=\min_{(S,T)}C(S,T).
$$

A residual graph records how much additional flow each edge can carry. Every forward edge has a paired reverse edge so earlier flow can be redirected.

## FIFO push-relabel implementation

Each vertex has excess flow $e(u)$ and height $h(u)$. Initialization raises the source height to the vertex count and saturates its outgoing edges. Nonterminal vertices with positive excess enter a FIFO queue.

An edge is admissible when its residual capacity $c_f(u,v)$ is positive and

$$
h(u)=h(v)+1.
$$

A push sends

$$
\Delta=\min\{e(u),c_f(u,v)\}
$$

and updates both vertices' excess and the paired residual capacities. If no edge is admissible, relabel the vertex:

$$
h(u)\leftarrow 1+\min_{v:\,c_f(u,v)>0}h(v).
$$

The implementation stores adjacency lists in flat arrays and periodically refreshes heights with reverse breadth-first searches. Sink-unreachable vertices receive distances back to the source so their excess can be returned. Processing ends when no active nonterminal vertex remains.

## Recovering the mask

A final traversal follows positive residual capacities from the source. Reachable pixel vertices become foreground; all other pixel vertices become background.

This cut minimizes GrabCut's binary energy for the current color models. GrabCut then updates those models before solving the next graph.

**Code:** `FlowGraph::maxFlow`, `sourceSide` in [MaxFlow.cpp](../src/vision/MaxFlow.cpp).

**References:** [11](../README.md#ref-11), [12](../README.md#ref-12).
