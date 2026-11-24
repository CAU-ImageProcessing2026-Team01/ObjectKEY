#include "vision/MaxFlow.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace objectkey::vision
{
FlowGraph::FlowGraph(int n) : head_(size_t(std::max(0, n)), -1)
{
    if (n < 2)
        throw std::runtime_error("Invalid flow graph size.");
}
void FlowGraph::reserve(size_t arcs)
{
    edges_.reserve(arcs);
}
void FlowGraph::addEdge(int from, int to, double capacity, double reverse)
{
    if (from < 0 || to < 0 || from >= int(head_.size()) || to >= int(head_.size()) || from == to ||
        !std::isfinite(capacity) || !std::isfinite(reverse) || capacity < 0 || reverse < 0)
        throw std::runtime_error("Invalid graph edge.");
    edges_.push_back({to, head_[from], capacity});
    head_[from] = int(edges_.size()) - 1;
    edges_.push_back({from, head_[to], reverse});
    head_[to] = int(edges_.size()) - 1;
}
double FlowGraph::maxFlow(int source, int sink)
{
    const int n = int(head_.size());
    constexpr double eps = 1e-9;
    if (source < 0 || sink < 0 || source >= n || sink >= n || source == sink)
        throw std::runtime_error("Invalid graph terminals.");
    std::vector<double> excess(n, 0);
    std::vector<int> height(n, 0), current = head_;
    std::vector<unsigned char> active(n, 0);
    std::deque<int> queue;
    auto enqueue = [&](int v) {
        if (v != source && v != sink && !active[v] && excess[v] > eps)
        {
            active[v] = 1;
            queue.push_back(v);
        }
    };
    height[source] = n;
    for (int e = head_[source]; e >= 0; e = edges_[e].next)
    {
        double f = edges_[e].capacity;
        if (f <= 0)
            continue;
        edges_[e].capacity = 0;
        edges_[e ^ 1].capacity += f;
        excess[edges_[e].to] += f;
        excess[source] -= f;
    }
    // Reverse BFS produces residual distances to each terminal. Excess with
    // no path to the sink is returned to the source through reverse arcs.
    auto globalRelabel = [&]() {
        std::fill(height.begin(), height.end(), n + 1);
        height[sink] = 0;
        std::vector<int> bfs{sink};
        for (size_t i = 0; i < bfs.size(); i++)
        {
            int u = bfs[i];
            for (int e = head_[u]; e >= 0; e = edges_[e].next)
            {
                int v = edges_[e].to;
                if (v != source && height[v] == n + 1 && edges_[e ^ 1].capacity > eps)
                {
                    height[v] = height[u] + 1;
                    bfs.push_back(v);
                }
            }
        }
        // Also compute residual distances back to the source. Leaving every
        // sink-unreachable pixel on one height plateau makes large image
        // grids needlessly circulate excess before it can return to source.
        // Vertices reaching neither terminal retain 2*n, so all residual
        // arcs preserve h(u) <= h(v)+1 even in disconnected directed graphs.
        for (auto &value : height)
            if (value == n + 1)
                value = 2 * n;
        height[source] = n;
        bfs.clear();
        bfs.push_back(source);
        std::vector<unsigned char> visited(n, 0);
        visited[source] = 1;
        for (size_t i = 0; i < bfs.size(); i++)
        {
            int u = bfs[i];
            for (int e = head_[u]; e >= 0; e = edges_[e].next)
            {
                int v = edges_[e].to;
                if (!visited[v] && height[v] >= n + 1 && edges_[e ^ 1].capacity > eps)
                {
                    visited[v] = 1;
                    height[v] = height[u] + 1;
                    bfs.push_back(v);
                }
            }
        }
        current = head_;
        queue.clear();
        std::fill(active.begin(), active.end(), static_cast<unsigned char>(0));
        for (int v = 0; v < n; v++)
            enqueue(v);
    };
    globalRelabel();
    size_t work = 0;
    const size_t interval = std::max<size_t>(1, edges_.size() * 4);
    while (!queue.empty())
    {
        int u = queue.front();
        queue.pop_front();
        active[u] = 0;
        while (excess[u] > eps)
        {
            int e = current[u];
            if (e < 0)
            {
                int minimum = 2 * n;
                for (int j = head_[u]; j >= 0; j = edges_[j].next)
                {
                    work++;
                    if (edges_[j].capacity > eps)
                        minimum = std::min(minimum, height[edges_[j].to]);
                }
                if (minimum >= 2 * n)
                    throw std::runtime_error("Disconnected flow excess.");
                height[u] = minimum + 1;
                current[u] = head_[u];
                continue;
            }
            work++;
            auto &edge = edges_[e];
            if (edge.capacity > eps && height[u] == height[edge.to] + 1)
            {
                double f = std::min(excess[u], edge.capacity);
                edge.capacity -= f;
                edges_[e ^ 1].capacity += f;
                excess[u] -= f;
                excess[edge.to] += f;
                enqueue(edge.to);
            }
            else
                current[u] = edge.next;
        }
        if (work > interval)
        {
            globalRelabel();
            work = 0;
        }
    }
    return excess[sink];
}
std::vector<unsigned char> FlowGraph::sourceSide(int source) const
{
    if (source < 0 || source >= int(head_.size()))
        throw std::runtime_error("Invalid source vertex.");
    std::vector<unsigned char> reached(head_.size(), 0);
    std::vector<int> queue{source};
    reached[source] = 1;
    for (size_t i = 0; i < queue.size(); i++)
        for (int e = head_[queue[i]]; e >= 0; e = edges_[e].next)
            if (edges_[e].capacity > 1e-9 && !reached[edges_[e].to])
            {
                reached[edges_[e].to] = 1;
                queue.push_back(edges_[e].to);
            }
    return reached;
}
} // namespace objectkey::vision
