#pragma once
#include <cstddef>
#include <vector>
namespace objectkey::vision
{
// Residual graph with paired arcs. No external graph/optimization library.
class FlowGraph
{
  public:
    explicit FlowGraph(int vertices);
    void reserve(size_t arcs);
    void addEdge(int from, int to, double capacity, double reverseCapacity = 0);
    double maxFlow(int source, int sink);
    std::vector<unsigned char> sourceSide(int source) const;

  private:
    struct Edge
    {
        int to, next;
        double capacity;
    };
    std::vector<int> head_;
    std::vector<Edge> edges_;
};
} // namespace objectkey::vision
