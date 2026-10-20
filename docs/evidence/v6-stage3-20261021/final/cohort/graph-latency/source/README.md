# Graph inference validation project

This is an additional project input for evaluating intake and operator
optimization on irregular gather/scatter work. It uses a two-layer mean
GraphSAGE encoder with FP32 weights, directed edges, self features, and
ReLU between layers. Weights, features and edges are seeded random fixtures.
There is no pretrained-model or downstream graph-accuracy claim.

```sh
python infer.py --workload standard --output /tmp/graph-reference
```

The timed task is a complete graph encoder forward with GPU-resident inputs.
Model/input construction, quality checking and statistics queries are outside
timing. Preserve the complete output tensor with the tolerances in
`workloads.json`, the graph, model dimensions, FP32 arithmetic and disabled TF32.
Inference has no cross-invocation state. Duplicate directed edges contribute
individually to mean aggregation; nodes without incoming edges use zero neighbor
features.

The standard graph has 4,096 nodes and 65,536 edges. `unseen_graph` is a separate
held-out graph with 7,168 nodes and 143,360 edges. Both are constructed through
the same source path. `workloads.json` declares latency and allocated-memory
goals; the memory goal preserves at least 95% of original throughput.

Only PyTorch is required. These source files contain no framework adapter,
operator groups, candidate implementations or optimization hints.
