# Parallel GC for CPython: When Does It Actually Help?

**TL;DR**: We built a parallel garbage collector for CPython and ran systematic experiments to understand when it provides benefit. The answer: large heaps (500K+ objects) with networked graph structures. AI/ML workloads with manual GC are the sweet spot, showing 1.3x speedup at scale.

---

## The Question

CPython's garbage collector runs single-threaded. With modern machines having 8-64+ cores, can we speed up GC by parallelizing it?

We implemented a parallel GC using work-stealing deques and ran systematic experiments to find out.

## What We Tested

We designed five experiments to validate our assumptions:

1. **Heap Size**: At what scale does parallel GC become beneficial?
2. **Survivor Rate**: Does the proportion of live objects matter?
3. **Graph Structure**: Do different object graph shapes affect parallelism?
4. **AI/ML Workloads**: Is manual GC with large heaps the ideal use case?
5. **Heap Maturity**: Does a "seasoned" heap behave differently than a fresh one?

---

## Key Findings

### Finding 1: Heap Size Matters Most

Parallel GC has fixed overhead from synchronization barriers, atomic operations, and work-stealing coordination. This overhead must be amortized over enough work.

| Heap Size | Parallel vs Incremental |
|-----------|------------------------|
| 10K objects | 0.84x (slower) |
| 50K objects | 0.74x (slower) |
| 100K objects | 0.86x (slower) |
| **500K objects** | **1.56x (faster)** |
| **1M objects** | **1.43x (faster)** |

**Crossover point: approximately 500K objects.**

Below this threshold, the overhead dominates. Above it, parallelism pays off.

### Finding 2: Graph Structure Has Massive Impact

Not all object graphs parallelize equally well.

| Structure | Speedup | Why |
|-----------|---------|-----|
| Layered (neural-net like) | **1.81x** | Many parallel edges between layers |
| Random | 1.14x | Cross-references enable parallel traversal |
| Linear chain | 0.86x | Sequential by nature |
| Binary tree | 0.53x | Level-by-level dependency |
| DAG | 0.63x | Topological ordering limits parallelism |

Neural network-style structures (fully connected layers) showed the best results. This aligns with our hypothesis about AI/ML workloads.

### Finding 3: AI/ML Workloads Are the Sweet Spot

We simulated PyTorch-like workloads: tensor metadata objects with autograd graph chains, mostly garbage between collections (10% survival rate), manual GC timing.

| Tensor Count | Total Objects | Speedup |
|--------------|--------------|---------|
| 10K tensors | 60K | 0.62x (slower) |
| 50K tensors | 300K | **1.27x** |
| 100K tensors | 600K | **1.12x** |
| 200K tensors | 1.2M | **1.33x** |

At 1.2 million objects, parallel GC completed in 391ms versus 519ms for incremental—a savings of 128ms per collection.

### Finding 4: Low Survival Rate Performed Best (Surprising)

We expected high survival rates to favor parallelism (more marking work to distribute). The data showed the opposite:

| Survival Rate | Speedup |
|--------------|---------|
| 10% | 1.19x |
| 30% | 0.81x |
| 50% | 0.87x |
| 70% | 0.94x |
| 90% | 0.82x |

Possible explanation: fewer live objects means less contention between workers competing to mark the same objects.

### Finding 5: Heap Maturity Shows Mixed Results

We tested heaps after 0, 5, 10, and 20 prior collections. Results were inconsistent (ranging from 0.75x to 1.29x). More investigation needed.

---

## Practical Recommendations

### When to Enable Parallel GC

Parallel GC is beneficial when:

- Heap contains **500K+ objects**
- Object graph has **networked/layered structure** (many cross-references)
- Using **manual GC** (throughput matters more than latency)
- Running **AI/ML workloads** with large tensor graphs

### When to Keep Incremental GC

Stick with the default incremental GC when:

- Heap contains **fewer than 100K objects**
- Object graph is **linear or tree-structured**
- Using **automatic GC** (many small collections)
- Application is **latency-sensitive**

### Example: AI/ML Training Loop

```python
import gc

# Disable automatic GC - we'll call it manually
gc.disable()

# Enable parallel GC with 4 workers
gc.enable_parallel(num_workers=4)

for epoch in range(epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()

    # GC at natural pause point (GPU idle during checkpoint)
    torch.cuda.synchronize()
    gc.collect()  # Uses parallel GC - 1.3x faster at scale
```

---

## Implementation Notes

### Current Threshold Is Too Low

The current threshold (10K objects) triggers parallel GC too aggressively. Based on our findings:

```c
// Current (too aggressive):
const size_t MIN_TOTAL_OBJECTS = 10000;

// Recommended:
const size_t MIN_TOTAL_OBJECTS = 500000;
```

### Work Distribution

Across experiments, we observed moderate work imbalance (workers handling 18-38% of objects each). The work-stealing mechanism compensates effectively—no worker sits idle while others have work.

---

## Open Questions

1. **Larger scale**: How does parallel GC perform at 10M, 50M, 100M objects?
2. **Worker scaling**: Is 4 workers optimal, or do 8/16 workers help at larger scales?
3. **Real workloads**: How does this translate to actual PyTorch training?
4. **Long-running processes**: Does behavior change over 30+ minutes of continuous operation?

---

## Summary

Parallel GC is not a general-purpose improvement. It's a specialized optimization for:

- **Large heaps** (500K+ objects)
- **Networked graph structures** (not linear chains)
- **Manual GC patterns** (AI/ML training loops)

When these conditions are met, expect **1.3-1.8x speedup** over incremental GC.
