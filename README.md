# higgs-cuda-kernel

Custom CUDA normalization kernel benchmarked against PyTorch CPU baseline on CERN Higgs Boson data, with a CUDA Streams pipeline and PyTorch MLP classifier.

---

## Results

Timings vary across runs and hardware. Numbers below are from a single Colab T4 session and are representative of the order of magnitude, not exact reproducible values.

| Step | Typical Time | Approximate Speedup |
|---|---|---|
| PyTorch CPU normalization (250k x 30) | ~25 ms | baseline |
| CUDA kernel (PyTorch GPU), compute only | ~0.5 ms | ~50x |
| CUDA kernel, with data transfer | ~14 ms | ~1.8x |
| CUDA Streams + pinned memory | ~15 ms | ~1.7x |
| MLP Validation AUC-ROC | 0.9090 | |

**Key finding:** The kernel itself is roughly 50x faster than CPU. Data transfer (CPU to GPU and back) dominates total time at this dataset size. This is why streams exist, and why they matter more at larger scales where compute and transfer time are comparable.

---

## Why This Project

PyTorch's built-in normalization on tabular data runs on CPU by default. On large datasets processed repeatedly (once per training epoch, once per inference batch), this becomes a measurable bottleneck. This project:

1. Measures the baseline CPU normalization time
2. Writes a custom CUDA kernel with a specific optimization strategy (2D thread grid + coalesced memory access)
3. Adds CUDA Streams to pipeline data transfer with computation
4. Trains a PyTorch MLP classifier on the normalized data and reports AUC-ROC

---

## Architecture

```
Raw CERN Data (250k collision events)
   -> Clean: replace -999.0 missing values with per-feature mean
   -> CPU baseline: PyTorch z-score normalization (timed)
   -> CUDA kernel: custom 2D grid normalization (timed)
   -> CUDA Streams: async pipelining with pinned memory (timed)
   -> PyTorch MLP: 30 -> 128 -> 64 -> 1 (sigmoid)
   -> AUC-ROC evaluation
```

---

## Implementation Decisions

**2D thread grid (x = features, y = rows)**  
Each thread handles one (row, feature) pair. The x-axis covers features so that threads in the same warp access consecutive features in the same row, which are consecutive memory addresses in row-major layout. This is coalesced memory access, the fastest access pattern on GPU global memory.

**Coalesced vs strided access**  
If x covered rows instead of features, warp threads would access the same feature across different rows, with addresses that are F elements apart in memory. Strided access significantly reduces memory throughput.

**CUDA Streams with pinned memory**  
Streams overlap data transfer of batch 2 with kernel execution on batch 1. This requires pinned (page-locked) host memory. Regular numpy arrays use pageable memory and cudaMemcpyAsync silently falls back to synchronous transfer, giving stream overhead with no benefit. At 250k rows the kernel (~0.5ms) is too fast relative to transfer (~13ms) for meaningful overlap. At larger scales, where compute and transfer are comparable in duration, streams provide real throughput gains.

**Median for benchmarking**  
CPU timing has high variance from OS scheduling and cache state. Median of 10 runs is more robust than mean since one slow outlier does not skew the result. Standard practice in systems benchmarking.

**Mean and std computed on CPU**  
Mean and std are computed once before training. Normalization runs every epoch and every inference batch. The repeated operation (normalization) is the correct optimization target, not the one-time setup.

---

## Dataset

**Kaggle Higgs Boson Machine Learning Challenge (2014)**  
CERN ATLAS experiment, simulated proton-proton collision events.

- 250,000 events, 30 kinematic features (DER_* derived, PRI_* primitive)
- Label: s = signal (Higgs boson decay), b = background noise
- Missing values encoded as -999.0 (measurement impossible, e.g. jet features when no jet fired)
- Class balance: 66% background, 34% signal

Download: [kaggle.com/competitions/higgs-boson](https://kaggle.com/competitions/higgs-boson)  
Accept competition rules before downloading.

---

## Setup and Usage

**Requirements:** Google Colab with T4 GPU runtime

**First time only:**
1. Download training.csv from the Kaggle link above
2. Run Section 1 of the notebook (Kaggle CLI download to Google Drive)

**Every session:**
1. Open notebook in Colab, Runtime > Change Runtime Type > T4 GPU
2. Run the 5 Session Setup cells at the top in order
3. Run any section independently after that

The setup cells handle: GPU check, Drive mount, data loading, CUDA compilation, library loading.

---

## Files

| File | Description |
|---|---|
| `higgs_pipeline.ipynb` | Main notebook, full pipeline with explanations |
| `normalize.cu` | CUDA kernel: single-stream z-score normalization |
| `normalize_streams.cu` | CUDA kernel: two-stream async pipelining version |
| `README.md` | This file |

Compiled .so binaries are not committed. They are generated at runtime by Session Setup Cell 4 in the notebook.

---

## Connection to Larger Work

This project seeds a larger streaming inference pipeline (Kafka + Spark/Dask + Kubernetes) built on the same CERN Open Data. Same dataset, same preprocessing direction, extended into a full distributed ML system.

## Acknowledgements
> Documentation and README presentation cleaned up with assistance from Claude (Anthropic).