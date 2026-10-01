# Dynamic Range Quantization for Vision-Based Parking Detection on Resource-Constrained Edge Infrastructure

Code and experiments for my ICORIS 2026 paper on compressing CNNs for parking-slot
occupancy detection on edge hardware.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Leiva230-2/edge-quantization-pklot/blob/main/NOTEBOOK_NAME.ipynb)

> **Aviel Aquino**, Ivan Sebastian Edbert, Jayson Prasada Siswoyo, Samson Ndruru.
> *Dynamic Range Quantization for Vision-Based Parking Detection on
> Resource-Constrained Edge Infrastructure.*
> International Conference on Cybernetics and Intelligent Systems (ICORIS) 2026.
> IEEE · Scopus-indexed.
>
> 📄 Paper: `[Is still being published]`

---

## What this is about

Smart parking systems need to run vision models on cheap hardware — a Raspberry
Pi next to a camera, not a GPU server. That means compressing the model. The
question is what you lose when you do, and whether the speedups you measure on
your laptop actually appear on the device.

This work benchmarks three compression methods on **MobileNetV3-Large**, trained
on the **PKLot** dataset, and evaluates them separately across sunny, overcast
and rainy conditions rather than reporting a single averaged accuracy.

## Headline results

| Method | Accuracy | Size | Notes |
|---|---|---|---|
| FP32 baseline | 96.50% | — | Reference |
| **Dynamic-range PTQ** | **96.78%** | **3.64× smaller** | No significant accuracy change (McNemar's exact test, p = 0.37) |
| QAT | 98.06% | 3.64× smaller | Best accuracy, but converged in only 3 of 4 runs |
| Unstructured pruning (20%) | 98.50% | No reduction | No size or latency benefit |
| Unstructured pruning (40%) | 88% | No reduction | Accuracy collapse |

**Post-training quantization is the practical choice.** It matches full-precision
accuracy, needs no retraining, and converges reliably.

### The finding that surprised us

Quantization speedups **reverse direction depending on the instruction set**:

| Architecture | Effect of quantization |
|---|---|
| x86-64 (AVX-512) | **1.53× slower** |
| ARM64 | **1.28× faster** |

Highly optimised FP32 SIMD paths on x86-64 can outrun INT8 execution, while ARM64
benefits from quantized integer arithmetic. The practical implication: **measure
compression gains on your target hardware, not on your development machine.**

### A failure mode worth documenting

The QAT model converged at 98.06% but dropped to **64.89%** after TFLite export.
The cause was double quantization during conversion — a defect in the export path,
not in training. If you're using this pipeline, validate accuracy *after* export,
not just after training.

## Caveat on the ARM64 numbers

ARM64 benchmarks were run on Apple Silicon, which includes the `asimddp`
dot-product extension. A Raspberry Pi 4 (Cortex-A72) lacks it, so the direction
of the result should hold but the magnitude of the speedup is likely overstated
for that class of device.

## Dataset

[PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) — parking lot
images from the Federal University of Paraná, labelled by occupancy and captured
across varied weather conditions. Not included in this repository; download it
from the source.

## Running the experiments

The notebooks are written for Google Colab and open directly via the badge above.
To run locally you'll need Python 3.10+, TensorFlow, and a Jupyter environment.

Latency benchmarks are hardware-dependent — your numbers will differ. The
comparison *between* precisions on the *same* machine is the meaningful signal,
not the absolute milliseconds.

## Contact

Aviel Aquino — Computer Science, BINUS University
[avielaquino.vercel.app](https://avielaquino.vercel.app) ·
[LinkedIn](https://www.linkedin.com/in/avielaquino/)
