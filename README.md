# Pay Attention

<!-- Can be lower -->
[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.13-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Pay Attention is a PyTorch library containing efficient implementations of various attention mechanisms.

## Overview

The goal of this library is to provide a user-friendly and efficient implementation of attention mechanisms for developers and researchers. It currently includes a collection of attention mechanisms, and serves as a centralized hub for exploring efficient attention computation.

## Available functions

Pay Attention currently implements the following attention mechanisms:

- [x] Standard attention (`standard`)
- [x] pytorch 2.x efficient-attention () <!-- TODO add name -->
- [x] xformers (`xformers`)
- [x] Batch/sequence chunked attention (`batch_chunked`, `sequence_chunked`, `batch_and_sequence_chunked`)
- [x] Memory-efficient attention (`memory_efficient`) [a.k.a sub-quadratic attention] (WIP)
- [ ] ToME (`tome`) (WIP)

The implementation has been tested with Python 3.9, and PyTorch 1.13.0.

# Installation

To install Pay Attention, simply clone this repository to your local machine:

```bash
pip install -U git+https://github.com/tfernd/pay-attention.git
```

# Usage Examples

Pay Attention provides a smart-attention method that automatically selects the appropriate attention mechanism depending on the available VRAM and whether the xformers library is installed. If xformers is installed, it will use `xformers_attention` for maximum efficiency. Otherwise, it will use the full `standard_attention`, or chunk the keys, queries, and values in the batch and sequence dimension to reduce VRAM usage using `chunked_attention`.

```python
from pay_attention import attention

out = attention(q, k, v)
```

Alternatively, you can use any of the individual attention modules directly:
```python
from pay_attention.modules import standard_attention, chunked_attention, xformers_attention, memory_efficient_attention
```

Additionally, you can check the memory usage of each attention method using the corresponding `_memory` function:
```python
from pay_attention.modules import standard_attention_memory, chunked_attention_memory, xformers_attention_memory, memory_efficient_attention_memory
```

By using these methods and functions, you can easily integrate efficient attention computation into your PyTorch projects.

# Project Status

Pay Attention is currently in active development and is being actively maintained. Our goal is to continue to add new attention mechanisms and optimize existing ones to make the library as useful and efficient as possible for developers and researchers.

In the near future, we plan to add provide more comprehensive benchmarks and documentation to help users understand the performance and use cases for each attention mechanism.

We welcome contributions from the community, including bug reports, feature requests, and pull requests. If you encounter any issues with Pay Attention or have any suggestions for future development, please don't hesitate to submit an issue or pull request on GitHub. We encourage collaboration and contributions from the community, and will do our best to respond to any feedback or suggestions in a timely manner.

# License

This project is licensed under the MIT License. See the LICENSE file for more information.