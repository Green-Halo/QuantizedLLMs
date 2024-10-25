# LLaMA3-8B-Instruct Quantization: Energy Efficiency and Performance Trade-offs

## Overview
This study aims to evaluate the **LLaMA3-8B-Instruct** model under 4-bit and 8-bit Post-Training Quantization (PTQ). Our focus is on **energy efficiency** and **performance** across different natural language processing (NLP) tasks, quantifying the trade-offs between reduced model precision and its impact on accuracy, energy consumption, and resource utilization.

Our analysis explores whether quantization, while improving energy efficiency, affects model accuracy across NLP tasks. We assess three primary NLP task types:
1. **Sentiment Analysis (SA)**
2. **Sentence Pair Semantic Similarity (SPS)**
3. **Natural Language Inference (NLI)**

## Requirements

### Prerequisites
- **Python 3**: Ensure Python 3 is installed on your system. [Download Python](https://www.python.org/downloads/)

### Installation 
1. **Set Up Power Measurement Tools**  
   Install `pyJoules` for energy measurements following instructions from the [pyJoules Repository](https://pypi.org/project/pyJoules/).

2. **Set Up GPU Monitoring Tools**  
   Install `pynvml` to monitor GPU utilization. Use the following command:
   ```bash
   pip install pynvml

