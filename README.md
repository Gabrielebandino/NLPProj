# PoliMillionaire NLP Benchmark

A multimodal, tool-augmented benchmark for playing a *Who Wants To Be a Millionaire*-style trivia game with open-weight language models.

The project evaluates model accuracy, inference speed, retrieval, mathematical reasoning, live web search, and speech recognition under a strict per-question time limit.

## Project Highlights

- Benchmarks four 7B-9B open-weight models:
  - Qwen3.5-9B
  - Llama 3.1 8B Instruct
  - Gemma 2 9B IT
  - Mistral 7B Instruct v0.3
- Compares Hugging Face + bitsandbytes 4-bit inference with llama.cpp + GGUF Q4_K_M.
- Uses CUDA and NVIDIA T4 GPUs for the main experiments.
- Adds specialized pipelines for:
  - Mathematical reasoning with PAL, SymPy, and RAG
  - Wikipedia BM25 retrieval
  - Live search with Exa and Firecrawl
  - Automatic speech recognition with Whisper-small
  - Confidence estimation experiments
- Records per-question answers, correctness, latency, retrieved context, generated code, and tool usage.

## Main Results

The notebook contains results from multiple runs and environments. The primary llama.cpp Q4_K_M comparison reports the following overall accuracies:

| Model | Overall accuracy |
|---|---:|
| Qwen3.5-9B | **79.88%** |
| Gemma-2-9B | 77.18% |
| Llama-3.1-8B | 76.01% |
| Mistral-7B-v0.3 | 65.53% |

Accuracy varies substantially by topic. The strongest documented topic scores include:

- Gemma-2-9B on Philosophy and Psychology: **97.03%**
- Gemma-2-9B on Science and Nature: **92.93%**
- Qwen3.5-9B on Science and Nature: **90.30%**
- Qwen3.5-9B on Philosophy and Psychology: **93.20%**

The live-search experiments are described in the notebook as frequently reaching **15/15** on News and general-knowledge runs. These results are experiment notes rather than a single reproducible aggregate score, so the exact percentages should be taken from the generated CSV logs when available.

## Throughput Results

Measured decoding throughput under the documented 4-bit setup:

| Model | Hugging Face | llama.cpp | Speedup |
|---|---:|---:|---:|
| Mistral-7B-v0.3 | 13.88 tok/s | **46.01 tok/s** | 3.31x |
| Llama-3.1-8B | 13.51 tok/s | 43.52 tok/s | 3.22x |
| Qwen3.5-9B | 10.55 tok/s | 39.23 tok/s | 3.72x |
| Gemma-2-9B | 8.59 tok/s | 33.16 tok/s | 3.86x |

The practical recommendation from the benchmark is:

- **Qwen3.5-9B** for the best balance of accuracy and reasoning robustness.
- **Mistral-7B-Instruct-v0.3** for maximum decoding throughput.
- **llama.cpp + GGUF Q4_K_M** for fast single-stream deployment on a T4.

## Notebook Structure

The main notebook is [`NLP_final.ipynb`](NLP_final.ipynb).

| Section | Purpose |
|---|---|
| 1 | Environment setup, authentication, and game client installation |
| 2 | Benchmark methodology |
| 3 | Quantization background |
| 4 | Shared configuration and helper functions |
| 5 | llama.cpp accuracy benchmark |
| 6 | Hugging Face versus llama.cpp throughput |
| 7 | Results loading, summaries, and charts |
| 8 | vLLM discussion and exclusion rationale |
| 9 | Mathematical tools, PAL, and RAG + PAL |
| 10 | Qwen3-8B LoRA fine-tuning experiment |
| 11-13 | Wikipedia, Exa, Firecrawl, and general-topic search |
| 14 | Whisper-small speech-mode benchmark |
| 15 | Confidence estimation discussion |
| 16-17 | Literature review, next steps, and AI-assisted development |

## Requirements

The original notebook was designed primarily for Kaggle GPU environments with CUDA. A complete run requires:

- Python 3.10+
- NVIDIA GPU with sufficient VRAM, preferably a T4 or better
- CUDA-compatible PyTorch
- Hugging Face access, including a token for gated models such as Llama 3.1
- Credentials for the PoliMillionaire game server
- Optional API keys for Exa and Firecrawl
- Optional Kaggle Secrets for credentials and API keys

Core Python packages include:

```text
accelerate
bitsandbytes
datasets
huggingface_hub
llama-cpp-python
matplotlib
numpy
pandas
rank_bm25
requests
scikit-learn
soundfile
sympy
torch
transformers
```

The notebook installs several dependencies in its setup cells. For the best compatibility, run it in the intended CUDA/Kaggle environment rather than a CPU-only local environment.

## Running the Notebook

The notebook is an ensemble of experiments collected from multiple runs. It can be executed standalone, but running every benchmark end to end is time-consuming and requires external services.

Recommended order:

1. Run **Section 1** to install dependencies and configure authentication.
2. Run **Section 4** to load the shared configuration and helpers.
3. Run **Section 5** for the llama.cpp accuracy pipeline.
4. Run **Section 6** for throughput measurements.
5. Run **Section 7** to load existing CSV results and generate comparisons.
6. Run Sections 9-14 only when the corresponding model, data, API keys, and services are available.

Expected output files include:

- `accuracy_per_question.csv`
- `accuracy_per_question_llamacpp.csv`
- `throughput_results.csv`
- `throughput_results_llamacpp.csv`
- `throughput_comparison.csv`
- `throughput_comparison.png`
- `question_logs_*.csv`
- `asr_speech_results.csv`

## Methodology Notes

The main accuracy pipeline combines:

1. Next-token logits scoring for option IDs `0` through `3`.
2. Short chain-of-thought generation when the logits confidence gap is low.
3. Strict answer parsing and per-question telemetry.

The throughput benchmark uses a fixed prompt, deterministic decoding, one warm-up run, and three timed runs. Both engines generate the same target number of new tokens.

The notebook also tests external tools where model-only reasoning is less reliable. Mathematical questions use generated Python, SymPy, retrieval from NuminaMath-CoT, and option matching. News and general-knowledge questions can use live search or Wikipedia-based BM25 retrieval. Speech mode uses Whisper-small to transcribe the question and answer options in memory.

## Limitations

- Results come from multiple runs and are not all directly comparable.
- Some sections depend on a live game server, external APIs, gated model access, and Kaggle Secrets.
- The complete notebook is not a lightweight local demo; several cells download multi-gigabyte models and run long benchmarks.
- The reported accuracy table is the authoritative documented aggregate for the main llama.cpp comparison. Exact scores for individual live-search experiments should be read from their generated logs.

## Authors

- Gabriele Bandino
- Matteo Parimbelli
- Francescomaria Reggio
