# Large Language Models (LLM) Roadmap

This roadmap guides learners from **LLM fundamentals** to **state-of-the-art generative models**, **retrieval-augmented systems**, and **production-ready LLM pipelines**.  
By the end, you should be able to **build, fine-tune, evaluate, and deploy LLM-based applications** at scale.

> A **complete Core LLM roadmap** — from tokens to intelligent systems.

---

## Table of Contents

- [Prerequisites](#prerequisites)

- [Beginner Level — LLM Foundations](#beginner-level--llm-foundations)
  - [Core Concepts](#core-concepts)

- [Intermediate Level — Core LLM Engineering](#intermediate-level--core-llm-engineering)
  - [Core Concepts](#core-concepts-1)
  - [Highly Recommended Resources ⭐](#highly-recommended-resources-)

- [Advanced Level — Modern & Production LLM Systems](#advanced-level--modern--production-llm-systems)
  - [Core Concepts (Advanced)](#core-concepts-2)

- [LLM Architectures & Techniques — Learning Resources](#llm-architectures--techniques--learning-resources)

- [Projects (Highly Recommended)](#projects-highly-recommended)
  - [Beginner Projects](#beginner-projects)
  - [Intermediate Projects](#intermediate-projects)
  - [Advanced Projects](#advanced-projects)

- [Final Outcome](#final-outcome)

- [Career Paths After This Roadmap](#career-paths-after-this-roadmap)

- [Final Notes](#final-notes)

---

## Prerequisites

Before starting LLMs, you should be comfortable with:

- **Mathematics**
  - Linear Algebra (embeddings, attention)
  - Probability & Statistics
  👉 [See README.md#the-math-behind-it-all](https://github.com/bishwaghimire/ai_ml_dl_books/blob/main/README.md#the-math-behind-it-all)

- **Programming**
  - Python
  - PyTorch basics

- **Deep Learning & NLP Fundamentals**
  - Neural Networks
  - Sequence Models
  - Tokenization

- **Tools**
  - Hugging Face Transformers
  - CUDA basics (optional but useful)

---

## Beginner Level — LLM Foundations

Understand how **large language models work internally**.

### Core Concepts
- [Language Modeling](https://web.stanford.edu/class/cs224n/)
- [Tokenization (BPE, WordPiece)](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Embeddings](https://jalammar.github.io/illustrated-word2vec/)
- [Attention Mechanism](https://jalammar.github.io/illustrated-transformer/)
- [Autoregressive vs Masked Models](https://huggingface.co/docs/transformers/model_summary)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | [Speech and Language Processing – Jurafsky](https://web.stanford.edu/~jurafsky/slp3/) | [3Blue1Brown – Attention](https://www.youtube.com/watch?v=aircAruvnKk) | [Stanford CS224n](https://web.stanford.edu/class/cs224n/) | Beginner |
| 2 | [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103248/) | [Hugging Face Course](https://huggingface.co/course) | [Oxford NLP](https://www.cs.ox.ac.uk/people/stephen.clark/) | Beginner |

---

## Intermediate Level — Core LLM Engineering

Learn to **fine-tune, adapt, and evaluate LLMs**.

### Core Concepts
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [Pretraining vs Fine-Tuning](https://huggingface.co/docs/transformers/training)
- [Parameter-Efficient Fine-Tuning (LoRA, PEFT)](https://huggingface.co/docs/peft/index)
- [Instruction Tuning](https://arxiv.org/abs/2109.01652)
- [Evaluation Metrics (Perplexity, BLEU, ROUGE)](https://huggingface.co/docs/evaluate/index)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | *Natural Language Processing with Transformers* | [CS224n – Stanford](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ) | [Stanford CS224n](https://web.stanford.edu/class/cs224n/) | Intermediate |
| 2 | *Designing Machine Learning Systems* – Chip Huyen | [Hugging Face Transformers](https://www.youtube.com/@huggingface) | [Full Stack Deep Learning](https://fullstackdeeplearning.com/) | Intermediate |
| 3 | *Deep Learning* – Goodfellow | [AssemblyAI – LLMs](https://www.youtube.com/@AssemblyAI) | [MIT 6.S191](https://introtodeeplearning.com/) | Intermediate |

---

### Highly Recommended Resources ⭐

- **Hugging Face LLM Course**
  https://huggingface.co/course

- **Stanford CS224n**
  https://web.stanford.edu/class/cs224n/

- **Attention Is All You Need (Paper)**
  https://arxiv.org/abs/1706.03762

- **PEFT / LoRA Documentation**
  https://huggingface.co/docs/peft/index

> ❗ *Hugging Face + CS224n together form the strongest practical LLM foundation available today.*

---

## Advanced Level — Modern & Production LLM Systems

Build **scalable, safe, and efficient LLM-powered systems**.

### Core Concepts
- [Large-Scale Pretraining](https://arxiv.org/abs/2005.14165)
- [RLHF](https://arxiv.org/abs/2203.02155)
- [Retrieval-Augmented Generation (RAG)](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Inference Optimization (Quantization, KV Cache)](https://huggingface.co/docs/transformers/perf_infer_gpu_one)
- [LLM System Design](https://www.deeplearning.ai/courses/machine-learning-engineering-for-production-mlops/)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | [Designing Machine Learning Systems – Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) | [Full Stack Deep Learning](https://www.youtube.com/@FullStackDeepLearning) | [Stanford CS25](https://web.stanford.edu/class/cs25/) | Advanced |
| 2 | [Generative AI with LLMs](https://www.coursera.org/learn/generative-ai-with-llms) | [Hugging Face Advanced](https://www.youtube.com/@huggingface) | [MIT LLM Systems](https://ocw.mit.edu/) | Advanced |

---

## LLM Architectures & Techniques — Learning Resources

| S.No | Architecture / Technique | Best Video | Best Blog / Article | Level |
|----:|--------------------------|------------|---------------------|-------|
| 1 | Transformer | [Illustrated Transformer](https://www.youtube.com/watch?v=4Bdc55j80l8) | [Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | Beginner |
| 2 | GPT-style Models | [GPT Explained](https://www.youtube.com/watch?v=4Bdc55j80l8) | [OpenAI GPT Paper](https://arxiv.org/abs/2005.14165) | Intermediate |
| 3 | BERT-style Models | [BERT Explained](https://www.youtube.com/watch?v=4Bdc55j80l8) | [BERT Paper](https://arxiv.org/abs/1810.04805) | Intermediate |
| 4 | Instruction Tuning | [Instruction Tuning](https://www.youtube.com/watch?v=1OQ5W4WJr7A) | [FLAN Paper](https://arxiv.org/abs/2109.01652) | Intermediate |
| 5 | RLHF | [RLHF Explained](https://www.youtube.com/watch?v=2MBJOuVqGdY) | [InstructGPT](https://arxiv.org/abs/2203.02155) | Advanced |
| 6 | RAG Systems | [RAG Explained](https://www.youtube.com/watch?v=T-D1OfcDW1M) | [Pinecone Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) | Advanced |

---

## Projects (Highly Recommended)

> Projects are **mandatory** to truly master LLMs.

---

### Beginner Projects

- **Prompt-Based Text Generator**
  ![Level](https://img.shields.io/badge/Level-Beginner-brightgreen)  
  *Skills:* Prompting, Tokenization

- **Text Classification with BERT**
  *Skills:* Fine-Tuning, Transformers

---

### Intermediate Projects

- **LLM Fine-Tuning (LoRA)**
  *Skills:* PEFT, Efficient Training

- **Document Q&A System**
  *Skills:* Embeddings, Vector Search

- **Chatbot with Memory**
  *Skills:* Context Management

---

### Advanced Projects

- **RAG-Based Production Chatbot**
  *Skills:* Retrieval, LLM Orchestration

- **Inference-Optimized LLM Deployment**
  *Skills:* Quantization, Serving

- **End-to-End LLM System Design**
  *Skills:* Scaling, Monitoring, Safety

---

## Final Outcome

By completing this roadmap, you will be able to:

- Understand how LLMs work internally  
- Fine-tune and adapt LLMs efficiently  
- Build production-grade GenAI systems  
- Evaluate and optimize LLM performance  

---

## Career Paths After This Roadmap

- LLM Engineer  
- Generative AI Engineer  
- Applied Research Engineer  
- AI Systems Engineer  

---

## Final Notes

> You don’t need to train trillion-parameter models.  
> **Understanding + adaptation + system design = real LLM expertise.**
