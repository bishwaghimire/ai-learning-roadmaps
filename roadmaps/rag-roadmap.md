# Retrieval-Augmented Generation (RAG) Roadmap

This roadmap guides learners from **information retrieval fundamentals** to **advanced RAG architectures**, **evaluation**, and **production-grade knowledge-grounded LLM systems**.  
By the end, you should be able to **design, build, evaluate, and deploy robust RAG systems** for real-world applications.

> A **complete Core RAG roadmap** — from embeddings to scalable knowledge-aware AI systems.

---

## Table of Contents

- [Prerequisites](#prerequisites)

- [Beginner Level — RAG Foundations](#beginner-level--rag-foundations)
  - [Core Concepts](#core-concepts)

- [Intermediate Level — Core RAG Engineering](#intermediate-level--core-rag-engineering)
  - [Core Concepts](#core-concepts-1)
  - [Highly Recommended Resources ⭐](#highly-recommended-resources-)

- [Advanced Level — Modern & Production RAG Systems](#advanced-level--modern--production-rag-systems)
  - [Core Concepts (Advanced)](#core-concepts-2)

- [RAG Architectures & Techniques — Learning Resources](#rag-architectures--techniques--learning-resources)

- [Projects (Highly Recommended)](#projects-highly-recommended)
  - [Beginner Projects](#beginner-projects)
  - [Intermediate Projects](#intermediate-projects)
  - [Advanced Projects](#advanced-projects)

- [Final Outcome](#final-outcome)

- [Career Paths After This Roadmap](#career-paths-after-this-roadmap)

- [Final Notes](#final-notes)

---

## Prerequisites

Before starting RAG, you should be comfortable with:

- **Mathematics**
  - Linear Algebra (embeddings, similarity)
  - Probability basics  
  👉 [See README.md#the-math-behind-it-all](https://github.com/bishwaghimire/ai_ml_dl_books/blob/main/README.md#the-math-behind-it-all)

- **Programming**
  - Python
  - APIs & async basics

- **ML / DL / NLP**
  - Transformers
  - Tokenization
  - Embeddings

- **LLM Fundamentals**
  - Prompting
  - Context windows
  - Inference basics

---

## Beginner Level — RAG Foundations

Understand **why RAG exists and how retrieval enhances LLMs**.

### Core Concepts
- [What is Retrieval-Augmented Generation](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Dense vs Sparse Retrieval](https://www.elastic.co/what-is/sparse-retrieval)
- [Embeddings & Vector Similarity](https://huggingface.co/docs/transformers/embeddings)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking/)
- [Prompt Injection Basics](https://arxiv.org/abs/2302.12173)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | *Natural Language Processing with Transformers* | [RAG Explained – Pinecone](https://www.youtube.com/@Pinecone-io) | [Stanford CS224n](https://web.stanford.edu/class/cs224n/) | Beginner |
| 2 | *Designing Machine Learning Systems* – Chip Huyen | [AssemblyAI – RAG](https://www.youtube.com/@AssemblyAI) | [MIT 6.S191](https://introtodeeplearning.com/) | Beginner |

---

## Intermediate Level — Core RAG Engineering

Build **reliable, modular, and evaluatable RAG pipelines**.

### Core Concepts
- [Vector Databases](https://www.pinecone.io/learn/vector-database/)
- [Hybrid Search (BM25 + Dense)](https://www.elastic.co/what-is/hybrid-search)
- [Re-ranking Models](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)
- [Context Construction](https://www.pinecone.io/learn/context-window/)
- [RAG Evaluation (Faithfulness, Relevance)](https://arxiv.org/abs/2307.03045)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | *Designing ML Systems* – Chip Huyen | [LangChain RAG](https://www.youtube.com/@LangChain) | [Full Stack Deep Learning](https://fullstackdeeplearning.com/) | Intermediate |
| 2 | *Natural Language Processing with Transformers* | [LlamaIndex Tutorials](https://www.youtube.com/@llama_index) | [Stanford CS25](https://web.stanford.edu/class/cs25/) | Intermediate |
| 3 | *Deep Learning* – Goodfellow | [Hugging Face RAG](https://www.youtube.com/@huggingface) | [Berkeley CS294](https://people.eecs.berkeley.edu/~jordan/courses/294-fall09/) | Intermediate |

---

### Highly Recommended Resources ⭐

- **Pinecone RAG Learning Center**  
  https://www.pinecone.io/learn/

- **LangChain Documentation (RAG)**  
  https://docs.langchain.com/

- **LlamaIndex RAG Guide**  
  https://docs.llamaindex.ai/

- **RAG Evaluation Paper**  
  https://arxiv.org/abs/2307.03045

> ❗ *RAG quality is 80% retrieval + context design, not the LLM.*

---

## Advanced Level — Modern & Production RAG Systems

Design **scalable, secure, and high-precision RAG systems**.

### Core Concepts
- [Multi-Stage RAG Pipelines](https://arxiv.org/abs/2310.05885)
- [Query Expansion & Decomposition](https://arxiv.org/abs/2305.10601)
- [Agentic RAG](https://arxiv.org/abs/2308.08155)
- [Caching & Latency Optimization](https://huggingface.co/docs/transformers/perf_infer_gpu_one)
- [RAG Security & Guardrails](https://arxiv.org/abs/2307.02483)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | *Designing Machine Learning Systems* | [Full Stack Deep Learning](https://www.youtube.com/@FullStackDeepLearning) | [Stanford CS25](https://web.stanford.edu/class/cs25/) | Advanced |
| 2 | *Generative AI with LLMs* | [Advanced RAG – Pinecone](https://www.youtube.com/@Pinecone-io) | [MIT GenAI Systems](https://ocw.mit.edu/) | Advanced |

---

## RAG Architectures & Techniques — Learning Resources

| S.No | Architecture / Technique | Best Video | Best Blog / Article | Level |
|----:|--------------------------|------------|---------------------|-------|
| 1 | Dense Retrieval | [Embeddings Explained](https://www.youtube.com/watch?v=5p248yoa3oE) | [SBERT Docs](https://www.sbert.net/) | Beginner |
| 2 | Hybrid Search | [Hybrid Search Explained](https://www.youtube.com/watch?v=6eYzU-2JY1Y) | [Elastic Guide](https://www.elastic.co/) | Intermediate |
| 3 | Re-Ranking | [Cross Encoder](https://www.youtube.com/watch?v=ZAMwW5QZJbQ) | [SBERT Rerank](https://www.sbert.net/) | Intermediate |
| 4 | Agentic RAG | [Agent RAG](https://www.youtube.com/watch?v=YxOT0pM8KfA) | [Agentic RAG Paper](https://arxiv.org/abs/2308.08155) | Advanced |
| 5 | Graph RAG | [Graph RAG](https://www.youtube.com/watch?v=HqH2f2E6Jq8) | [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | Advanced |

---

## Projects (Highly Recommended)

> RAG mastery requires **real retrieval problems**, not toy demos.

---

### Beginner Projects

- **PDF Question Answering System**
  ![Level](https://img.shields.io/badge/Level-Beginner-brightgreen)  
  *Skills:* Chunking, Embeddings

- **FAQ Chatbot**
  *Skills:* Vector Search, Prompting

---

### Intermediate Projects

- **Hybrid Search RAG System**
  *Skills:* BM25 + Dense Retrieval

- **RAG with Re-Ranker**
  *Skills:* Cross-Encoders

- **Domain-Specific Knowledge Bot**
  *Skills:* Evaluation, Prompt Design

---

### Advanced Projects

- **Production-Scale RAG System**
  *Skills:* Latency, Caching, Scaling

- **Agentic RAG Pipeline**
  *Skills:* Tool Use, Query Planning

- **Enterprise Secure RAG**
  *Skills:* Access Control, Guardrails

---

## Final Outcome

By completing this roadmap, you will be able to:

- Design high-quality retrieval pipelines  
- Build reliable RAG systems  
- Evaluate hallucination and faithfulness  
- Deploy scalable knowledge-grounded LLM apps  

---

## Career Paths After This Roadmap

- RAG Engineer  
- LLM Application Engineer  
- AI Systems Engineer  
- Generative AI Engineer  

---

## Final Notes

> **Weak retrieval = hallucinations.**  
> **Strong retrieval + clean context = trustworthy AI.**
