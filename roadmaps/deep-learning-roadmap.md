# Deep Learning Roadmap

This roadmap guides learners from **neural network fundamentals** to **advanced deep learning systems**, **state-of-the-art architectures**, and **production-grade deep learning workflows**.  
By the end, you should be able to **design, train, optimize, and deploy deep neural networks** used in real-world AI systems.

> A **complete Core Deep Learning roadmap** — from first neural network to research-level mastery.

---

## Table of Contents

- [Prerequisites](#prerequisites)

- [Beginner Level — Neural Network Foundations](#beginner-level--neural-network-foundations)
  - [Core Concepts](#core-concepts)

- [Intermediate Level — Core Deep Learning](#intermediate-level--core-deep-learning)
  - [Core Concepts](#core-concepts-1)
  - [Highly Recommended Resources ⭐](#highly-recommended-resources-)

- [Advanced Level — Modern & Production Deep Learning](#advanced-level--modern--production-deep-learning)
  - [Core Concepts (Advanced)](#core-concepts-2)

- [Deep Learning Architectures — Learning Resources](#deep-learning-architectures--learning-resources)

- [Projects (Highly Recommended)](#projects-highly-recommended)
  - [Beginner Projects](#beginner-projects)
  - [Intermediate Projects](#intermediate-projects)
  - [Advanced Projects](#advanced-projects)

- [Final Outcome](#final-outcome)

- [Career Paths After This Roadmap](#career-paths-after-this-roadmap)

- [Final Notes](#final-notes)

---

## Prerequisites

Before starting Deep Learning, you should be comfortable with:

- **Mathematics**
  - Linear Algebra (vectors, matrices, eigenvalues)
  - Probability & Statistics
  - Calculus (gradients, partial derivatives)
  👉 [See README.md#the-math-behind-it-all](https://github.com/bishwaghimire/ai_ml_dl_books/blob/main/README.md#the-math-behind-it-all)

- **Programming**
  - Python (functions, OOP, generators)
  - Basic ML workflows

- **Machine Learning Fundamentals**
  - Regression & Classification
  - Overfitting vs Underfitting
  - Bias–Variance Tradeoff

- **Tools**
  - NumPy, Pandas
  - Scikit-learn
  - PyTorch or TensorFlow (basic familiarity)

---

## Beginner Level — Neural Network Foundations

Understand how **neural networks work from first principles** and train your **first deep models**.

### Core Concepts
- [What is a Neural Network?](https://www.deeplearning.ai/resources/)
- [Perceptron & Multi-Layer Perceptron](https://cs231n.github.io/neural-networks-1/)
- [Activation Functions](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)
- [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Backpropagation](https://cs231n.github.io/optimization-2/)
- [Gradient Descent Variants](https://ruder.io/optimizing-gradient-descent/)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | [Deep Learning – Ian Goodfellow](https://www.deeplearningbook.org/) | [3Blue1Brown – Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | [DeepLearning.AI – Neural Networks](https://www.coursera.org/learn/neural-networks-deep-learning) | Beginner |
| 2 | [Neural Networks from Scratch – Harrison Kinsley](https://nnfs.io/) | [Neural Networks from Scratch](https://www.youtube.com/c/Sentdex) | [MIT 6.S191](https://introtodeeplearning.com/) | Beginner |

---

## Intermediate Level — Core Deep Learning

Master **CNNs, RNNs, optimization techniques**, and modern training strategies.

### Core Concepts
- [Convolutional Neural Networks (CNNs)](https://cs231n.github.io/convolutional-networks/)
- [Recurrent Neural Networks (RNNs)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [LSTM & GRU](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Weight Initialization](https://cs231n.github.io/neural-networks-2/)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)
- [Dropout & Regularization](https://jmlr.org/papers/v15/srivastava14a.html)
- [Optimizers (Adam, RMSProp)](https://pytorch.org/docs/stable/optim.html)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | *Deep Learning* – Goodfellow | [CS231n – Stanford](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) | [Stanford CS231n](https://cs231n.stanford.edu/) | Intermediate |
| 2 | *Dive into Deep Learning* – Zhang et al. | [MIT 6.S191](https://www.youtube.com/c/mitdeeplearning) | [MIT 6.S191](https://introtodeeplearning.com/) | Intermediate |
| 3 | *Neural Networks and Deep Learning* – Nielsen | [Aladdin Persson – PyTorch](https://www.youtube.com/@AladdinPersson) | [Oxford Deep Learning](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/) | Intermediate |

---

### Highly Recommended Resources ⭐

- **Deep Learning Specialization – Andrew Ng**  
  https://www.coursera.org/specializations/deep-learning

- **CS231n – Convolutional Neural Networks**  
  https://cs231n.stanford.edu/

- **Dive Into Deep Learning (Free Book)**  
  https://d2l.ai/

- **Neural Networks from Scratch**  
  https://nnfs.io/

> ❗ *Andrew Ng’s Deep Learning Specialization is one of the most influential DL courses ever created.  
> It builds intuition that separates users from true practitioners.*

---

## Advanced Level — Modern & Production Deep Learning

Work with **transformers, self-supervised learning, scalability, and deployment**.

### Core Concepts
- [Transformers & Attention](https://jalammar.github.io/illustrated-transformer/)
- [Self-Supervised Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)
- [Transfer Learning](https://cs231n.github.io/transfer-learning/)
- [Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Model Compression & Quantization](https://pytorch.org/docs/stable/quantization.html)
- [Deep Learning System Design](https://www.deeplearning.ai/courses/machine-learning-engineering-for-production-mlops/)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | [Designing Deep Learning Systems – Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) | [Hugging Face Course](https://www.youtube.com/@huggingface) | [Stanford CS25](https://web.stanford.edu/class/cs25/) | Advanced |
| 2 | [Deep Learning Systems – MIT](https://dlsyscourse.org/) | [MIT Deep Learning Systems](https://www.youtube.com/c/mitdeeplearning) | [MIT 6.824 + DL Systems](https://dlsyscourse.org/) | Advanced |

---

## Deep Learning Architectures — Learning Resources

| S.No | Architecture | Best Video | Best Blog / Article | Level |
|----:|-------------|------------|---------------------|-------|
| 1 | MLP | [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) | [CS231n](https://cs231n.github.io/neural-networks-1/) | Beginner |
| 2 | CNN | [CS231n CNN](https://www.youtube.com/watch?v=bNb2fEVKeEo) | [CS231n Notes](https://cs231n.github.io/convolutional-networks/) | Intermediate |
| 3 | RNN | [Karpathy RNN](https://www.youtube.com/watch?v=yCC09vCHzF8) | [Colah Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Intermediate |
| 4 | LSTM / GRU | [StatQuest LSTM](https://www.youtube.com/watch?v=8HyCNIVRbSU) | [Colah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Intermediate |
| 5 | Autoencoders | [Autoencoders – DeepLearningAI](https://www.youtube.com/watch?v=9zKuYvjFFS8) | [DeepLearning.ai](https://www.deeplearning.ai/) | Intermediate |
| 6 | GANs | [GANs – Ian Goodfellow](https://www.youtube.com/watch?v=AJVyzd0rqdc) | [NIPS Paper](https://arxiv.org/abs/1406.2661) | Advanced |
| 7 | Transformers | [Illustrated Transformer](https://www.youtube.com/watch?v=4Bdc55j80l8) | [Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | Advanced |

---

## Projects (Highly Recommended)

> Projects are **mandatory** to truly understand Deep Learning.

---

### Beginner Projects

- **MNIST Digit Classification**
  ![Level](https://img.shields.io/badge/Level-Beginner-brightgreen)
  https://www.kaggle.com/c/digit-recognizer  
  *Skills:* MLP, Backpropagation, PyTorch Basics

- **Fashion-MNIST Classification**
  *Skills:* CNNs, Regularization

---

### Intermediate Projects

- **Image Classification (CIFAR-10)**
  *Skills:* CNN Architectures, Data Augmentation

- **Sentiment Analysis (RNN/LSTM)**
  https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews  
  *Skills:* NLP, Sequence Models

- **Face Mask Detection**
  *Skills:* Transfer Learning, CNNs

---

### Advanced Projects

- **Transformer from Scratch**
  *Skills:* Attention, Self-Attention, Architecture Design

- **Image Generation with GANs**
  *Skills:* GAN Training, Stability Tricks

- **End-to-End Deep Learning Deployment**
  *Skills:* Model Serving, Optimization, Monitoring

---

## Final Outcome

By completing this roadmap, you will be able to:

- Build deep neural networks from scratch  
- Train CNNs, RNNs, and Transformers  
- Understand modern research papers  
- Deploy deep learning systems at scale  

---

## Career Paths After This Roadmap

- Deep Learning Engineer  
- AI Engineer  
- Research Scientist  
- Applied ML / DL Engineer  

---

## Final Notes

> You don’t need to know *every* architecture.  
> **Strong fundamentals + modern architectures + real projects = mastery.**
