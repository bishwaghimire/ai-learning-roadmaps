# Computer Vision Roadmap

This roadmap guides learners from **computer vision fundamentals** to **advanced vision systems**, **modern architectures**, and **production-ready CV pipelines**.  
By the end, you should be able to **build, train, evaluate, and deploy computer vision models** used in real-world applications.

> A **complete Core Computer Vision roadmap** — from pixels to perception.

---

## Table of Contents

- [Prerequisites](#prerequisites)

- [Beginner Level — Vision Foundations](#beginner-level--vision-foundations)
  - [Core Concepts](#core-concepts)

- [Intermediate Level — Core Computer Vision](#intermediate-level--core-computer-vision)
  - [Core Concepts](#core-concepts-1)
  - [Highly Recommended Resources ⭐](#highly-recommended-resources-)

- [Advanced Level — Modern & Production CV](#advanced-level--modern--production-cv)
  - [Core Concepts (Advanced)](#core-concepts-2)

- [Computer Vision Tasks & Architectures — Learning Resources](#computer-vision-tasks--architectures--learning-resources)

- [Projects (Highly Recommended)](#projects-highly-recommended)
  - [Beginner Projects](#beginner-projects)
  - [Intermediate Projects](#intermediate-projects)
  - [Advanced Projects](#advanced-projects)

- [Final Outcome](#final-outcome)

- [Career Paths After This Roadmap](#career-paths-after-this-roadmap)

- [Final Notes](#final-notes)

---

## Prerequisites

Before starting Computer Vision, you should be comfortable with:

- **Mathematics**
  - Linear Algebra (matrices, convolutions)
  - Probability & Statistics
  - Calculus (gradients)
  👉 [See README.md#the-math-behind-it-all](https://github.com/bishwaghimire/ai_ml_dl_books/blob/main/README.md#the-math-behind-it-all)

- **Programming**
  - Python
  - NumPy, Matplotlib

- **Machine Learning & Deep Learning Basics**
  - Regression & Classification
  - Neural Networks
  - Backpropagation

- **Tools**
  - OpenCV
  - PyTorch or TensorFlow

---

## Beginner Level — Vision Foundations

Learn how computers **see, represent, and process images** before deep learning.

### Core Concepts
- [Image Representation & Pixels](https://opencv.org/)
- [Color Spaces (RGB, HSV, Grayscale)](https://learnopencv.com/color-spaces-in-opencv-cpp-python/)
- [Image Filtering & Convolution](https://cs231n.github.io/convolutional-networks/)
- [Edge Detection (Sobel, Canny)](https://learnopencv.com/edge-detection-using-opencv/)
- [Image Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | [Computer Vision: Algorithms and Applications – Szeliski](https://szeliski.org/Book/) | [OpenCV Course – freeCodeCamp](https://www.youtube.com/watch?v=oXlwWbU8l2o) | [Introduction to Computer Vision – Georgia Tech](https://www.udacity.com/course/introduction-to-computer-vision--ud810) | Beginner |
| 2 | [Learning OpenCV – Bradski](https://www.oreilly.com/library/view/learning-opencv/9780596516130/) | [Murtaza’s Workshop – OpenCV](https://www.youtube.com/c/MurtazasWorkshopRoboticsandAI) | [Brown CS1430](https://cs.brown.edu/courses/cs1430/) | Beginner |

---

## Intermediate Level — Core Computer Vision

Move from classical CV to **deep learning–based vision systems**.

### Core Concepts
- [Convolutional Neural Networks (CNNs)](https://cs231n.github.io/convolutional-networks/)
- [Image Classification](https://paperswithcode.com/task/image-classification)
- [Object Detection](https://paperswithcode.com/task/object-detection)
- [Image Segmentation](https://paperswithcode.com/task/semantic-segmentation)
- [Data Augmentation](https://albumentations.ai/)
- [Transfer Learning](https://cs231n.github.io/transfer-learning/)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | *Deep Learning* – Goodfellow | [CS231n – Stanford](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) | [Stanford CS231n](https://cs231n.stanford.edu/) | Intermediate |
| 2 | *Dive Into Deep Learning* – Zhang et al. | [Aladdin Persson – CV](https://www.youtube.com/@AladdinPersson) | [MIT 6.S191](https://introtodeeplearning.com/) | Intermediate |
| 3 | *Practical Deep Learning for CV* | [PyTorch CV Tutorials](https://www.youtube.com/@PyTorch) | [Oxford Vision Course](https://www.robots.ox.ac.uk/~vgg/teaching/) | Intermediate |

---

### Highly Recommended Resources ⭐

- **CS231n – Convolutional Neural Networks**
  https://cs231n.stanford.edu/

- **Dive Into Deep Learning (CV Chapters)**
  https://d2l.ai/

- **OpenCV Documentation**
  https://docs.opencv.org/

- **Papers With Code – Vision**
  https://paperswithcode.com/area/computer-vision

> ❗ *CS231n is the single most important Computer Vision course ever created.  
> Master it and you master modern CV.*

---

## Advanced Level — Modern & Production CV

Build **state-of-the-art and scalable computer vision systems**.

### Core Concepts
- [Vision Transformers (ViT)](https://arxiv.org/abs/2010.11929)
- [Self-Supervised Vision Learning](https://lilianweng.github.io/posts/2020-04-07-self-supervised/)
- [Multi-Modal Vision Models](https://openai.com/research/clip)
- [Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [Model Optimization & Deployment](https://pytorch.org/mobile/home/)
- [Edge AI & ONNX](https://onnx.ai/)

---

| S.No | Best Book | Best YouTube Playlist | Best University Course | Level |
|----|---------|----------------------|------------------------|-------|
| 1 | [Designing Machine Learning Systems – Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) | [Hugging Face Vision](https://www.youtube.com/@huggingface) | [Stanford CS25](https://web.stanford.edu/class/cs25/) | Advanced |
| 2 | [Deep Learning for Vision Systems](https://www.manning.com/books/deep-learning-for-vision-systems) | [NVIDIA CV](https://www.youtube.com/@NVIDIA) | [MIT Vision](https://ocw.mit.edu/) | Advanced |

---

## Computer Vision Tasks & Architectures — Learning Resources

| S.No | Task / Architecture | Best Video | Best Blog / Article | Level |
|----:|---------------------|------------|---------------------|-------|
| 1 | Image Classification | [CS231n](https://www.youtube.com/watch?v=OoUX-nOEjG0) | [CS231n Notes](https://cs231n.github.io/classification/) | Beginner |
| 2 | CNN Architectures | [ResNet – StatQuest](https://www.youtube.com/watch?v=GWt6Fu05voI) | [ResNet Paper](https://arxiv.org/abs/1512.03385) | Intermediate |
| 3 | Object Detection (YOLO) | [YOLO Explained](https://www.youtube.com/watch?v=MPU2HistivI) | [YOLO Docs](https://pjreddie.com/darknet/yolo/) | Intermediate |
| 4 | Semantic Segmentation | [U-Net Explained](https://www.youtube.com/watch?v=azM57JuQpQI) | [U-Net Paper](https://arxiv.org/abs/1505.04597) | Intermediate |
| 5 | Vision Transformers | [ViT Explained](https://www.youtube.com/watch?v=TrdevFK_am4) | [Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | Advanced |
| 6 | Multi-Modal Vision | [CLIP Explained](https://www.youtube.com/watch?v=Kc1rY0PqG1A) | [CLIP Paper](https://arxiv.org/abs/2103.00020) | Advanced |

---

## Projects (Highly Recommended)

> Projects are **non-negotiable** for mastering Computer Vision.

---

### Beginner Projects

- **Image Classification (CIFAR-10)**
  ![Level](https://img.shields.io/badge/Level-Beginner-brightgreen)
  https://www.kaggle.com/c/cifar-10  
  *Skills:* CNNs, Training Loops

- **Face Detection (OpenCV)**
  *Skills:* Haar Cascades, Image Processing

---

### Intermediate Projects

- **Object Detection (YOLO / SSD)**
  https://www.kaggle.com/datasets/ultralytics/coco128  
  *Skills:* Bounding Boxes, Detection Metrics

- **Semantic Segmentation**
  *Skills:* U-Net, Pixel-wise Classification

- **Face Recognition System**
  *Skills:* Embeddings, Similarity Metrics

---

### Advanced Projects

- **Vision Transformer from Scratch**
  *Skills:* Attention, Patch Embeddings

- **Real-Time CV System**
  *Skills:* Optimization, Latency, Deployment

- **Multi-Modal Vision System**
  *Skills:* Vision + Language Models (CLIP)

---

## Final Outcome

By completing this roadmap, you will be able to:

- Build classical and deep learning vision systems  
- Work on real-world image and video data  
- Read and implement CV research papers  
- Deploy scalable computer vision pipelines  

---

## Career Paths After This Roadmap

- Computer Vision Engineer  
- Deep Learning Engineer  
- AI Engineer  
- Research Scientist (Vision)

---

## Final Notes

> You don’t need to memorize every model.  
> **Strong fundamentals + modern architectures + real projects = elite CV skills.**
