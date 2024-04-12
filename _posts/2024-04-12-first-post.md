---
layout: post
title: Machine Learning - Part 1 
date: 2024-04-12 12:33 -0500
author: rwatsh
tags: [Books]
---
Following is my notes from the book Python Machine Learning with PyTorch and Scikit-Learn by Sebastuan Raschka, et. al.

### Setup Python Development Environment

```bash
conda create -n "pyml-book" python=3.9 numpy=1.21.2 scipy=1.7.0 scikit-learn=1.0 matplotlib=3.4.3 pandas=1.3.2

conda activate pyml-book
```

Go for either miniforge or miniconda distributions and set it up as described above.

### Types of Machine learning

![image-20240412105606096](/assets/images/image-20240412105606096.png)

### Supervised Learning

![image-20240412113818576](/assets/images/image-20240412113818576.png)

A supervised learning task with discrete class labels, such as in the previous email spam filtering example, is also called a **classification task**. Another subcategory of supervised learning is **regression**, where the outcome signal is a continuous value.
