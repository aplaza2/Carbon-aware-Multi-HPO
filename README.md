# Multi-Objective Hyperparameter Optimization for Sustainable Machine Learning

This repository contains the experimental code used for the work **“Multi-Objective Hyperparameter Optimization for Sustainable Machine Learning”**.

**Authors:** Alejandro Plaza, Denis Parra, Rodrigo Toro
**Affiliation:** Pontificia Universidad Católica de Chile

## Overview

The goal of this project is to study the use of multi-objective hyperparameter optimization (HPO) for sustainable machine learning. In particular, the experiments compare standard single-objective HPO, which optimizes only predictive performance, against multi-objective HPO, which jointly considers predictive performance and carbon emissions.

The repository is organized by experimental domain. Each folder contains the code required to run the experiments for one domain.


## Optimization Setup

Across domains, the experiments compare two HPO settings:

### Single-objective HPO

The single-objective setting optimizes only the predictive or task-specific metric:

* Image classification: classification performance.
* Sequential recommendation: recommendation quality.
* Reinforcement learning: reward.

### Multi-objective HPO

The multi-objective setting optimizes two objectives:

1. Maximize the task-specific performance metric.
2. Minimize carbon emissions.

Carbon emissions are measured during each trial and stored together with the trial results.

