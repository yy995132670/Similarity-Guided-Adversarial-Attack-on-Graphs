# Relation-Guided Adversarial Attack on Graphs

This repository contains the **official implementation** of the paper:

> **"Relation-Guided Adversarial Attack on Graphs"**
> *Dongdong An, Yi Yang, Xin Gao, Qin Zhao, Hongda Qi*
> **Intelligent Society and Scientific Computing Laboratory**
> **Shanghai Normal University**, Shanghai, China
> Presented at **The 4th National Conference on Network Computing (ICNC 2025)**.
> *(A recommended journal version will be published soon.)*

##  Overview

Graph Neural Networks (GNNs) are widely used in various applications but remain **highly vulnerable to adversarial attacks**.
In this work, we propose a **Relation-Guided Adversarial Attack (RGAA)** method that **leverages graph structural relationships** to craft more effective and stealthy adversarial perturbations.

Key highlights of our approach:

* Utilizes **relation-aware strategies** to guide attack selection.
* Focuses on **maximizing model degradation** with minimal perturbations.
* Outperforms baseline adversarial attack methods on multiple benchmark datasets.

##  Implementation

Our method is built upon the [**DeepRobust** library](https://github.com/DSE-MSU/DeepRobust/tree/master).
To use our method:

1. Clone the [DeepRobust repository](https://github.com/DSE-MSU/DeepRobust).
2. Navigate to the `targeted_attack` folder inside DeepRobust.
3. Replace or modify the corresponding files with those provided in **this repository**.
4. Run the scripts to perform **relation-guided adversarial attacks**.

##  Experimental Results

Extensive experiments demonstrate that RGAA:

* Achieves **higher attack success rates** than existing baselines.
* Maintains **lower perturbation rates** while effectively degrading model performance.
* Validates its effectiveness across diverse GNN architectures and datasets.
