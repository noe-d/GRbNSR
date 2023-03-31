<a class="anchor" id="top"></a>
# Graph Representation based Network Generator Analysis (GR.NGA)
<p align="center">
 • <a href="#repository_goals">🏔</a>
 • <a href="#quick_start">🚞</a>
 • <a href="#synthetic">🌳</a> 
 • <a href="#dl_module">🌊</a>
 • <a href="#nsr">🧭</a>
 • <a href="#space_segregation">🔭</a>
 • <a href="#references">📚</a>
 •
</p>

![files](https://tokei.rs/b1/github/noe-d/GRbNSR?category=files) ![codelines](https://tokei.rs/b1/github/noe-d/GRbNSR?category=code)

*[🚧] This repository needs to be further cleaned.*

<a class="anchor" id="repository_goals"></a>
## 🏔 Repository goals
<p align="right"><a href="#top">🔝</a></p>

The present repository hosts the code to replicate the experiments presented in the Master Thesis *Graph Representation based Network Symbolic Regression* undertook from October 2022 to April 2023 at Centre Marc Bloch's Computational Social Science Team, in the context of EPFL's Digital Humanities master program.

The main goal of this repository is to better understand Graph Representation Learning through the empirical analysis of the spatial representation induced by such models, as well as probing their efficiency in tasks departing from standard classification.

The repository provides two foundation modules, [`DL_module`](./DL_module/) and [`synthetic`](./synthetic/), together with ensuing analysis and application modules: [`reproducibility`](./reproducibility/) and [`space_seg`](./space_seg/).

<a class="anchor" id="quick_start"></a>
## 🚞 Quick Start
<p align="right"><a href="#top">🔝</a></p>

### Set up

It is recommended to create a work environment, and to install the required packages in the environment with the following command line:

```shell
conda create -n env_grbnsr python=3.10.6 jupyterlab
conda activate env_grbnsr
pip install -r requirements.txt
conda install graph-tool
```

### Run the code

To run the code, please refer to the folders sheltering the different moduels described below, their associated folders and/or the demo notebooks:
- [Deep Graph Representation Learning](#dl_module): [ 🗂 Folder ](./DL_module/) | [ 📓 Notebook ](./DL_module/pipeline.ipynb)
- [Network symbolic Regression](#nsr): [ 🗂 Folder ](./reproducibility) | [ 📓 Notebook ](repro_test.ipynb)
- [Spatial Segregation Analysis](#space_segregation): [ 🗂 Folder ](./space_seg/) | [ 📓 Notebook ](space_seg.ipynb)

The natural pipeline is to:
1. first [train](./DL_module/README.md#-5-usage-) deep GRL models (or download them);
2. then [generate a dataset](./space_seg/README.md) of synthetic networks with known generative processes (or download it);
3. a) analyse the [capabilities of the models to cluster networks](./space_seg/README.md#-analysis-synembanalyzer) stemming from different generators in distinct regions of space.
  b) And / Or : perform [network symbolic regression](./reproducibility/README.md#replication), using the pre-trained GRL model to compute the distance between networks directly in the representation space.



---

<a class="anchor" id="synthetic"></a>
## 🌳 Synthetic
<p align="right">( <a href="./synthetic">Folder</a> ) <a href="#top">🔝</a></p>

The [synthetic folder](./synthetic) hosts the code for the network symbolic regression <a class="anchor" id="ref_2014_0">[[1]](#bib_2014).

This algorithm draws from genetic programming in order to retrieve plausible generative processes responsible for a target network through mutations and selection mechanisms.
The notion of generators and the definition of distances between graphs, used to guide the selection process, are two cornerstones of this algorithm. Both are then combined to perform evolutionary search in order to retrieve a satisfying solution in an iterative manner.

*Generators* are construed as iterative stochastic procedures that produce links one by one based on probabilistic preferences to construct graphs. The probabilistic weights are determined by mathematical functions applied to variables of the graphs such as nodes' degree, distance between nodes or nodes' identifiers for instance.

In the original paper, the distance between networks is computed by comparing summary statistics of pre-determined features from the graph, eg. degree centralities distribution or pattern counts.

In practice, given a target graph, the algorithm performs an evolutionary search in the space of generators, and outputs the best generator found to fit the network under the form of a tree-based computer program.

<a class="anchor" id="dl_module"></a>
## 🌊 DL Module
<p align="right">( <a href="./DL_module">Folder</a> • <a href="./DL_module/README.md#-5-usage-">How to ?</a> ) <a href="#top">🔝</a></p>

The [deep learning module](./DL_module) is leveraged to obtain GRL models. Indeed, the aim of this repository is to probe the capabilities of such technologies in different settings.

Inspired by the paradigmatic shift towards pre-trained foundation models in various ML fiels, we focused on self-supervised models. Two state-of-the-art self-supervised GRL models are re-implemented: GraphMAE and PGCL.

The DL module enables to train these models and to evaluate them on standard graph classification benchmarks.

The general idea is to get pre-trained models that can produce vector representation of any input graph and to use these representations in any downstream task.

---


<a class="anchor" id="nsr"></a>
## 🧭 Network Symbolic Regression
<p align="right">( <a href="./reproducibility">Folder</a> • <a href="./reproducibility/README.md">How to ?</a> • <a href="./repro_evo.ipynb">Demo</a> ) <a href="#top">🔝</a></p>

The network symbolic regression algorithm is re-encapsulated in the [reproducibility folder](./reproducibility).

Besides reproducing the results obtained with the original algorithm, this repository is used to leverage trained DL models to define the notion of distance of the algorithm. Under this framework, the distance is not anymore the difference between hand-engineered features distributions, but the geometric distance between the networks' representation:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./illustrations/dist_illust_light.png">
  <source media="(prefers-color-scheme: dark)" srcset="./illustrations/dist_illust_dark.png">
  <img alt="Illustration - Distance in Representation Space.">
</picture>

Practical usage of this code is documented in the [repro notebook](./repro.ipynb).

<a class="anchor" id="space_segregation"></a>
## 🔎 Space Segregation
<p align="right"> ( <a href="./space_seg">Folder</a> • <a href="./space_seg.ipynb">Demo</a> ) <a href="#top">🔝</a></p>

The capabilities of the GRL models to produce semantically segregated representation spaces are probed in [`space_seg`](./space_seg/).

This module allows to generate controlled datasets of synthetic graphs based on the notion of *generators* (as introduced for the network symbolic regression).

Then, the custom dataset can be used to assess the ability of the model to represent networks stemming from semantically distinct generative procsesses in different regions of space, or not.

Insights can be drawn from the 2D visualisation of the spatial distribution of the networks representations.
The evaluation procedure is also systematise through the use of traditional cluster analysis measures.

The whole pipeline is illustrated in the [companion notebook](./space_seg.ipynb).


<details><summary>🔮 Visualisation Examples</summary><br/>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="./illustrations/tsne_gmm_degs.png">
    <source media="(prefers-color-scheme: dark)" srcset="./illustrations/tsne_gmm_degs_dm.png">
    <img width=80% alt="Illustration - Space Discrimination from degrees.">
  </picture>
</p>

</details>

---

<a class="anchor" id="references"></a>
## 📚 References
<p align="right"><a href="#top">🔝</a></p>

<a class="anchor" id="bib_2014"></a> [1] [ [paper](https://www.nature.com/articles/srep06284) | [code](https://github.com/telmomenezes/synthetic) ] <br> Telmo Menezes, & Camille Roth (2014). Symbolic regression of generative network models. *Sci Rep **4***, 6284.

---
<p align="center">
 • <a href="#repository_goals">🏔</a>
 • <a href="#quick_start">🚞</a>
 • <a href="#synthetic">🌳</a> 
 • <a href="#dl_module">🌊</a>
 • <a href="#nsr">🧭</a>
 • <a href="#space_segregation">🔭</a>
 • <a href="#references">📚</a>
 •
</p>
