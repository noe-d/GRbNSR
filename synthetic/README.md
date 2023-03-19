<a class="anchor" id="top"></a>
# Graph Representation based Generative Network Symbolic Regression (GR.GNSR)

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org) [![badge](https://img.shields.io/badge/launch-binder-579aca.svg?logo=data:image/png;base64,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)](https://mybinder.org) [![OpenInColab](https://colab.research.google.com/assets/colab-badge.svg)](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjf9arI8uT7AhXqR_EDHVXdCvsQFnoECBEQAQ&url=https%3A%2F%2Fcolab.research.google.com%2F&usg=AOvVaw3A5aPK2kLFzKOzb6sOckVw)


This repository is an extension of the [`synthetic` repository <img src="https://github.com/fluidicon.png" height=14>](https://github.com/telmomenezes/synthetic). It contains a python implementation of the Generative Network Symbolic Regression algorithm <a class="anchor" id="ref_2014_0">[[1]](#bib_2014).
Here, the capabilities of Graph Representations Learning algorithms to alleviate the distance metric are enquired.

💡 *An overview of the theoretical grounding is given [here](#foundations).*

🐍 *Help to run the code is provided in the [Quick start](#quick_start) and [Tutorial](#tutorial) sections. For instructions regarding the reproduction of previous studies jump [here](#reproduction).*

<div style="background-color:rgba(0, 0, 126, 0.1); vertical-align: middle; padding:10px 0;">
<a class="anchor" id="ToC"></a>
<b><u>Table of Content</u></b>

<p align="center">
 <a href="#quick_start">⚡️</a> • <a href="#foundations">🧬</a> • <a href="#architecture">🧱</a> • <a href="#usage">🐾</a> • <a href="#reproduction">🦎</a> • <a href="#folder_structure">🌵</a> • <a href="#references">📚</a>
</p>


</div>

<a class="anchor" id="quick_start"></a>
## ⚡️ 1. Quick start
<p align="right"><a href="#top">🔝</a></p>

Set up the environment:
```shell
pip install -r requirements.txt
```

Run the graph representation based generative network for symbolic regression algorithm on an observed network:
```shell
! PYTHONPATH=$(pwd) python3 synthetic/cli.py evo --inet <OG_NETWORK> --odir <DIR> --undir
```

---

<a class="anchor" id="foundations"></a>
## 🧬 2. Foundations
<p align="right"><a href="#top">🔝</a></p>

TODOs:
- [ ] motivate extension
- [ ] link to papers and what it can be used for

<a class="anchor" id="evo"></a>
### 2a. Evolutionary process
<p align="right"><a href="#foundations">🧬 </a></p>

TODOs:
- [ ] brief description
- [ ] insert scheme (eg. from 2019)

<a class="anchor" id="distance"></a>
### 2b. Distance metric
<p align="right"><a href="#foundations">🧬</a></p>

The distance between networks is based on their representation in an embedding space resulting from the Graph Representation methods. The distance is computed using a distance metric (eg. euclidian, cosine, Manhattan, etc.) between the corresponding networks' coordinate within the chosen representation space.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./illustrations/dist_illust_light.png">
  <source media="(prefers-color-scheme: dark)" srcset="./illustrations/dist_illust_dark.png">
  <img alt="Illustration - Distance in Representation Space.">
</picture>

TODOs:
- [ ] point towards GRSSL module
- [x] tikz schemes


<a class="anchor" id="architecture"></a>
## 🧱 3. Architecture
<p align="right"><a href="#top">🔝</a></p>

TODOs:
- [ ] cards ?
- [ ] `cli` calls

<details><summary>Other bricks</summary><br/>
A few words about other features (eg. generation or pruning)
</details>

<a class="anchor" id="usage"></a>
## 🐾 4. Tutorial
<p align="right"><a href="#top">🔝</a></p>

TODOs:
- [ ] input formats
- [ ]  link to OG documentation
- [ ] main command lines

<details><summary>🖼 Illustrated example</summary><br/>
Make visualisation of the evolutionary process and comparisons between og graph and output


TODOs:
- [ ] **All of that in a jupyter-notebook**
- [ ] describe and plot OG net
- [ ] print command lines
- [ ] excerpt of expected logs
- [ ] plot trajectory in representation space
- [ ] plot each graph mutation
- [ ] document evolutionary process (time, loss, ...)
- [ ] comparison between OG plot and output

</details>

<a class="anchor" id="reproduction"></a>
## 🦎 5. Reproducibility
<p align="right"><a href="#top">🔝</a></p>

TODOs:
- [ ] Note on data (what is it ? where to get it ? etc.)
- [ ] motivation: why important to do that ? how can it be different ?

<a class="anchor" id="reproduction_2014"></a>
### 5a. Evolutionary fits <a class="anchor" id="ref_2014_5a"></a>[[1]](#bib_2014)
<p align="right"><a href="#reproduction">🦎</a></p>


```shell
./repro_scripts/2014_experiments.sh <MODEL_PATH> <OUTPUT_FOLDER> <N_REPEAT>
```
For each of the networks:
- do evolutionary fit `n` times
- find best run (ie. lowest final loss)
- store `bestprog`, `bestnet`
- visualise side by side: OLD | OG | NEW (1:1:1 or 2//1:1)
- Compare best nets from OLD and NEW with both OLD and NEW metrics (see if each one is better based on their own metric or not)
- replicate figure 4 with: (=> compute generator dissimilarities)
  - old best as basis generator
  - new best as basis generator

<details><summary>⚙️ Process</summary><br/>

</details>

<details><summary>🔮 Output</summary><br/>

</details>

<a class="anchor" id="reproduction_2019"></a>
### 5b. Families of generators <a class="anchor" id="ref_2019_5b"></a>[[2]](#bib_2019)
<p align="right"><a href="#reproduction">🦎</a></p>

```shell
./repro_scripts/2019_cluster.sh <MODEL_PATH> <OUTPUT_FOLDER>
```
For each of the networks:
- classification of OLD `bestprog`s based on NEW distance on `bestnet`s
- visualisation of the embeddings
  - with ground truth
  - with new clusters
  - (also interactive to view progs)
- clustering based on NEW distance and coherence w/ ground truth (kind of multiclass F1 score)

<details><summary>⚙️ Process</summary><br/>
The set of generators studied in the <i>Automatic Discovery of Families of Network Generative Processes</i> article <a class="anchor" id="ref_2019_5b_process"></a><a href="#bib_2019">[2]</a> are embedded with the trained Graph Representation algorithm.
Its propensity to cluster generators coherently (by embedding the resulting networks) is analysed and compared with the family labels drawn from the earlier paper.

To do so, we re-use the found generators (see eg. Tab.2 <a href="#bib_2019">[2]</a>).
The vector representation of the networks associated with the mentioned generators are computed with the trained Graph Representation method (used to compute the distance in the presented GR.GNSR version). These vector representations of the networks are used as a proxy of the representation of their generators.
Multidimensional Scaling (MDS) is applied to these representations to obtain 2-dimensional representations of the generators that are easily explorable.
They are then clustered (using K-Means) and a clustering score (accuracy) with respect to pre-determined families is computed. The clusters are shown on the 2D visualisation.
</details>

<details><summary>🔮 Output</summary><br/>

</details>

---
<a class="anchor" id="folder_structure"></a>
## 🌵 6. Folder structure
<p align="right"><a href="#top">🔝</a></p>

TODOs:
- [ ] explain the different folders and modules
- [ ] generate tree of the folder


<a class="anchor" id="references"></a>
## 📚 7. References <a class="anchor" id="references"></a>
<p align="right"><a href="#top">🔝</a></p>

<a class="anchor" id="bibliography"></a>
### Bibliography
<p align="right"><a href="#references">📚</a></p>

<a class="anchor" id="bib_2014"></a> [1] (^back to: [<sup>Intro</sup>](#ref_2014_0); [<sup>5a</sup>](#ref_2014_5a)) [ [paper](https://www.nature.com/articles/srep06284) | [code](https://github.com/telmomenezes/synthetic) ] <br> Telmo Menezes, & Camille Roth (2014). Symbolic regression of generative network models. *Sci Rep **4***, 6284.

<a class="anchor" id="bib_2019"></a> [2] (^back to: [<sup>Intro</sup>](#ref_2019_0); [<sup>5b</sup>](#ref_2019_5b); [<sup>5b-process</sup>](#ref_2019_5b_process)) [ [paper](http://arxiv.org/abs/1906.12332) ]
Telmo Menezes, & Camille Roth (2019). Automatic Discovery of Families of Network Generative Processes. *CoRR, abs/1906.12332*.

<a class="anchor" id="acknoledgements"></a>
### Acknoledgements
<p align="right"><a href="#references">📚</a></p>

- XX
- XX

<p align="right"><a href="#top">🔝</a></p>
<p align="center">
 <a href="#quick_start">⚡️</a> • <a href="#architecture">🧬</a> • <a href="#usage">🐾</a> • <a href="#reproduction">🦎</a> • <a href="#folder_structure">🌵</a> • <a href="#references">📚</a>
</p>
