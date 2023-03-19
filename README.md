<a class="anchor" id="top"></a>
# Graph Representation based Network Generator Analysis (GR.NGA)
<p align="center">
 • <a href="#repository_goals">🏔</a>
 • <a href="#quick_start">🚞</a>
 • <a href="#roadmap">🛤</a>
 • <a href="#synthetic">🌳</a> 
 • <a href="#dl_module">🌊</a>
 • <a href="#reproducibility">🎰</a>
 • <a href="#space_segregation">🔭</a>
 • <a href="#additional_experiments">🧫</a>
 • <a href="#structure">🌵</a>
 • <a href="#references">📚</a>
 •
</p>

![files](https://tokei.rs/b1/github/noe-d/AlignmentTool?category=files) ![codelines](https://tokei.rs/b1/github/noe-d/AlignmentTool?category=code) ![comments](https://tokei.rs/b1/github/noe-d/AlignmentTool?category=comments)

<a class="anchor" id="repository_goals"></a>
## 🏔 Repository goals
<p align="right"><a href="#top">🔝</a></p>

TODO:
- [ ] motivations / research questions / aims
- [ ] different resources gathered etc.

<a class="anchor" id="quick_start"></a>
## 🚞 Quick Start
<p align="right"><a href="#top">🔝</a></p>

### Set up

TODO:
- [ ] coarse description of the content of the folder
- [ ] instructions on how to make things run / set up

### Run the code

TODOs:
- [ ] train DL model
  - [ ] assess performances
  - [ ] visualise embeddings
- [ ] analyse segregative capabilities on synthetic generated graphs
- [ ] reproduce symbolic regression

<a class="anchor" id="roadmap"></a>
## 🛤 Roadmap
<p align="right"> (&rarr; <a href="./ROADMAP.md"> 🛤 ROADMAP.md</a> ) <a href="#top">🔝</a></p>


---

<a class="anchor" id="synthetic"></a>
## 🌳 Synthetic
<p align="right">( <a href="./synthetic">Folder</a> • <a href="./synthetic/README.md#usage">How to ?</a> ) <a href="#top">🔝</a></p>

Sumary. Phasellus diam magna, consequat ac dictum nec, aliquam eu nunc. Aliquam et congue quam. In sagittis lectus tellus, a euismod magna malesuada a. Aliquam malesuada fermentum risus sed egestas. Nunc nisl odio, tristique eget mattis non, sodales varius erat.

> **Warning**
> This is a note

Mauris sed congue elit. Sed commodo leo augue, a commodo eros lobortis quis. Ut luctus fringilla ligula, sit amet consectetur purus sollicitudin non. Duis ornare ipsum ipsum, sed rhoncus odio varius non. Nulla cursus pulvinar lacus, ac egestas tortor euismod vitae. Nulla sollicitudin, nulla quis facilisis viverra, ipsum turpis cursus velit, id bibendum lectus lectus nec turpis.

<a class="anchor" id="dl_module"></a>
## 🌊 DL Module
<p align="right">( <a href="./DL_module">Folder</a> • <a href="./DL_module/README.md#-5-usage-">How to ?</a> ) <a href="#top">🔝</a></p>

Sumary. Phasellus diam magna, consequat ac dictum nec, aliquam eu nunc. Aliquam et congue quam. In sagittis lectus tellus, a euismod magna malesuada a. Aliquam malesuada fermentum risus sed egestas. Nunc nisl odio, tristique eget mattis non, sodales varius erat. Maecenas consequat semper sapien, non mattis dui blandit vel. Quisque eget mattis urna, ac sodales massa. Mauris sed congue elit. Sed commodo leo augue, a commodo eros lobortis quis. Ut luctus fringilla ligula, sit amet consectetur purus sollicitudin non. Duis ornare ipsum ipsum, sed rhoncus odio varius non. Nulla cursus pulvinar lacus, ac egestas tortor euismod vitae. Nulla sollicitudin, nulla quis facilisis viverra, ipsum turpis cursus velit, id bibendum lectus lectus nec turpis. Aliquam eu dui ut odio ornare sodales. Maecenas bibendum porttitor libero, non ullamcorper mauris finibus ac.

> **Note**
> This is a note

---
TODOS for each of the sub-sections:
- [ ] research question / motivations
  - [ ] repro
  - [ ] space_seg
  - [ ] exps...
- [ ] links to the folders
  - [x] repro
  - [x] space_seg
  - [ ] exps...
- [ ] few lines on whats going on in the code
  - [ ] repro
  - [ ] space_seg
  - [ ] exps...
- [ ] display main outputs
  - [ ] repro
  - [ ] space_seg
  - [ ] exps...


<a class="anchor" id="reproducibility"></a>
## 🎰 Reproducibility
<p align="right"><a href="#top">🔝</a></p>

### Evolutionary search [2014]
<p align="right">( <a href="./reproducibility">Folder</a> • <a href="./reproducibility/README.md">How to ?</a> • <a href="./repro_evo.ipynb">Demo</a> ) <a href="#reproducibility">🎰</a></p>

<details><summary>⚙️ Process</summary><br/>

</details>

<details><summary>🔮 Output</summary><br/>

Objectives: replication of the following figures:

**Table 2:** evolutionary search of the best generator and visual comparison of the outputs
<p align="center">
  <picture>
    <source srcset="./illustrations/obj/2014_tab2.png">
    <img width=80% alt="Illustration - Space Discrimination from degrees.">
  </picture>
</p>

**Figure 4:** generator dissimilarity (computed through generative process) vs. fitness (av. score over 30 runs (?))
<p align="center">
  <picture>
    <source srcset="./illustrations/obj/2014_fig4.png">
    <img width=80% alt="Illustration - Space Discrimination from degrees.">
  </picture>
</p>

</details>

### Generators' families [2019]
<p align="right">( <a href="./reproducibility">Folder</a> • <a href="./reproducibility/README.md">How to ?</a> • <a href="./repro_fams.ipynb">Demo</a> ) <a href="#reproducibility">🎰</a></p>

<details><summary>⚙️ Process</summary><br/>

</details>

<details><summary>🔮 Output</summary><br/>

Objectives: replication of the following figures:

**Figure 3:** MDS projection of generator's instances pairwise-distances
<p align="center">
  <picture>
    <source srcset="./illustrations/obj/2019_fig3.png">
    <img width=80% alt="Illustration - Space Discrimination from degrees.">
  </picture>
</p>

</details>

<a class="anchor" id="space_segregation"></a>
## 🔭 Space Segregation
<p align="right"> ( <a href="./space_seg">Folder</a> • <a href="./space_seg/README.md#-5-usage-">How to ?</a> • <a href="./space_seg.ipynb">Demo</a> ) <a href="#top">🔝</a></p>

<details><summary>⚙️ Process</summary><br/>

</details>

<details><summary>🔮 Output</summary><br/>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="./illustrations/deg_space_seg.png">
    <source media="(prefers-color-scheme: dark)" srcset="./illustrations/deg_space_seg_dm.png">
    <img width=80% alt="Illustration - Space Discrimination from degrees.">
  </picture>
</p>

</details>

<a class="anchor" id="additional_experiments"></a>
## 🧫 Additional experiments
<p align="right"> ( <a href="./add_exps">Folder</a> • <a href="./add_exps/README.md#-5-usage-">How to ?</a> • <a href="./add_exps.ipynb">Demo</a> ) <a href="#top">🔝</a></p>

---
<a class="anchor" id="structure"></a>
## 🌵 Repository structure
<p align="right"><a href="#top">🔝</a></p>

<a class="anchor" id="references"></a>
## 📚 References
<p align="right"><a href="#top">🔝</a></p>

<a class="anchor" id="bibliography"></a>
### Bibliography

<a class="anchor" id="acknowledgements"></a>
### Acknowledgements

- XX
- XX

---
<p align="center">
 • <a href="#repository_goals">🏔</a>
 • <a href="#quick_start">🚞</a>
 • <a href="#synthetic">🌳</a>
 • <a href="#dl_module">🌊</a>
 • <a href="#reproducibility">🎰</a>
 • <a href="#space_segregation">🔭</a>
 • <a href="#additional_experiments">🧫</a>
 • <a href="#structure">🌵</a>
 • <a href="#references">📚</a>
 •
</p>
