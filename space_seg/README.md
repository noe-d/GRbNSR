<a class="anchor" id="top"></a>
# 🔭 Space Segregation
<p align="center">
 • <a href="#quick_start">⚡️</a>
 • <a href="#data">💾</a>
 • <a href="#embed">📍</a>
 • <a href="#analysis">🩺</a>
 • <a href="#structure">🌵</a>
 • <a href="#references">📚</a>
 •
</p>

![](../illustrations/space_seg/space_seg_quick_pipe.png)

<a class="anchor" id="quick_start"></a>
## ⚡️ Quick Start
<p align="right"><a href="#top">🔝</a></p>

TODO:
- [ ] download the dataset
- [ ] download the pre-trained model (or use traditional statistics)
- [ ] run the script
- [ ] links: more details about generation, embedding computation, analysis


---
<a class="anchor" id="data"></a>
## 💾 Data Generation <br>(`SynNetsGenerator`)
<p align="right"><a href="#top">🔝</a></p>

`SynNetsGenerator` class is constructed around a given dictionnary containing the required informations to generate the networks. It should have the following structure:
```python
gen_dict = {
  <FAMILY_NAME>:{
    <TYPE_NAME>:{
      "generator_path": <path_to_file>,
      "number_generated": <n_graphs>,
    },
    ...
  },
  ...
}
```

A [`SynNetsEmbedder` object](#embed) can be instantiated with the `make_embedder` method, given an embedding method.

<a class="anchor" id="embed"></a>
## 📍 Embeddings Computation <br>(`SynNetsEmbedder`)
<p align="right"><a href="#top">🔝</a></p>

A [`SynEmbAnalyzer` object](#analysis) can be instantiated with the `make_analyzer` method.

<a class="anchor" id="analysis"></a>
## 🩺 Analysis <br>(`SynEmbAnalyzer`)
<p align="right"><a href="#top">🔝</a></p>

<a class="anchor" id="analysis_visu"></a>
### Visualisation
<p align="right"><a href="#analysis">🩺</a></p>

<details><summary>🔮 Output</summary><br/>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="../illustrations/space_seg/GrMAE_vGCN_v100_space.gif">
    <source media="(prefers-color-scheme: dark)" srcset="../illustrations/space_seg/GrMAE_vGCN_v100_space.gif">
    <img width=90% alt="Animated Space Discrimination with GraphMAE on 100-nodes networks.">
  </picture>
</p>
</details>

<a class="anchor" id="analysis_stats"></a>
### Statistical measures
<p align="right"><a href="#analysis">🩺</a></p>

---
<a class="anchor" id="structure"></a>
## 🌵 Repository structure
<p align="right"><a href="#top">🔝</a></p>

<a class="anchor" id="references"></a>
## 📚 References
<p align="right"><a href="#top">🔝</a></p>


---
<p align="center">
 • <a href="#quick_start">⚡️</a>
 • <a href="#data">💾</a>
 • <a href="#embed">📍</a>
 • <a href="#analysis">🩺</a>
 • <a href="#structure">🌵</a>
 • <a href="#references">📚</a>
 •
</p>
