<a class="anchor" id="top"></a>
# Network Symbolic Regression


This repository is cloned from the [`synthetic` repository <img src="https://github.com/fluidicon.png" height=14>](https://github.com/telmomenezes/synthetic). It contains a python implementation of the Generative Network Symbolic Regression algorithm.

💡 *For more information regarding the code, please refer to the original repository.*

## Novelty

The main novelty brought here is the introduction of [*deep distances*](./deep_distance.py). Graph Representation Learning methods are probed to alleviate the original hand-engineered distance metric.

Within this new framework, the distance between networks is based on their representation in an embedding space resulting from the Graph Representation methods. The distance is computed using a distance metric (eg. euclidian, cosine, Manhattan, etc.) between the corresponding networks' coordinate within the chosen representation space.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="../illustrations/dist_illust_light.png">
  <source media="(prefers-color-scheme: dark)" srcset="../illustrations/dist_illust_dark.png">
  <img alt="Illustration - Distance in Representation Space.">
</picture>
