<a class="anchor" id="replication"></a>
# Replication

The symbolic regression is re-encapsulated here. You can run symbolic regression from the following command line:

```shell
python ./reproducibility/evolutionary_search.py
```

Default arguments are stored in the `.py` file. They can also be parsed from command line interface:

| Arguments | Description | Default |
| :---: | :--- | :---: |
|`--output_path`| path to store the outputs of the regression process (by default: the parameters of the experiment in `params.txt`, the best found generator in `bestprog.txt`, its related network instance in `bestnet.gml`, and `evo.csv` which tracks the evolution of the loss and stores computational time) | `data/symbolic_regression_reproduction/` |
| `--verbosity` | level of verbosity (minimal: 0, show additional information during process:1) | `0`|
| `--save_all` | save each of the best found generator and associated network instance along the process (0:no, 1:yes) | `0` |
| `--data_path` | path to the folder containing the target network(s) | `data/space_seg_data/v100_e552/ER/` |
| `--inets` | name of the target netwok(s) files | `c1_1.pickle`, `c1_2.pickle` and `c1_3.pickle` |
| `--directed` | directed graphs or not | `False` |
| `--n_rep` | number of run(s) per target network | `10` |
| `--do_parallel` | run mutliple symbolic regression processes in parallel (not supported for deep-distance strategies) | `False` |
| `--n_processes` | number of parallel processes | `10` |
| `--tolerance` | tolerance window of the regression algorithm | Default `synthetic` tolerance |
| `--gens` | number of stable generations needed for the algorithm to stop | Default `synthetic` gens |
| `--sr` | sample weight for the generation procedure | Default `synthetic` sr |
| `--gen_type` | generation type | Default `synthetic` gen type |
| `--model_path` | if using deep-distance strategy: path toward the embedder model | Default `synthetic` tolerance | `./DL_module/saved/best_models/trad_degs/` |
| `--norm_samples` | if using traditional approach: number of ER networks to perform dissimilarity distance normalisation | Default `synthetic` number of normalisation samples |
| `--bins` | if using traditional approach: number of bins used in the histograms to compute dissimilarities | Default `synthetic` bins |
| `--max_dist` | maximal distance value (cut-off) | Default `synthetic` max distance |
| `--rw` | use random walk distance | `True` |
| `--force_trad` | force to use the traditional approach | `False` |



<details><summary>📝 Example run & outputs</summary><br/>

For instance to perform symbolic regression once on the network named `'k_1.pickle'` in the folder `'data/space_seg_data/v100_e552/PA/'`, with the PGCL based (saved at `'./DL_module/saved/best_models/PGCL/'`) distance, and to store the outcome at `data/repro_PGCL_PA/`, go to the root folder and run:

```shell
python ./reproducibility/evolutionary_search.py\
  --model_path './DL_module/saved/best_models/PGCL/'\
  --data_path 'data/space_seg_data/v100_e552/PA/'\
  --inets 'k_1.pickle'\
  --output_path './data/repro_PGCL_PA/'\
  --n_rep 1
```

The outcome will be saved in the given folder and the following kind of output is awaited:

```
Nets files :  ['k_1.pickle']
🌱 Setting seed to 0
Initialising distance metric...
Stable gens:  12%|██                   | 121/1000 [00:27<03:39,  4.01it/s, #=7, loss=0.00555, size=21]
```

When the computation is finished it will output information about the best found generator (the one with lowest associated loss) over the different runs for each given network. In the example only one run was undertook over one network instance, the output is:
```
{'k_1': {'best_run_id': 0, 'best_run_path': './data/repro_PGCL_PA/k_1.0/', 'best_fit': 0.0055505693890154, 'n_runs': 1}}
```

</details>

<!--

```shell
python ./reproducibility/evolutionary_search.py \
  --output_path 'data/repro_words/GraphMAE_vGCN_o64/'\
  --model_path './DL_module/saved/best_models/GraphMAE_GCNversion_o64/'\
  --data_path './data/data_2013/obs/'\
  --inets 'words.gml'\
  --n_rep 30
```

```shell
python ./reproducibility/evolutionary_search.py \
  --output_path 'data/repro_save_d/GraphMAE_vGCN_o64/'\
  --model_path './DL_module/saved/best_models/GraphMAE_GCNversion_o64/'\
  --data_path 'space_seg/data/data_v2/v100_e552/d/'\
  --inets 'd_1.pickle'\
  --save_all 1\
  --seed 1\
  --n_rep 1
```

```shell
python ./reproducibility/evolutionary_search.py \
  --output_path 'data/repro_save_k/trad2014/'\
  --data_path 'space_seg/data/data_v2/v100_e552/PA/'\
  --force_trad\
  --rw\
  --inets 'k_1.pickle'\
  --save_all 1\
  --seed 4\
  --n_rep 1
```

```shell
python ./reproducibility/evolutionary_search.py \
  --output_path 'data/repro_words/trad2014/'\
  --data_path './data/data_2013/obs/'\
  --inets 'words.gml'\
  --force_trad\
  --rw\
  --n_rep 30\
  --do_parallel\
  --n_processes 2
```


<a class="anchor" id="deep_repro"></a>
# Deep reproducibility
<p align="right"><a href="#top">🔝</a></p>

-->
