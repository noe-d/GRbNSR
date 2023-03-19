<a class="anchor" id="top"></a>
# Reproducibility

<a class="anchor" id="replication"></a>
# Replication
<p align="right"><a href="#top">🔝</a></p>

```shell
python ./reproducibility/evolutionary_search.py
```

TODO:
- [ ] Describe arguments that can be parsed
- [ ] Implement parser

Default arguments are stored in the `.py` file. They can also be parsed from command line interface:
- `--output_path`
- `--verbosity`
- `--save_all`
- `--data_path`
- `--inets`
- `--directed`
- `--n_rep`
- `--do_parallel`
- `--n_processes`
- `--tolerance`
- `--gens`
- `--sr`
- `--gen_type`
- `--model_path`
- `--dist_type`
- `--norm_samples`
- `--bins`
- `--max_dist`
- `--rw`: random walk distance (default: `True`)
- `--force_trad`

<details><summary>📝 Example run & outputs</summary><br/>

```shell
python ./reproducibility/evolutionary_search.py \
  --model_path './DL_module/saved/best_models/trad_degs/' \
  --data_path 'space_seg/data/data_v2/v100_e552/PA/' \
  --inets 'k_1.pickle' \
  --n_rep 1
```

</details>

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
