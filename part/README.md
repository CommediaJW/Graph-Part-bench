# Graph partition scripts and benchmarks

## dgl_metis_partition_mask.py

Partition a [DistGNN](https://github.com/CommediaJW/Dist-GNN/tree/main)-processed graph with [dgl's metis API](https://docs.dgl.ai/en/1.1.x/generated/dgl.metis_partition.html) (no halo vertices). This script will generate masks of every partition and their corresponding training vertices.
```shell
python3 dgl_metis_partition_mask.py --root ${path to the dataset} --dataset ogbn-papers100M --save-path ${path to save the partitions} --num-parts ${#partitions}
```

## metis_format_partition_mask.py

Read a metis-format partition result file, and generate corresponding partition masks and training vertices.

The tools that can generate metis-format partitions:
- [METIS](https://github.com/KarypisLab/METIS/tree/master)
- [ParMETIS](https://github.com/KarypisLab/ParMETIS/tree/main)
- [XtraPuLP](https://github.com/HPCGraphAnalysis/PuLP/tree/master/xtrapulp/0.3)
```shell
python3 metis_format_partition_mask.py --root ${path to the dataset} --dataset ogbn-papers100M --save-path ${path to save the results} --num-parts ${#partitions} --parts ${path to the metis-format partition result file}
```

## khops_subgraph.py

Read a [DistGNN](https://github.com/CommediaJW/Dist-GNN/tree/main)-processed graph and a group of partition masks and training vertices. With these information, this script will generate a k-hops subgraph, and evaluate the partition quality (if you set `--eval`).
```shell
python3 khops_subgraph.py --root ${path to the dataset} --dataset ogbn-papers100M --train-nids ${path to the input training vertices file} --parts ${path to the input parition masks file} --hops 3 [--eval] [--save-path ${path to save the generated k-hops subgraph}]
```
