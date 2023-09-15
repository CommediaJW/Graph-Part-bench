# Graph process scripts

## heat.py

Compute sampling & feature heat. The input dataset is processed by [DistGNN](https://github.com/CommediaJW/Dist-GNN/tree/main).
```shell
python3 heat.py --root ${path to the dataset} --dataset ogbn-papers100M --fan-out 5,10,15 [--bias]
```

## drop.py

Drop the vertices whose feature heat is equal to 0, relabel the rest vertices and edges and generate a new graph.
```shell
python3 drop.py --root ${path to the dataset} --dataset ogbn-papers100M [--bias] --save-path ${path to save the new graph} --feat-heat ${path to the feature heat file} --adj-heat ${path to the sampling heat file} 
```

## metis_format.py & weighted_metis_format.py

Convert a [DistGNN](https://github.com/CommediaJW/Dist-GNN/tree/main) dataset into a [metis format](http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf) graph. `weighted_metis_format.py` will assign edge weights to the graph.
```shell
python3 weighted_metis_format.py --root ${path to the dataset} --dataset ogbn-papers100M --save-path ${path to save the graph} [--weight-save-path ${path to save the edge weight file}] --heat ${path to the heat file}
```

```shell
python3 metis_format.py --root ${path to the dataset} --dataset ogbn-papers100M --save-path ${path to save the graph}
```

## xtrapulp_binary_format.py

Convert a DistGNN dataset into a [XtraPuLP binary format](https://github.com/HPCGraphAnalysis/PuLP/tree/master/xtrapulp/0.3) graph.

```shell
python3 xtrapulp_binary_format.py --root ${path to the dataset} --dataset ogbn-papers100M --save-path ${path to save the graph}
```
