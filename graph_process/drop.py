import argparse
import torch
import DistGNN
import os
import time
import dgl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument(
        "--dataset",
        default="ogbn-papers100M",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-papers400M"])
    parser.add_argument("--save-path", default=None, type=str)
    parser.add_argument("--feat-heat", default=None, type=str)
    parser.add_argument("--adj-heat", default=None, type=str)
    args = parser.parse_args()
    print(args)

    assert args.feat_heat != None and args.adj_heat != None
    feat_heat = torch.load(args.feat_heat)
    adj_heat = torch.load(args.adj_heat)
    nids = torch.nonzero(feat_heat).flatten()

    start = time.time()
    graph, _ = DistGNN.dataloading.load_dataset(args.root,
                                                args.dataset,
                                                with_feature=False,
                                                with_probs=args.bias)
    end = time.time()
    print("Loading graph takes {:.3f} s".format(end - start))

    train_mask = torch.zeros((feat_heat.numel(), ), dtype=torch.bool)
    train_mask[graph["train_idx"]] = True

    dgl_graph = dgl.graph(
        ("csc", (graph["indptr"], graph["indices"], torch.tensor([]))))
    dgl_graph.ndata["labels"] = graph["labels"]
    dgl_graph.ndata["train_mask"] = train_mask
    if args.bias:
        dgl_graph.edata["probs"] = graph["probs"]
    dgl_graph.ndata["feat_heat"] = feat_heat
    dgl_graph.ndata["adj_heat"] = adj_heat

    indptr, indices, eid = dgl_graph.adj_tensors('csc')

    del graph

    assert dgl_graph.num_nodes() == feat_heat.numel()

    start = time.time()
    processed_graph = dgl.node_subgraph(dgl_graph,
                                        nids,
                                        relabel_nodes=True,
                                        store_ids=True)
    end = time.time()
    print("Dropping nodes takes {:.3f} s".format(end - start))

    print(processed_graph)

    indptr, indices, eid = processed_graph.adj_tensors('csc')
    labels = processed_graph.ndata["labels"]
    train_idx = torch.nonzero(processed_graph.ndata["train_mask"]).flatten()
    if args.bias:
        probs = processed_graph.edata["probs"][eid]

    print("Save processed graph...")
    start = time.time()
    meta_data = {
        "dataset": args.dataset,
        "num_nodes": dgl_graph.num_nodes(),
        "num_edges": dgl_graph.num_edges(),
        "num_classes": labels[~torch.isnan(labels)].unique().numel(),
        "num_train_nodes": train_idx.numel()
    }
    torch.save(meta_data, os.path.join(args.save_path, "metadata.pt"))
    torch.save(train_idx, os.path.join(args.save_path, "train_idx.pt"))
    torch.save(indptr, os.path.join(args.save_path, "indptr.pt"))
    torch.save(indices, os.path.join(args.save_path, "indices.pt"))
    torch.save(labels, os.path.join(args.save_path, "labels.pt"))
    torch.save(processed_graph.ndata["feat_heat"],
               os.path.join(args.save_path, "feat_heat.pt"))
    torch.save(processed_graph.ndata["adj_heat"],
               os.path.join(args.save_path, "adj_heat.pt"))
    if args.bias:
        torch.save(probs, os.path.join(args.save_path, "probs.pt"))
    end = time.time()
    print("Saving graph takes {:.3f} s".format(end - start))
