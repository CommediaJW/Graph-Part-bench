import argparse
import torch
import dgl
import os
import time
import DistGNN
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument(
        "--dataset",
        default="ogbn-papers100M",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-papers400M"])
    parser.add_argument("--save-path", default=None, type=str)
    parser.add_argument("--train-nids", default=None, type=str)
    parser.add_argument("--hops", default=3, type=int)
    parser.add_argument("--bench", default=False, action="store_true")
    parser.add_argument("--parts", default=None, type=str)
    args = parser.parse_args()
    print(args)

    start = time.time()
    if args.dataset == "ogbn-products":
        graph, num_classes = DistGNN.dataloading.load_dataset(
            args.root, "ogbn-products", with_feature=False)
    elif args.dataset == "ogbn-papers100M":
        graph, num_classes = DistGNN.dataloading.load_dataset(
            args.root, "ogbn-papers100M", with_feature=False)
    end = time.time()
    print("Loading graph takes {:.3f} s".format(end - start))

    indptr = graph["indptr"]
    indices = graph["indices"]
    del graph
    dgl_g = dgl.graph(("csc", (indptr, indices, torch.tensor([]))))

    if args.bench:
        assert args.parts != None
        parts_mask_list = torch.load(args.parts)

    start = time.time()
    train_nids_list = torch.load(args.train_nids)
    subgraph_mask_list = []
    for rank, train_nids in enumerate(train_nids_list):
        if args.bench:
            parts_nids = parts_mask_list[rank].nonzero().flatten()
            subg = dgl.node_subgraph(dgl_g,
                                     parts_nids,
                                     relabel_nodes=False,
                                     store_ids=False)
            print(
                "Rank {:4d}, #seeds = {:12d}, #original_part_nodes = {:12d}, #original_part_edges = {}"
                .format(rank, train_nids.numel(), parts_nids.numel(),
                        subg.num_edges()))

        seeds = train_nids
        total_nids_list = []
        total_nids_list.append(seeds)
        for i in range(args.hops):
            neighbors_list = []
            for nid in seeds:
                head = indptr[nid]
                tail = indptr[nid + 1]
                neighbors_list.append(indices[head:tail])
            neighbors = torch.cat(neighbors_list).unique()
            total_nids_list.append(neighbors)
            total_nids = torch.cat(total_nids_list).unique()

            if args.bench:
                subg = dgl.node_subgraph(dgl_g,
                                         total_nids,
                                         relabel_nodes=False,
                                         store_ids=False)
                print("{:4d} hops subgraph, #nodes = {:12d}, #edges = {:12d}".
                      format(i, total_nids.numel(), subg.num_edges()))
                intersect_nids = torch.from_numpy(
                    np.intersect1d(total_nids.numpy(), parts_nids.numpy()))
                subg = dgl.node_subgraph(dgl_g,
                                         intersect_nids,
                                         relabel_nodes=False,
                                         store_ids=False)
                print(
                    "Intersection with original parts, #nodes = {:12d}, #edges = {:12d}"
                    .format(intersect_nids.numel(), subg.num_edges()))
            seeds = neighbors

        mask = torch.zeros((indptr.numel() - 1, ), dtype=torch.bool)
        mask[total_nids] = True
        subgraph_mask_list.append(mask)
    end = time.time()
    print("Extracting khops subgraph takes {:.3f} s".format(end - start))

    if args.save_path != None:
        save_path = args.save_path
    else:
        save_path = args.root
    save_fn = os.path.join(
        save_path, args.dataset + "_" + str(args.hops) + "hops_" +
        str(len(train_nids_list)) + "parts_subgraph_mask.pkl")
    torch.save(subgraph_mask_list, save_fn)
    print("Result saved to", save_fn)
