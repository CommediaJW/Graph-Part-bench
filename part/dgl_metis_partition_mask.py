import argparse
import torch
import os
import dgl
import DistGNN
import time

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
    parser.add_argument("--num-parts", default=4, type=int)
    args = parser.parse_args()
    print(args)

    start = time.time()
    graph, _ = DistGNN.dataloading.load_dataset(args.root,
                                                args.dataset,
                                                with_feature=False)
    end = time.time()
    print("Loading graph takes {:.3f} s".format(end - start))

    start = time.time()
    dgl_g = dgl.graph(("csc", (graph["indptr"], graph["indices"], [])))
    num_nodes = dgl_g.num_nodes()
    train_mask = torch.zeros((num_nodes, ), dtype=bool)
    train_mask[graph["train_idx"]] = True
    out = dgl.metis_partition(dgl_g,
                              args.num_parts,
                              extra_cached_hops=0,
                              reshuffle=False,
                              balance_ntypes=train_mask,
                              balance_edges=True,
                              mode="recursive")
    end = time.time()
    print("Partition takes {:.3f} s".format(end - start))

    parts_mask_list = []
    train_nids_list = []
    for i in range(args.num_parts):
        part_mask = torch.zeros((num_nodes, ), dtype=bool)
        part_mask[out[i].ndata["_ID"]] = True
        parts_mask_list.append(part_mask)
        local_train_nids = graph["train_idx"][part_mask[graph["train_idx"]]]
        train_nids_list.append(local_train_nids)
        print("Part {:4d}, nodes num = {:12d}, train nids num = {:12d}".format(
            i,
            torch.nonzero(part_mask).numel(), local_train_nids.numel()))

    if args.save_path != None:
        save_path = args.save_path
    else:
        save_path = args.root
    mask_save_fn = os.path.join(
        save_path,
        args.dataset + "_metis_" + str(args.num_parts) + "parts_mask.pkl")
    train_save_fn = os.path.join(
        save_path, args.dataset + "_metis_" + str(args.num_parts) +
        "parts_train_nids.pkl")
    torch.save(parts_mask_list, mask_save_fn)
    torch.save(train_nids_list, train_save_fn)
    print("Result saved to {} and {}".format(mask_save_fn, train_save_fn))
