from DistGNN.dataloading import load_dataset
import argparse
import os
import time
import torch
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument("--dataset",
                        default="ogbn-papers100M",
                        choices=["ogbn-products", "ogbn-papers100M"])
    parser.add_argument("--save-path", type=str, default=".")
    args = parser.parse_args()

    graph, _ = load_dataset(args.root,
                            args.dataset,
                            with_feature=False,
                            with_probs=True)
    indptr = graph["indptr"]
    indices = graph["indices"]
    num_nodes = indptr.numel() - 1
    train_mask = torch.zeros((num_nodes, ), dtype=torch.bool)
    train_mask[graph["train_idx"]] = True
    num_edges = 0
    start = time.time()
    wholefile = ""
    with tqdm(total=indptr[num_nodes].item()) as pbar:
        for nid in range(num_nodes):
            if train_mask[nid].item() == True:
                linestr = "0 1 "
            else:
                linestr = "1 0 "
            ecnt = (indptr[nid + 1] - indptr[nid]).item()
            line = (indices[indptr[nid]:indptr[nid + 1]] + 1).tolist()
            linestr += " ".join(map(str, line)) + "\n"
            wholefile += linestr

            num_edges += ecnt
            pbar.update(ecnt)

    filename = os.path.join(args.save_path, args.dataset + "_xtrapulp.graph")
    f = open(filename, "w")
    hearder = "{} {} 010 2\n".format(num_nodes, num_edges)
    f.write(hearder)

    print("[Write]")
    f.write(wholefile)

    end = time.time()
    print("Generate metis graph takes {:.3f} s".format(end - start))

    f.close()
