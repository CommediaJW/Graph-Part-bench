from DistGNN.dataloading import load_dataset
import argparse
import os
import time
import torch

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

    lines = [[] for _ in range(num_nodes)]
    accessed = set([])  # to remove duplicate edges
    num_edges = 0
    start = time.time()
    for nid in range(num_nodes):
        if train_mask[nid].item() == True:
            lines[nid].insert(0, "1")
            lines[nid].insert(0, "0")
        else:
            lines[nid].insert(0, "0")
            lines[nid].insert(0, "1")
        for i in range(indptr[nid].item(), indptr[nid + 1].item()):
            neighbor = indices[i].item()
            if ((nid, neighbor) not in accessed) and ((neighbor, nid)
                                                      not in accessed):
                # metis requires undirected graph
                lines[nid].append(str(neighbor + 1))
                lines[neighbor].append(str(nid + 1))
                accessed.add((nid, neighbor))
                accessed.add((neighbor, nid))
                num_edges += 1
        if (nid + 1) % 200000 == 0:
            print("[Generate lines] processed {} nodes.".format(nid + 1))

    filename = os.path.join(args.save_path, args.dataset + "_metis.graph")
    f = open(filename, "w")
    hearder = "{} {} 010 2\n".format(num_nodes, num_edges)
    f.write(hearder)

    for it, line in enumerate(lines):
        line = " ".join(line) + "\n"
        f.write(line)
        if (it + 1) % 200000 == 0:
            print("[Write] processed {} nodes.".format(nid + 1))

    end = time.time()
    print("Generate metis graph takes {:.3f} s".format(end - start))

    f.close()
