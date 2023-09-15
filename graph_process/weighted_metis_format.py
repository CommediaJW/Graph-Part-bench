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
    parser.add_argument("--weight-save-path", type=str, default=None)
    parser.add_argument("--heat", type=str, default=None)
    args = parser.parse_args()

    graph, _ = load_dataset(args.root,
                            args.dataset,
                            with_feature=False,
                            with_probs=True)
    heat = torch.load(args.heat)

    indptr = graph["indptr"]
    indices = graph["indices"]
    weight = graph["probs"]
    num_nodes = indptr.numel() - 1

    start = time.time()
    for nid in range(num_nodes):
        weight[indptr[nid]:indptr[nid + 1]] *= heat[nid]
        if (nid + 1) % 200000 == 0:
            print("[Generating weight] processed {} nodes.".format(nid + 1))
    end = time.time()
    print("Generating edge weights takes {:.3f} s".format(end - start))
    if args.weight_save_path != None:
        weight_save_fn = os.path.join(args.weight_save_path,
                                      args.dataset + "_edge_weight.pkl")
        torch.save(weight, weight_save_fn)
        print("Edge weight saved to", weight_save_fn)
    # weight = torch.load(
    #     "/data/graph_part_bench/ogbn-papers100M-dropped/ogbn-papers100M_edge_weight.pkl"
    # )

    weight = (weight * indices.numel() * 2 / torch.sum(weight) + 1).long()
    print(torch.max(weight))
    print(torch.min(weight))
    print(torch.sum(weight))
    print(torch.sum(weight) / weight.numel())

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
                lines[nid].append(str(weight[i].item()))
                lines[neighbor].append(str(nid + 1))
                lines[neighbor].append(str(weight[i].item()))
                accessed.add((nid, neighbor))
                accessed.add((neighbor, nid))
                num_edges += 1
        if (nid + 1) % 200000 == 0:
            print("[Generate lines] processed {} nodes.".format(nid + 1))

    filename = os.path.join(args.save_path,
                            args.dataset + "_metis_weighted.graph")
    f = open(filename, "w")
    hearder = "{} {} 011 2\n".format(num_nodes, num_edges)
    f.write(hearder)

    for it, line in enumerate(lines):
        line = " ".join(line) + "\n"
        f.write(line)
        if (it + 1) % 200000 == 0:
            print("[Write] processed {} nodes.".format(nid + 1))

    end = time.time()
    print("Generate metis graph takes {:.3f} s".format(end - start))

    f.close()
