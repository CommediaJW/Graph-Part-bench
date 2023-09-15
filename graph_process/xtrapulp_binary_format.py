from DistGNN.dataloading import load_dataset
import argparse
import os
import time
import struct

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

    graph, _ = load_dataset(args.root, args.dataset, with_feature=False)

    indptr = graph["indptr"]
    indices = graph["indices"]
    num_nodes = indptr.numel() - 1
    num_edges = indices.numel()

    filename = os.path.join(args.save_path, args.dataset + "_xtrapulp.bin")
    f = open(filename, "wb", 0)

    start = time.time()
    for nid in range(num_nodes):
        neighbor_list = indices[indptr[nid]:indptr[nid + 1]].tolist()
        for neighbor in neighbor_list:
            f.write(struct.pack("<I", neighbor))
            f.write(struct.pack("<I", nid))
        if (nid + 1) % 200000 == 0:
            print("processed {} nodes.".format(nid + 1))
    end = time.time()

    print("Processing graph takes {:.3f} s".format(end - start))

    f.close()
