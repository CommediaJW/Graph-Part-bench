import argparse
import torch
import time
import DistGNN
from DistGNN.cache import get_structure_space, get_feature_space
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
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument("--parts-mask", default=None, type=str)
    args = parser.parse_args()
    print(args)

    parts = torch.load(args.parts_mask)

    start = time.time()
    graph, _ = DistGNN.dataloading.load_dataset(args.root,
                                                args.dataset,
                                                with_probs=args.bias)
    end = time.time()
    print("Loading graph takes {:.3f} s".format(end - start))

    if args.bias:
        probs_key = "probs"
    else:
        probs_key = None

    adj_log = []
    feat_log = []
    for i, mask in enumerate(parts):
        nids = torch.nonzero(mask).flatten()
        adj_size = torch.sum(get_structure_space(
            nids, graph, probs=probs_key)).item() / 1024 / 1024 / 1024
        feat_size = get_feature_space(
            graph) * nids.numel() / 1024 / 1024 / 1024
        print(
            "Part {:3d} | Structure size {:6.3f} GB | Feature size {:6.3f} GB".
            format(i, adj_size, feat_size))
        adj_log.append(adj_size)
        feat_log.append(feat_size)

    print("Structure size: min {:6.3f} GB, max {:6.3f} GB, avg {:6.3f} GB".
          format(np.min(adj_log), np.max(adj_log), np.mean(adj_log)))
    print(
        "Feature size: min {:6.3f} GB, max {:6.3f} GB, avg {:6.3f} GB".format(
            np.min(feat_log), np.max(feat_log), np.mean(feat_log)))
