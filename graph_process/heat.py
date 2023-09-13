import argparse
import torch
import DistGNN
import os
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument('--fan-out', type=str, default='5,10,15')
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument(
        "--dataset",
        default="ogbn-papers100M",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-papers400M"])
    parser.add_argument("--save-path", default=None, type=str)
    args = parser.parse_args()
    print(args)

    start = time.time()
    graph, _ = DistGNN.dataloading.load_dataset(args.root,
                                                args.dataset,
                                                with_feature=False,
                                                with_probs=args.bias)
    end = time.time()
    print("Loading graph takes {:.3f} s".format(end - start))

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]

    start = time.time()
    if args.bias:
        sampling_heat, feat_heat = DistGNN.cache.get_node_heat(
            graph["indptr"],
            graph["indices"],
            graph["train_idx"],
            fan_out,
            probs=graph["probs"])
    else:
        sampling_heat, feat_heat = DistGNN.cache.get_node_heat(
            graph["indptr"], graph["indices"], graph["train_idx"], fan_out)
    torch.cuda.synchronize()
    end = time.time()
    print("Computing heat takes {:.3f} s".format(end - start))

    sampling_heat, feat_heat = sampling_heat.cpu(), feat_heat.cpu()

    if args.save_path != None:
        save_path = args.save_path
    else:
        save_path = args.root

    if args.bias:
        feat_heat_save_fn = os.path.join(
            save_path,
            args.dataset + "_" + args.fan_out + "_biasd_feature_heat.pt")
        sampling_heat_save_fn = os.path.join(
            save_path,
            args.dataset + "_" + args.fan_out + "_biasd_sampling_heat.pt")
    else:
        feat_heat_save_fn = os.path.join(
            save_path, args.dataset + "_" + args.fan_out + "_feature_heat.pt")
        sampling_heat_save_fn = os.path.join(
            save_path, args.dataset + "_" + args.fan_out + "_sampling_heat.pt")

    torch.save(feat_heat, feat_heat_save_fn)
    torch.save(sampling_heat, sampling_heat_save_fn)

    print("Feature heat saved to", feat_heat_save_fn)
    print("Sampling heat saved to", sampling_heat_save_fn)
