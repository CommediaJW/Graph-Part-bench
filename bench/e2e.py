import argparse
import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import time
import numpy as np
from utils.models import SAGE

import DistGNN
from DistGNN.cache import get_node_heat, get_cache_nids_selfish, get_structure_space, get_feature_space
from DistGNN.dataloading import SeedGenerator
from DistGNN.dist import create_communicator


def build_blocks(batch):
    blocks = []
    for layer in batch:
        seeds, frontier, coo_row, coo_col = layer
        block = dgl.create_block((coo_col, coo_row),
                                 num_src_nodes=frontier.numel(),
                                 num_dst_nodes=seeds.numel())
        block.srcdata[dgl.NID] = frontier
        block.dstdata[dgl.NID] = seeds
        blocks.insert(0, block)
    return blocks


def run(data, args):
    graph, num_classes, train_nids_list = data

    torch.cuda.set_device(0)
    dist.init_process_group('nccl',
                            'tcp://127.0.0.1:12347',
                            world_size=1,
                            rank=0)
    create_communicator(1)

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]

    bandwidth_gpu = 120.62
    bandwidth_host = 8.32
    sampling_read_bytes_gpu = 480
    sampling_read_bytes_host = 480
    feature_read_bytes_gpu = 480
    feature_read_bytes_host = 512

    parts_iteration_time_log = []
    parts_sampling_time_log = []
    parts_loading_time_log = []
    parts_training_time_log = []

    for i, train_nids in enumerate(train_nids_list):
        print("Part {:3d}, training nids num = {:10d}".format(
            i, train_nids.numel()))

        torch.cuda.empty_cache()
        if args.bias:
            probs_key = "probs"
            probs = graph["probs"]
            sampling_heat, feature_heat = get_node_heat(graph["indptr"],
                                                        graph["indices"],
                                                        train_nids,
                                                        fan_out,
                                                        probs=graph[probs_key])
        else:
            probs_key = None
            probs = torch.Tensor()
            sampling_heat, feature_heat = get_node_heat(
                graph["indptr"], graph["indices"], train_nids, fan_out)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # create model
        model = SAGE(graph["features"].shape[1], 256, num_classes,
                     len(fan_out))
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        # create dataloader
        train_dataloader = SeedGenerator(train_nids.cuda(),
                                         args.batch_size,
                                         shuffle=True)

        available_mem = args.cache_budget * 1024 * 1024 * 1024

        # get cache nids
        sampling_cache_nids, feature_cache_nids = get_cache_nids_selfish(
            graph,
            sampling_heat,
            feature_heat,
            available_mem,
            bandwidth_gpu,
            sampling_read_bytes_gpu,
            feature_read_bytes_gpu,
            bandwidth_host,
            sampling_read_bytes_host,
            feature_read_bytes_host,
            probs=probs_key)
        del sampling_heat, feature_heat

        print(
            "Part {:3d}, sampling cache nids num = {:10d}, cache size = {:5.2f} GB"
            .format(
                i, sampling_cache_nids.numel(),
                torch.sum(
                    get_structure_space(
                        sampling_cache_nids, graph, probs=probs_key)) / 1024 /
                1024 / 1024))
        print(
            "Part {:3d}, feature cache nids num = {:10d}, cache size = {:5.2f} GB"
            .format(
                i, feature_cache_nids.numel(),
                get_feature_space(graph) * feature_cache_nids.numel() / 1024 /
                1024 / 1024))

        for key in graph:
            DistGNN.capi.ops._CAPI_tensor_pin_memory(graph[key])

        # cache
        torch.cuda.empty_cache()
        sampler = DistGNN.capi.classes.P2PCacheSampler(graph["indptr"],
                                                       graph["indices"], probs,
                                                       sampling_cache_nids, 0)

        torch.cuda.empty_cache()
        feature_server = DistGNN.capi.classes.P2PCacheFeatureServer(
            graph["features"], feature_cache_nids, 0)

        iteration_time_log = []
        sampling_time_log = []
        loading_time_log = []
        training_time_log = []
        model.train()
        for it, seed_nids in enumerate(train_dataloader):
            torch.cuda.synchronize()
            sampling_start = time.time()
            batch = sampler._CAPI_sample_node_classifiction(
                seed_nids, fan_out, False)
            blocks = build_blocks(batch)
            torch.cuda.synchronize()
            sampling_end = time.time()

            loading_start = time.time()
            batch_inputs = feature_server._CAPI_get_feature(
                blocks[0].srcdata[dgl.NID])
            batch_labels = DistGNN.capi.ops._CAPI_cuda_index_select(
                graph["labels"], seed_nids)
            torch.cuda.synchronize()
            loading_end = time.time()

            training_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = F.cross_entropy(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            training_end = time.time()

            sampling_time_log.append(sampling_end - sampling_start)
            loading_time_log.append(loading_end - loading_start)
            training_time_log.append(training_end - training_start)
            iteration_time_log.append(training_end - sampling_start)

        sampling_time, loading_time, training_time, iteration_time = np.mean(
            sampling_time_log[3:]) * 1000, np.mean(
                loading_time_log[3:]) * 1000, np.mean(
                    training_time_log[3:]) * 1000, np.mean(
                        iteration_time_log[3:]) * 1000
        parts_sampling_time_log.append(sampling_time)
        parts_loading_time_log.append(loading_time)
        parts_training_time_log.append(training_time)
        parts_iteration_time_log.append(iteration_time)

        print(
            "Part {:3d} | Sampling {:8.3f} ms | Loading {:8.3f} ms | Training {:8.3f} ms | Iteration {:8.3f} ms"
            .format(i, sampling_time, loading_time, training_time,
                    iteration_time))

        for key in graph:
            DistGNN.capi.ops._CAPI_tensor_unpin_memory(graph[key])
        del sampler, feature_server

    print(
        "Sampling: min {:8.3f} ms, max {:8.3f} ms, avg {:8.3f} ms, std {:8.3f} ms"
        .format(np.min(parts_sampling_time_log),
                np.max(parts_sampling_time_log),
                np.average(parts_sampling_time_log),
                np.std(parts_sampling_time_log)))
    print(
        "Loading: min {:8.3f} ms, max {:8.3f} ms, avg {:8.3f} ms, std {:8.3f} ms"
        .format(np.min(parts_loading_time_log), np.max(parts_loading_time_log),
                np.average(parts_loading_time_log),
                np.std(parts_loading_time_log)))
    print(
        "Training: min {:8.3f} ms, max {:8.3f} ms, avg {:8.3f} ms, std {:8.3f} ms"
        .format(np.min(parts_training_time_log),
                np.max(parts_training_time_log),
                np.average(parts_training_time_log),
                np.std(parts_training_time_log)))
    print(
        "Iteration: min {:8.3f} ms, max {:8.3f} ms, avg {:8.3f} ms, std {:8.3f} ms"
        .format(np.min(parts_iteration_time_log),
                np.max(parts_iteration_time_log),
                np.average(parts_iteration_time_log),
                np.std(parts_iteration_time_log)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        default='dataset/',
                        help='Path of the dataset.')
    parser.add_argument(
        "--dataset",
        default="ogbn-papers100M",
        choices=["ogbn-products", "ogbn-papers100M", "ogbn-papers400M"])
    parser.add_argument("--train-nids", default=None, type=str)
    parser.add_argument("--batch-size",
                        default="1000",
                        type=int,
                        help="The number of seeds of sampling.")
    parser.add_argument('--fan-out', type=str, default='5,10,15')
    parser.add_argument('--bias',
                        action='store_true',
                        default=False,
                        help="Sample with bias.")
    parser.add_argument('--cache-budget',
                        type=float,
                        default=1,
                        help="Unit: GB")
    args = parser.parse_args()
    print(args)

    torch.manual_seed(1)

    start = time.time()
    graph, num_classes = DistGNN.dataloading.load_dataset(args.root,
                                                          args.dataset,
                                                          with_probs=args.bias)
    end = time.time()
    print("Loading graph takes {:.3f} s".format(end - start))
    train_nids_list = torch.load(args.train_nids)

    graph["labels"] = graph["labels"].long()
    data = graph, num_classes, train_nids_list

    run(data, args)
