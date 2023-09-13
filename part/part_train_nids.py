import argparse
import torch
import os

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
    parser.add_argument("--parts", default=None, type=str)
    args = parser.parse_args()
    print(args)

    train_nids = torch.load(os.path.join(args.root, "train_idx.pt"))
    parts_mask_list = torch.load(args.parts)
    assert torch.max(train_nids) < parts_mask_list[0].numel()

    train_nids_list = []
    num_parts = len(parts_mask_list)
    for i, part_mask in enumerate(parts_mask_list):
        local_train_nids = train_nids[part_mask[train_nids]]
        print("Part {:4d}, nodes num = {:12d}, train nids num = {:12d}".format(
            i,
            torch.nonzero(part_mask).numel(), local_train_nids.numel()))
        train_nids_list.append(local_train_nids)

    if args.save_path != None:
        save_path = args.save_path
    else:
        save_path = args.root
    save_fn = os.path.join(
        save_path,
        args.dataset + "_" + str(num_parts) + "parts_train_nids.pkl")
    torch.save(train_nids_list, save_fn)
    print("Result saved to", save_fn)
