import json
import argparse
import os
import torch
import random
from ltr.dataset import Lasot, Got10k, TrackingNet


def parse_args():
    parser = argparse.ArgumentParser(description='Vanilla sampling key frames')

    # Dataset and split
    # Examples: "LaSOT" + "train", "GOT-10k" + "vottrain", "TrackingNet" + "0,1,2,3"/"0-3"
    parser.add_argument('--dataset', type=str, help='training dataset')
    parser.add_argument('--split', type=str, help='dataset split')
    # Configs for sampling
    parser.add_argument('--mode', type=str, default='key_frames', help='sampling mode')
    parser.add_argument('--num', type=int, default=3, help='number of key frames')
    parser.add_argument('--use_all_anno', action='store_true', help='do not use human annotation only')
    # Other arguments
    parser.add_argument('--no_parse', action='store_true', help='whether to parse datasets')
    parser.add_argument('--merge', action='store_true', help='whether to merge all')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite current sampling result')

    args = parser.parse_args()
    return args


def get_dataset(dataset_name, split_string):
    if dataset_name == "LaSOT":
        split_string = "train" if split_string == "" else split_string
        lasot_train = Lasot(split=split_string)
        return lasot_train
    elif dataset_name == "GOT-10k":
        split_string = "vottrain" if split_string == "" else split_string
        got10k_train = Got10k(split=split_string)
        return got10k_train
    elif dataset_name == "TrackingNet":
        split_string = "0-3" if split_string == "" else split_string
        trackingnet_train = TrackingNet(split=split_string)
        return trackingnet_train
    else:
        raise NotImplementedError("Dataset {} is not supported yet.".format(dataset_name))


def decode_split(dataset_name, split_string):
    if dataset_name in ["GOT-10k", "LaSOT"]:
        return [split_string]

    if "-" in split_string:
        partitions = split_string.split(",")
        set_ids = []
        for part in partitions:
            if '-' in part:
                left_id, right_id = int(part.split('-')[0]), int(part.split('-')[1])
                set_ids.extend(list(range(left_id, right_id + 1)))
            else:
                set_ids.append(int(part))
    else:
        set_ids = [int(s) for s in split_string.split(",")]
    return set_ids


def _sample_visible_ids(visible, num_ids=1, min_id=None, max_id=None):
    """
    Sampling num_ids frames between min_id and max_id for which target is visible

    args:
        visible - 1d Tensor indicating whether target is visible for each frame
        num_ids - number of frames to be samples
        min_id - Minimum allowed frame number
        max_id - Maximum allowed frame number

    returns:
        list - List of sampled frame numbers. None if not sufficient visible frames could be found.
    """
    if num_ids == 0:
        return []
    if min_id is None or min_id < 0:
        min_id = 0
    if max_id is None or max_id > len(visible):
        max_id = len(visible)

    valid_ids = [i for i in range(min_id, max_id) if visible[i]]

    # No visible ids
    if len(valid_ids) <= 1:
        return None

    return random.choices(valid_ids, k=num_ids)


def sample_frames(dataset, args, set_id):
    """
    Sample N (typically 2, 3, 5) key frames with exact human annotations from videos
    """

    # Check sampling mode
    sampling_mode = args.mode
    key_frames_num = args.num
    use_human_anno = not args.use_all_anno

    if sampling_mode not in ["key_frames"]:
        raise NotImplementedError("Not supporting sampling mode {}".format(sampling_mode))

    # Init an empty dict for sampling result
    simplify_dict = {"key_frames": "kf"}
    name = "{}_{}_{}_{:03d}".format(dataset.get_name(), str(set_id), simplify_dict[sampling_mode], key_frames_num)
    res = {"fs_settings": {"name": name, "dataset": dataset.get_name(), "split": str(set_id),
                           "mode": sampling_mode, "num": key_frames_num}}
    sampling_result_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        "..", "sampling_results", "{}.json".format(name))

    # Check overwrite tag
    if os.path.exists(sampling_result_file) and not args.overwrite:
        print("Sampling result file already exists, and you do not turn on --overwrite. Stop sampling!")
        return

    # Do statistics for bounding box number
    dataset_bounding_boxes = 0
    failure_seqs = []
    sparse_sample_result = dict()

    # Sample sparse annotated frames all every video sequences
    for seq_id in range(0, dataset.get_num_sequences()):

        # Get sequence name
        seq_name = dataset.sequence_list[seq_id]
        if args.dataset == "TrackingNet":
            seq_name_formatted = "{}_{:02d}_{}".format(dataset.get_name(), seq_name[0], seq_name[1])
        elif args.dataset == "LaSOT":
            seq_name_formatted = "{}_{}".format(dataset.get_name(), seq_name)
        else:
            seq_name_formatted = seq_name

        seq_info_dict = dataset.get_sequence_info(seq_id)
        visible_info = seq_info_dict['visible']
        # The length of the video
        seq_length = len(visible_info)
        # Sampled frames
        sampled_frames_visible = []

        # Mode 'key_frames' means sampling N separated labeled frames from whole video (typically 2, 3, 5)
        # Note that here we only sample 'visible' frames from videos
        # This setting is reasonable, as human never annotate on frames where the target does not exist
        for key_id in range(key_frames_num):

            # Now begin to sample frames with visible targets
            # Each key_id takes over sampling in a sub-part of the video
            max_gap = max(1, seq_length // (key_frames_num * 8))
            while True:
                base_frame = min(seq_length - 1, max(0, key_id * seq_length // key_frames_num +
                                                     seq_length // (key_frames_num * 2)))
                key_frame_id = _sample_visible_ids(visible_info, num_ids=1,
                                                   min_id=base_frame - max_gap,
                                                   max_id=base_frame + max_gap)

                if key_frame_id is not None:
                    # Note: Since TrackingNet gives automatic labels together with
                    #       human annotations (in around 1 FPS), please turn on "only_human_anno" to
                    #       disable sampling automatic labels as key frames for TrackingNet.
                    if dataset.get_name() == 'TrackingNet' and use_human_anno:
                        chosen_frame_id = key_frame_id[0]
                        while True:
                            chosen_anno = seq_info_dict['bbox'][chosen_frame_id]
                            total_round_difference = float(torch.sum(
                                torch.abs(chosen_anno - torch.round(chosen_anno))))
                            # Empirically, coordinates formatted [xx.00] * 4 infer manual label.
                            # We once emailed the authors of TrackingNet, and the above
                            #          methodology of filtering manual labels was confirmed by them.
                            if abs(total_round_difference) <= 0.01 or chosen_frame_id == 0:
                                break
                            else:
                                chosen_frame_id -= 1
                    else:
                        chosen_frame_id = key_frame_id[0]
                    sampled_frames_visible.append(chosen_frame_id)
                    break

                max_gap += max(1, seq_length // (key_frames_num * 8))
                if max_gap > seq_length:
                    break

        # Cache the sequence frame sampling result
        if len(sampled_frames_visible) != key_frames_num:
            failure_seqs.append(seq_name_formatted)
        else:
            # Remove duplicate frame sampling
            sampled_frames_visible = list(set(sampled_frames_visible))
            sampled_frames_visible.sort()
            dataset_bounding_boxes += len(sampled_frames_visible)
            sparse_sample_result[seq_name_formatted] = [{"f_ids": sampled_frames_visible}]

    print("Bounding box number for dataset {} is {}.".format(dataset.get_name(), dataset_bounding_boxes))

    res['failure_seqs'] = failure_seqs
    res['sparse_sample_result'] = sparse_sample_result
    json.dump(res, open(sampling_result_file, 'w'), indent=4, sort_keys=True)


def merge_split_results(args, set_ids):
    # Init an empty dict for sampling result
    simplify_dict = {"key_frames": "kf"}
    merged_name = "{}_{}_{}_{:03d}".format(args.dataset, args.split,
                                           simplify_dict[args.mode], args.num)
    merged_res = {"name": merged_name, "dataset": args.dataset,
                  "split": args.split, "mode": args.mode, 'num': args.num}
    merged_sampling_result_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               "..", "sampling_results", "{}.json".format(merged_name))

    # Check overwrite tag
    if os.path.exists(merged_sampling_result_file) and not args.overwrite:
        print("Sampling result file already exists, and you do not turn on --overwrite. Stop sampling!")
        return

    failure_seqs = []
    sampling_results = dict()

    for set_id in set_ids:
        # Load the sampled results
        split_name = "{}_{}_{}_{:03d}".format(args.dataset, str(set_id),
                                              simplify_dict[args.mode], args.num)
        sampling_result_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "..", "sampling_results", "{}.json".format(split_name))
        loaded_sampling_result = json.load(open(sampling_result_file, 'r'))

        # Split valid check
        assert loaded_sampling_result['fs_settings']['dataset'] == merged_res['dataset']
        assert loaded_sampling_result['fs_settings']['mode'] == merged_res['mode']
        assert loaded_sampling_result['fs_settings']['num'] == merged_res['num']
        assert loaded_sampling_result['fs_settings']['split'] == str(set_id)

        failure_seqs.extend(loaded_sampling_result['failure_seqs'])
        sampling_results.update(loaded_sampling_result['sparse_sample_result'])

    res = {'failure_seqs': failure_seqs,
           'sparse_sample_result': sampling_results,
           'fs_settings': merged_res}

    json.dump(res, open(merged_sampling_result_file, 'w'), indent=4, sort_keys=True)


if __name__ == '__main__':
    args = parse_args()
    set_ids = decode_split(args.dataset, args.split)
    if not args.no_parse:
        for set_id in set_ids:
            dataset = get_dataset(args.dataset, str(set_id))
            sample_frames(dataset, args, set_id)
    if args.merge and len(set_ids) >= 2:
        merge_split_results(args, set_ids)
