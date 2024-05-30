import torch
import os
import os.path
import numpy as np
import pandas
import json
import copy
from collections import OrderedDict
from ltr.data.image_loader import jpeg4py_loader, opencv_loader
from .base_video_dataset import BaseVideoDataset
from ltr.admin.environment import env_settings


def list_sequences(root, set_ids):
    """ Lists all the videos in the input set_ids. Returns a list of tuples (set_id, video_name)

    args:
        root: Root directory to TrackingNet
        set_ids: Sets (0-11) which are to be used

    returns:
        list - list of tuples (set_id, video_name) containing the set_id and video_name for each sequence
    """
    sequence_list = []

    for s in set_ids:
        anno_dir = os.path.join(root, "TRAIN_" + str(s), "anno")

        sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if f.endswith('.txt')]
        sequence_list += sequences_cur_set

    return sequence_list


class TrackingNet(BaseVideoDataset):
    """ TrackingNet dataset.

    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf

    Download the dataset using the got10k_toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None):
        """
        args:
            root        - The path to the TrackingNet folder, containing the training sets.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            set_ids (None) - List containing the ids of the TrackingNet sets to be used for training. If None, all the
                            sets (0 - 11) will be used.
        """
        root = env_settings().train_dataset_dirs[self.get_name()] if root is None else root
        super().__init__('TrackingNet', root, image_loader)

        set_ids = self._decode_split(split)
        if set_ids is None:
            set_ids = [i for i in range(12)]

        self.set_ids = set_ids

        # Keep a list of all videos. Sequence list is a list of tuples (set_id, video_name) containing the set_id and
        # video_name for each sequence
        # self.sequence_list = list_sequences(self.root, self.set_ids)
        self.sequence_list_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               "..", "data_specs", "trackingnet_all.txt")
        self.sequence_list = self._load_sequence_list(self.sequence_list_file, set_ids)

        self.seq_to_class_map, self.seq_per_class = self._load_class_info()

        # we do not have the class_lists for the tracking net
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def _load_class_info(self):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        class_map_path = os.path.join(ltr_path, 'data_specs', 'trackingnet_classmap.txt')

        with open(class_map_path, 'r') as f:
            seq_to_class_map = {seq_class.split('\t')[0]: seq_class.rstrip().split('\t')[1] for seq_class in f}

        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = seq_to_class_map.get(seq[1], 'Unknown')
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_to_class_map, seq_per_class

    def _load_sequence_list(self, file_path, set_ids):
        sequence_list_file = open(file_path, "r")
        lines = sequence_list_file.readlines()
        sequence_list = []
        for line in lines:
            if line == '' or line == '\n':
                continue
            else:
                line_split = line.split(',')
                split_id, video_name = int(line_split[0]), line_split[1].rstrip('\n')
                if split_id in set_ids:
                    sequence_list.append((split_id, video_name))
        return sequence_list

    def get_name(self):
        return 'TrackingNet'

    def has_class_info(self):
        return True

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_id):
        set_id = self.sequence_list[seq_id][0]
        vid_name = self.sequence_list[seq_id][1]
        bb_anno_file = os.path.join(self.root, "TRAIN_" + str(set_id), "anno", vid_name + ".txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32,
                             na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        seq_name = self.sequence_list[seq_id]
        seq_name_key = "{:02d}_{}".format(seq_name[0], seq_name[1])
        result = {'bbox': bbox, 'valid': valid, 'visible': visible}

        return result

    def _get_frame(self, seq_id, frame_id):
        set_id = self.sequence_list[seq_id][0]
        vid_name = self.sequence_list[seq_id][1]
        frame_path = os.path.join(self.root, "TRAIN_" + str(set_id), "frames", vid_name, str(frame_id) + ".jpg")
        return self.image_loader(frame_path)

    def _get_class(self, seq_id):
        seq_name = self.sequence_list[seq_id][1]
        return self.seq_to_class_map[seq_name]

    def get_class_name(self, seq_id):
        obj_class = self._get_class(seq_id)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):

        # A cache mechanism to prevent re-loading images from disk, in order to speed up dataloader
        # Note that in SPOT, this mechanism is helpful since labeled frames are very limited
        frame_list = []
        for i in range(len(frame_ids)):
            f = frame_ids[i]
            first_showup_index = frame_ids.index(f)
            if i == first_showup_index:
                frame_list.append(self._get_frame(seq_id, f))
            else:
                frame_list.append(copy.deepcopy(frame_list[first_showup_index]))

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        try:
            obj_class = self._get_class(seq_id)
        except:
            print("No class info for {}.".format(seq_id))
            obj_class = None

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return anno_frames

    def _decode_split(self, split_string):
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
