import random
import torch.utils.data
from ltr.utils import TensorDict
import numpy as np
import bisect
import copy
import os
import json
from part import FrozenIntervalSet
from ltr.data.image_loader import jpeg4py_loader


def no_processing(data):
    return data


def seq_key_name(dataset, seq_name):
    if dataset.get_name() == "TrackingNet":
        seq_name_formatted = "{}_{:02d}_{}".format(dataset.get_name(), seq_name[0], seq_name[1])
    elif dataset.get_name() == "LaSOT":
        seq_name_formatted = "{}_{}".format(dataset.get_name(), seq_name)
    else:
        seq_name_formatted = seq_name
    return seq_name_formatted


class SparseTrackingSampler(torch.utils.data.Dataset):
    """
        Class responsible for sampling sparse frames from training sequences to form batches.
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch,
                 num_labeled_frames=2, num_unlabeled_frames=2,
                 frame_sample_settings=None, curriculum_settings=None,
                 processing=no_processing, burn_in=False, name='train',
                 sampler_state_file=None):
        """
        Tracking Sampler for SPOT
        This class is responsible for: 1. (offline) sparsely sampling frames from videos as labeled ones
                                       2. (online) constructing supervised/unsupervised training instances

        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch

            max_gap - Maximum gap, often use None for SPOT compatibility (except fully-supervised baseline)
            num_labeled_frames - Number of labeled frames to sample.
            num_unlabeled_frames - Number of unlabeled frames to sample.

            frame_sample_settings - Label sampling mode, only support 'key_frames' in released version
            curriculum_settings - Settings related to the training curriculum (reserved for future features)

            processing - An instance SparseProcessingImpl class which performs the necessary processing of the data

            burn_in - Boolean tag indicating whether the sampler works as burn_in stage or sparsely-supervised stage
            name - Loader name

            sampler_state_file - The file for storing current sampling state

        """
        # Name and datasets
        self.name = name
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        # Parameters
        self.samples_per_epoch = samples_per_epoch
        self.max_gap = curriculum_settings['max_gap']
        self.num_labeled_frames = num_labeled_frames
        self.num_unlabeled_frames = num_unlabeled_frames

        # Whether it is a burn-in stage sampler (sampling only labeled frames)
        self.burn_in = burn_in

        # Init the sparsely sampled frames for all datasets
        self.sparse_sample_result = dict()
        self.failure_seqs = []
        self.sampler_state = dict()
        self.sampler_state_file = sampler_state_file

        # Frame sampling settings
        self.fs_settings = frame_sample_settings
        # Hyper-parameters for 'key_frames' sampling mode
        self.curriculum_settings = curriculum_settings

        # sparsely-supervised preprocessor
        self.processing = processing

        # IMPORTANT: loading the sampling results from folder "sampling/sampling_results"
        self._load_sampled_results()

    def __len__(self):
        return self.samples_per_epoch

    def prepare_sampler_state(self):
        """
        Prepare the sampler state file (initializing a new one, or loading an existing one)
        """
        if self.sampler_state_file is not None:

            if os.path.exists(self.sampler_state_file):
                self._load_sampler_state_file(self.sampler_state_file)
            else:
                self._init_sampler_state_file(self.sampler_state_file)

    def _load_sampled_results(self):
        """
        Loading an existing sampling result from stored files
        We only support sampling mode 'key_frames' (k=2,3,5) in the released SPOT version
        For other sampling modes used in ablation study, e.g. det, mot, manual_all,
                    please contact me if you are quite interested and really want to have access to those codes.
        Load the sampled result if the sampling result files already exists,
               else the framework will automatically re-sample labels and generate a new sampling file
        """
        sampling_file_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          "..", "..", "sampling", "sampling_results")
        total_bounding_boxes = 0

        for dataset in self.datasets:

            dataset_name = dataset.get_name()
            dataset_fs_setting = self.fs_settings[dataset_name]
            name = dataset_fs_setting['name']

            # All frames means that all frames should be sampled as key frames (typically for full-label training)
            if name in ["all_frames"]:
                # Count all annotated frames for all video sequences (typically for GOT-10k and LaSOT)
                self.fs_settings[dataset_name]['mode'] = name
                dataset_bounding_boxes = 0
                for seq_id in range(0, dataset.get_num_sequences()):
                    # Get sequence name
                    seq_info_dict = dataset.get_sequence_info(seq_id)
                    dataset_bounding_boxes += len(seq_info_dict['visible'])

            else:
                sample_file = os.path.join(sampling_file_root, "{}.json".format(name))
                if os.path.exists(sample_file):
                    print("Loading sampling result of dataset {} split {}, file name {}.".format(
                        dataset_name, dataset_fs_setting['split'], name))
                else:
                    raise NotImplementedError("File {}.json does not exist in ./sampling/sampling_results. \n"
                                              "Please resample or check the mistake.".format(name))
                loaded_sampled_result = json.load(open(sample_file, 'r'))

                self.fs_settings[dataset_name].update(loaded_sampled_result['fs_settings'])
                self.sparse_sample_result[dataset_name] = loaded_sampled_result['sparse_sample_result']
                self.failure_seqs.extend(loaded_sampled_result['failure_seqs'])

                # do statistics for bounding box number
                dataset_bounding_boxes = 0

                # Count sparse annotated frames for all video sequences
                for seq_id in range(0, dataset.get_num_sequences()):

                    # Get sequence name
                    seq_name = dataset.sequence_list[seq_id]
                    seq_name_formatted = seq_key_name(dataset, seq_name)

                    if seq_name_formatted in self.failure_seqs:
                        continue

                    dataset_bounding_boxes += \
                        sum([len(p[0]['f_ids']) if isinstance(p, list) else len(p['f_ids'])
                             for p in self.sparse_sample_result[dataset_name][seq_name_formatted]])

            total_bounding_boxes += dataset_bounding_boxes
            print("Sampled key frame number for dataset {}, split {} is {}.".format(
                dataset_name, dataset_fs_setting['split'], dataset_bounding_boxes))

        print("Total key frame number sampled is {}.".format(total_bounding_boxes))
        print("Current frame sampling setting: \n {}".format(self.fs_settings))

    def _load_sampler_state_file(self, sampler_state_file):
        """
        Loading sampler state from existing sampler state file
        """

        print("Loading sampler state file as {}".format(sampler_state_file))

        # Load recorded locations from working dir
        self.sampler_state = json.load(open(sampler_state_file, 'r'))

        # Do statistics for number of relay frames
        key_frame_number = 0
        for key in self.sampler_state.keys():
            recorded_key_frames = self.sampler_state[key]
            key_frame_number += sum([len(path_state['relay_frame_ids']) for path_state in recorded_key_frames])

        print("Total number of recorded relay frames is {}.".format(key_frame_number))

    def _init_sampler_state_file(self, sampler_state_file):
        """
        Initializing sampler state based on the sampled labeled frames
        """

        print("Initializing sampler state file as {}".format(sampler_state_file))

        key_frame_number = 0

        for dataset_name in self.sparse_sample_result.keys():

            dataset = None
            for d in self.datasets:
                if d.get_name() == dataset_name:
                    dataset = d
            if self.fs_settings[dataset_name]['mode'] in ["all_frames"]:
                continue
            if dataset is None:
                raise Exception("Dataset {} is not found when initializing sampler state.".format(dataset_name))

            # Number of sequence with preprocessing failure
            num_preprocessing_failure = 0
            # Iter for all sequences in dataset
            for seq_id in range(0, dataset.get_num_sequences()):

                # Use key frames with ground truth labels as init coarse boxes for moving objects relay
                seq_name = dataset.sequence_list[seq_id]
                seq_name_formatted = seq_key_name(dataset, seq_name)

                if seq_name_formatted not in self.sparse_sample_result[dataset_name]:
                    assert seq_name_formatted in self.failure_seqs, "Sequence {} should be in failure seqs list!"
                    num_preprocessing_failure += 1
                    continue

                all_paths = self.sparse_sample_result[dataset_name][seq_name_formatted]
                self.sampler_state[seq_name_formatted] = []

                for path in all_paths:
                    key_frame_ids = path['f_ids']
                    # Default box format: [x, y, w, h], both for preprocessing and gt annos
                    if 'bboxes' in path:
                        # For automatic target discovery: mot/det
                        bboxes = [torch.tensor(bbox) for bbox in path['bboxes']]
                    else:
                        # For conventional labeled frames
                        bboxes = dataset.get_annos(seq_id, key_frame_ids)['bbox']
                    # Do statistics for key frame number
                    key_frame_number += len(key_frame_ids)
                    path_init_state = {
                        # Frame id list for key frames (a fixed list)
                        'key_frame_ids': key_frame_ids,
                        # Frame id list for relay frames (object locations predicted or annotated)
                        'relay_frame_ids': key_frame_ids,
                        # Bounding box list, inited by annotated bbox and accumulated by predicted relay bboxes
                        'bboxes': [bbox.tolist() for bbox in bboxes],
                        # Confidence score predicted by the attached IoU prediction head
                        'confidence': [float(1.0) for _ in key_frame_ids],
                    }
                    # Record init info in coarse location dict
                    self.sampler_state[seq_name_formatted].append(path_init_state)

            # Valid check for dataloader and preprocessed results
            assert dataset.get_num_sequences() == \
                   len(self.sparse_sample_result[dataset_name].keys()) + num_preprocessing_failure, \
                   "The number of dataset instances does not match with the preprocessed results!"

        print("Total number of init relay frames is {}.".format(key_frame_number))

        json.dump(self.sampler_state, open(sampler_state_file, 'w'), indent=4, sort_keys=True)

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
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

    def _sample_from_candidates(self, candidate, num_ids=1, max_id=None, min_id=None):
        """
        Samples a single frame between min_id and max_id from candidate frame id list only

        args:
            candidate - List for candidate frame ids available (WARNING: frame ids already sorted)
            id_num - Number of frame ids to be sampled
            min_id - Minimum allowed frame id
            max_id - Maximum allowed frame id

        returns:
            frame_ids - List of sampled frame ids.
        """

        # Extreme cases or with no frame_id limitation is set
        if min_id is None or min_id < candidate[0]:
            min_id = candidate[0]
        if max_id is None or max_id > candidate[-1]:
            max_id = candidate[-1]
        if max_id < min_id:
            min_id, max_id = max_id, min_id

        # Find sample index range with binary search
        min_candidate_index = bisect.bisect_left(candidate, min_id, lo=0, hi=len(candidate))
        max_candidate_index = bisect.bisect(candidate, max_id, lo=min_candidate_index, hi=len(candidate))

        # Sample the target frames
        frame_ids = random.choices(candidate[min_candidate_index:max_candidate_index], k=num_ids)

        return frame_ids

    def _find_nearest_frames(self, candidate, target_ids):
        """
        From all candidate frames, find the 1st nearest / 2nd nearest candidate frame of the target_ids

        args:
            candidate - List for candidate frame ids available (WARNING: frame ids already sorted)
            target_ids - List for target ids

        returns:
            nearest_frame_ids - List of candidate frames nearest to the target ids
            second_nearest_frame_ids - List of candidate frames 2nd nearest to the target ids
        """

        # List of 1st and 2nd nearest frames
        nearest_frame_ids = []
        second_nearest_frame_ids = []

        for target_id in target_ids:
            # Find the nearest two frames in candidate list
            if target_id > candidate[-1]:
                left_index, right_index = len(candidate) - 1, len(candidate) - 1
            elif target_id < candidate[0]:
                left_index, right_index = 0, 0
            else:
                if target_id == candidate[-1]:
                    # Just the last frame is selected (out of range if not being processed carefully)
                    right_index = len(candidate) - 1
                else:
                    # Other frame except the last frame is selected
                    right_index = bisect.bisect(candidate, target_id, lo=0, hi=len(candidate))
                left_index = right_index - 1

            # Find which index is the nearest (left or right)
            left_tag = abs(target_id - candidate[left_index]) < abs(target_id - candidate[right_index])
            if left_tag:
                nearest_frame_ids.append(candidate[left_index])
                second_nearest_frame_ids.append(candidate[right_index])
            else:
                nearest_frame_ids.append(candidate[right_index])
                second_nearest_frame_ids.append(candidate[left_index])

        return nearest_frame_ids, second_nearest_frame_ids

    def _linear_interpolation_bbox(self, target_frame_ids, frame_ids_1,
                                   bboxs_1, frame_ids_2, bboxs_2, interpolation_limit=4):
        """
        Use linear interpolation to find coarse boxes between relay frames.
        Note: If the frame interval between fid_1 and fid_2 is too long, simply use the nearest one as coarse box

        args:
            target_frame_ids - The target frames to calculate interpolated boxes
            frame_ids_1 - List of frame ids of box 1
            bboxs_1 - List of box coordinates of box 1
            frame_ids_2 - List of frame ids of box 2
            bboxs_2 - List of box coordinates of box 2
            interpolation_limit - The maximum frame interval for box linear interpolation

        returns:
            interpolated_bboxs - List of interpolated boxes for target frames
        """

        # Return list for interpolated bboxes
        interpolated_bboxs = []

        for target_fid, fid_1, bbox_1, fid_2, bbox_2 in \
                zip(target_frame_ids, frame_ids_1, bboxs_1, frame_ids_2, bboxs_2):

            if fid_2 == fid_1 or abs(fid_1 - fid_2) > interpolation_limit:
                # Do not need to interpolate if fid_1 == fid_2, or the frame interval is too long
                bbox_ip = copy.deepcopy(bbox_1)
            else:
                # Interpolation weight for bbox_1
                alpha_1 = float(abs(fid_2 - target_fid)) / abs(fid_1 - fid_2)
                # Linear interpolation
                bbox_ip = alpha_1 * bbox_1 + (1 - alpha_1) * bbox_2

            interpolated_bboxs.append(bbox_ip)

        return interpolated_bboxs

    def _sample_unlabeled_frames(self, dataset, seq_id, path_id, seq_info_dict):
        """
        Basic logic of sampling unlabeled frames based on current sampler state

        Two modes are alternatively used, including reviewing mode and propagating mode.
        Please refer to our paper for more details.

        args:
            dataset - The sampled dataset
            seq_id - The sampled sequence id in certain dataset
            path_id - The sampled object trajectory (not used in burnin training)
            seq_indo_dict - Sequence info

        returns:
            unlabeled_frame_ids - Frame ids of the sampled unlabeled (arbitrary) frames
            unlabeled_record_keys - Necessary information of those unlabeled frames (used in Middleware)
            unlabeled_coarse_boxes - Coarse boxes of the targets (not gt, but boxes estimated from relay boxes)
        """

        # Fetch recorded frames and necessary sequence info
        seq_name = dataset.sequence_list[seq_id]
        seq_name_formatted = seq_key_name(dataset, seq_name)
        existing_relay_frame_ids = self.sampler_state[seq_name_formatted][path_id]['relay_frame_ids']
        bboxes = self.sampler_state[seq_name_formatted][path_id]['bboxes']
        seq_length = len(seq_info_dict['visible'])

        # We alternatively switch between propagating mode and reviewing mode
        already_explored_area = len(existing_relay_frame_ids)
        propagating_ratio = 1 - (already_explored_area / seq_length)
        # Here reviewing_mode is a Boolean tag indicating whether to use propagation mode
        trigger_reviewing_mode = True if random.random() > propagating_ratio else False

        # In propagating mode, we only sample those frames,
        #            where the spatial location of the object has not been coarsely located.
        prop_step = max(self.curriculum_settings['propagating_step'], 1)
        explored_step = max(0, min(prop_step // 2, prop_step - 1))
        cur_prop_step = random.randrange(explored_step + 1, prop_step + 1, 1)
        if not trigger_reviewing_mode:
            explored_ranges = [(existing_relay_frame_ids[j] - explored_step,
                                existing_relay_frame_ids[j] + explored_step, True, True)
                               for j in range(len(existing_relay_frame_ids))]
            propagate_ranges = [(existing_relay_frame_ids[j] - cur_prop_step,
                                 existing_relay_frame_ids[j] + cur_prop_step, True, True)
                                for j in range(len(existing_relay_frame_ids))]
            valid_ranges = [(0, seq_length - 1, True, True)]
            located_interval = FrozenIntervalSet(explored_ranges)
            propagate_interval = FrozenIntervalSet(propagate_ranges)
            valid_interval = FrozenIntervalSet(valid_ranges)

            # Calculate the sentinel sampling range
            propagate_interval_valid = propagate_interval.intersection(valid_interval)
            sentinel_sampling_interval = propagate_interval_valid.difference(located_interval)

            # Check whether sentinel sampling interval is Empty
            if len(sentinel_sampling_interval) == 0:
                # Exit sampling sentinel frames, conduct normal sampling instead
                trigger_reviewing_mode = True
            else:
                # Calculate and normalize sampling probabilities for all intervals
                sampling_l_u_bound_close = [(itv.lower_value if itv.lower_closed else itv.lower_value + 1,
                                             itv.upper_value if itv.upper_closed else itv.upper_value - 1)
                                            for itv in sentinel_sampling_interval]
                sampling_probability = [bound_close[1] - bound_close[0] + 1
                                        for bound_close in sampling_l_u_bound_close]
                prob_total = sum(sampling_probability)

                # In case all open intervals for the sampling ranges are empty
                if prob_total <= 0:
                    trigger_reviewing_mode = True

        unlabeled_frame_ids = []
        if trigger_reviewing_mode:
            # Sampling in reviewing mode (frames already explored)
            max_id, min_id = None, None
            sampled_frame_ids = self._sample_from_candidates(existing_relay_frame_ids, num_ids=2,
                                                             min_id=min_id, max_id=max_id)
            unlabeled_frame_ids.extend(sampled_frame_ids)
        else:
            # Sampling unlabeled frame in propagating mode (frames never explored)
            prob_normed = [x / prob_total for x in sampling_probability]
            bound_close_sampled = random.choices(sampling_l_u_bound_close, prob_normed)[0]
            sampled_frame_id = random.choices(range(bound_close_sampled[0],
                                                    bound_close_sampled[1] + 1))[0]
            unlabeled_frame_ids.append(sampled_frame_id)

            # The relay frame is always collected from already explored frames
            max_id, min_id = None, None
            sampled_frame_ids = self._sample_from_candidates(existing_relay_frame_ids, num_ids=1,
                                                             min_id=min_id, max_id=max_id)
            unlabeled_frame_ids.extend(sampled_frame_ids)

        # Find 1st / 2nd nearest relay frame ids as reference from all relay frames with coarse locations
        relay_frame_ids, second_nearest_relay_frame_ids = \
            self._find_nearest_frames(existing_relay_frame_ids, unlabeled_frame_ids)

        # Get recorded coarse boxes for 1st / 2nd nearest relay frames (using coarse boxes, not gt anno)
        relay_c_box = []
        for f_id in relay_frame_ids:
            idx = existing_relay_frame_ids.index(f_id)
            relay_c_box.append(torch.tensor(bboxes[idx]))
        second_near_relay_c_box = []
        for f_id in second_nearest_relay_frame_ids:
            idx = existing_relay_frame_ids.index(f_id)
            second_near_relay_c_box.append(torch.tensor(bboxes[idx]))

        # Bounding box linear interpolation, getting a coarse location for unlabeled target (for patch cropping)
        unlabeled_coarse_boxes = self._linear_interpolation_bbox(unlabeled_frame_ids,
                                                                 relay_frame_ids, relay_c_box,
                                                                 second_nearest_relay_frame_ids,
                                                                 second_near_relay_c_box)

        # Record key is used to record some necessary info from dataloader and send them to middleware
        # UNLABELED_RECORD_KEY (used in middleware): [sequence_key, path_id, unlabeled_frame_id]
        unlabeled_record_keys = [(seq_name_formatted, path_id, unlabeled_frame_ids[j])
                                 for j in range(len(unlabeled_frame_ids))]

        return unlabeled_frame_ids, unlabeled_record_keys, unlabeled_coarse_boxes

    def _necessary_info_filling(self, key_frames, key_anno, cand_f_ids=None, cand_boxes=None, sampled_f_ids=None):
        """
        The average channel calculator for sampled frames

        args:
            key_frames: raw frames in videos
            key_anno: the annotation structure for sampled frames, calculate if 'avg_channels' is not inside

        returns:
            key_anno: processed anno structure, with avg_channels calculated

        """
        # Replace gt box anno with the processed boxes
        if cand_boxes is not None:
            processed_boxes = []
            for sampled_f_id in sampled_f_ids:
                idx = cand_f_ids.index(sampled_f_id)
                # Processed boxes also in format [x, y, w, h]
                processed_boxes.append(torch.tensor(cand_boxes[idx]))
            key_anno['bbox'] = processed_boxes

        # Filling the avg_channel info from frame
        if 'avg_channels' in key_anno:
            return key_anno
        else:
            avg_channels = []
            for key_frame in key_frames:
                avg_channel = np.mean(key_frame, axis=(0, 1))
                avg_channels.append(torch.tensor(avg_channel))
            key_anno['avg_channels'] = avg_channels
            return key_anno

    def _meta_recorder_burn_in(self, dataset, seq_id, path_id, template_frame_ids):
        """
        Build record_keys for supervised instances
        """

        seq_key = "{}_{:06d}".format(dataset.get_name(), seq_id)
        template_record_keys = [(seq_key, path_id, f_id) for f_id in template_frame_ids]
        return template_record_keys

    def _meta_recorder_sparse(self, unlabeled_record_keys, key_frame_ids):
        """
        Build record_keys for unsupervised instances
        """

        # Generate necessary auxiliary information for sampled key frames
        seq_key = unlabeled_record_keys[0][0]
        path_id = unlabeled_record_keys[0][1]
        key_record_keys = [(seq_key, path_id, f_id) for f_id in key_frame_ids]
        return unlabeled_record_keys, key_record_keys

    def _construct_burn_in_material(self, template_frames, template_anno,
                                    search_frames, search_anno):
        """
        Construct a formative data TensorDict for burn_in stage
        """

        # Return TensorDict for key and unlabeled frames
        # Note that the unlabeled pseudo anno is only an interpolated object location used for coarsely cropping
        data = TensorDict({
            'template_images': template_frames,
            'template_anno': template_anno['bbox'],
            'template_avg': template_anno['avg_channels'],
            'search_images': search_frames,
            'search_anno': search_anno['bbox'],
            'search_avg': search_anno['avg_channels'],
        })
        return data

    def _construct_sparse_material(self, key_record_keys, key_frames, key_anno, unlabeled_frames,
                                   unlabeled_coarse_boxes, unlabeled_anno, unlabeled_record_keys):
        """
        Construct a formative data TensorDict for sparse-supervised stage
        """

        # Return TensorDict for key and unlabeled frames
        # Here 'unlabeled_pseudo_anno' is only an interpolated object location used for coarsely cropping
        # The actual manual annotations for unlabeled frames are never used here !!!!
        data = TensorDict({
            # Key frames (nearest labeled frames to unlabeled ones)
            'key_images': key_frames,
            'key_anno': key_anno['bbox'],
            'key_avg': key_anno['avg_channels'],
            'key_record_keys': key_record_keys,

            # Unlabeled frames (as search patches)
            'unlabeled_images': unlabeled_frames[:1],
            'unlabeled_anno': unlabeled_coarse_boxes[:1],
            'unlabeled_avg': unlabeled_anno['avg_channels'][:1],
            'unlabeled_record_keys': unlabeled_record_keys[:1],

            # Relay frames (as relay templates for transitive consistency)
            'relay_images': unlabeled_frames[1:2],
            'relay_anno': unlabeled_coarse_boxes[1:2],
            'relay_avg': unlabeled_anno['avg_channels'][1:2],
            'relay_record_keys': unlabeled_record_keys[1:2]
        })
        return data

    def _sample_burn_in_instances(self, dataset, seq_id, path_id, cand_f_ids, cand_boxes, seq_info_dict):
        """
        The main logic for sampling supervised instances in burn-in/sparsely-supervised training stage

        args:
            dataset - The sampled dataset
            seq_id - The sampled sequence id in certain dataset
            path_id - The sampled object trajectory (not used in burnin training)
            cand_f_ids - The List of labeled frame ids
            cand_boxes: - The List of labeled frame boxes
            seq_indo_dict - Sequence info

        returns:
            TensorDict - dict containing all the data blocks and other info (avg/anno)
        """

        # Sample a base frame
        base_frame_id = self._sample_from_candidates(cand_f_ids)[0]
 
        # Sample template and search frames from sparsely labeled candidates
        # Note: max_gap used in typical template-based trackers is not fully compatible with our framework.
        # In our experiment, we use max_gap when running all baselines, but turn off max_gap for SP training.
        min_id = base_frame_id - self.max_gap if self.max_gap is not None else None
        max_id = base_frame_id + self.max_gap if self.max_gap is not None else None
        template_frame_id = self._sample_from_candidates(cand_f_ids, num_ids=1,
                                                         min_id=min_id, max_id=max_id)
        search_frame_id = self._sample_from_candidates(cand_f_ids, num_ids=1,
                                                       min_id=min_id, max_id=max_id)

        # Get template frames as well as their annotations from dataset
        template_frames, template_anno, _ = dataset.get_frames(seq_id, template_frame_id, seq_info_dict)
        # Here avg_channel will be filled, and key box will also
        template_anno = self._necessary_info_filling(
            template_frames, template_anno, cand_f_ids, cand_boxes, template_frame_id)

        # Get search frames as well as their annotations from dataset
        search_frames, search_anno, _ = dataset.get_frames(seq_id, search_frame_id, seq_info_dict)
        search_anno = self._necessary_info_filling(
            search_frames, search_anno, cand_f_ids, cand_boxes, search_frame_id)

        # Assemble all data before processing them
        data = self._construct_burn_in_material(template_frames, template_anno, search_frames, search_anno)

        # Use conventional TransT processing for burn-in stage
        return self.processing(data)

    def _sample_sparse_instances(self, dataset, seq_id, path_id, cand_f_ids, cand_boxes, seq_info_dict):
        """
        The main logic for sampling unsupervised instances in sparsely-supervised training stage

        args:
            dataset - The sampled dataset
            seq_id - The sampled sequence id in certain dataset
            path_id - The sampled object trajectory (reserved for det sampling and mot sampling)
            cand_f_ids - The List of labeled frame ids
            cand_boxes: - The List of labeled frame boxes
            seq_indo_dict - Sequence info

        returns:
            TensorDict - dict containing all the data blocks and other info (avg/anno)
        """

        # Sample two unlabeled frames from all available frames, together with recorded object location
        unlabeled_frame_ids, unlabeled_record_keys, unlabeled_coarse_boxes = \
            self._sample_unlabeled_frames(dataset, seq_id, path_id, seq_info_dict)

        # Get unlabeled frames (without 'bbox' or 'visible' annotation provided) from dataset
        # Here unlabeled_anno is only used for storing the avg_channel of loaded frames, bbox annos are never used
        unlabeled_frames, unlabeled_anno, _ = dataset.get_frames(seq_id, unlabeled_frame_ids, seq_info_dict)
        unlabeled_anno = self._necessary_info_filling(unlabeled_frames, unlabeled_anno)

        # Find 1st nearest labeled frame ids as reference from all labeled key frames
        key_frame_ids, _ = self._find_nearest_frames(cand_f_ids, unlabeled_frame_ids[:1])

        # Construct meta-data for both key frames and unlabeled frames
        unlabeled_record_keys, key_record_keys = \
            self._meta_recorder_sparse(unlabeled_record_keys, key_frame_ids)

        # Get raw images as well as annotations for nearest labeled frames (key frames)
        key_frames, key_anno, _ = dataset.get_frames(seq_id, key_frame_ids, seq_info_dict)
        key_anno = self._necessary_info_filling(key_frames, key_anno, cand_f_ids, cand_boxes, key_frame_ids)

        # Construct data for preprocessing
        data = self._construct_sparse_material(key_record_keys, key_frames, key_anno,
                                               unlabeled_frames, unlabeled_coarse_boxes,
                                               unlabeled_anno, unlabeled_record_keys)

        # Use sparsely-supervised TransT processing
        return self.processing(data)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks and other info (avg/anno)
        """

        # Randomly select a video dataset
        # Always use video datasets for sparse annotations (so MS-COCO is not compatible)
        while True:
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            mode = self.fs_settings[dataset.get_name()]['mode']
            use_stages = self.fs_settings[dataset.get_name()]['stage']
            # Some datasets are only used for certain stages (e.g. full-label videos only used for burnin)
            if (self.burn_in and "burnin" in use_stages) or (not self.burn_in and "sparse" in use_stages):
                break

        # Sample a sequence from all videos
        # Compared with the LTR sampler, any visible information is not available in our case
        while True:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)
            # Get sequence name
            seq_name = dataset.sequence_list[seq_id]
            seq_name_formatted = seq_key_name(dataset, seq_name)
            if seq_name_formatted not in self.failure_seqs:
                break
        seq_info_dict = dataset.get_sequence_info(seq_id)

        # Get the previously sampled sparsely annotated frame list
        cand_boxes = None
        if mode in ['key_frames', 'detection', 'manual_all', "mot", "manual_sparse"]:
            all_paths = self.sparse_sample_result[dataset.get_name()][seq_name_formatted]
            path_id = random.choice(range(len(all_paths)))
            cand_f_ids = all_paths[path_id]['f_ids']
            if 'bboxes' in all_paths[path_id]:
                cand_boxes = all_paths[path_id]['bboxes']
        elif mode in ['all_frames']:
            path_id = 0
            # For all_frames mode, only sample visible objects (typically for training a fully-sup baseline)
            cand_f_ids = (seq_info_dict['visible'] > 0.5).nonzero()[:, 0].tolist()
        else:
            raise NotImplementedError("Currently does not support sampling mode {}".format(mode))

        try:
            # Resample if there are no visible objects in the selected video
            if len(cand_f_ids) == 0:
                return self.__getitem__(index)
            # Burn-in stage, only use the original annotated frames
            if self.burn_in:
                return self._sample_burn_in_instances(dataset, seq_id, path_id, cand_f_ids, cand_boxes, seq_info_dict)
            # Sparsely-supervised stage with sparse annotations, use both originally annotated frames and unlabeled frames
            else:
                return self._sample_sparse_instances(dataset, seq_id, path_id, cand_f_ids, cand_boxes, seq_info_dict)

        except Exception as e:
            # This Resampling strategy exists because we find some annotation errors in datasets.
            # For example, "train/GOT-10k_Train_008628" has wrong labels, causing the loader to break down.
            loader_type = "burn_in" if self.burn_in else "sparse_sup"
            print("Something unexpected happened for {} loader, dataset {}, seq id {},"
                  " thus begin resampling!".format(loader_type, dataset.get_name(), seq_id))
            print("Error info: {}".format(e))
            return self.__getitem__(index)


class SparseTransTSampler(SparseTrackingSampler):
    """ See TrackingSparseSampler """

    def __init__(self, datasets, p_datasets, samples_per_epoch,
                 num_labeled_frames=1, num_unlabeled_frames=2,
                 frame_sample_settings=None, curriculum_settings=None,
                 processing=no_processing, burn_in=False, name='train',
                 sampler_state_file=None):
        super().__init__(datasets=datasets, p_datasets=p_datasets,
                         samples_per_epoch=samples_per_epoch,
                         num_labeled_frames=num_labeled_frames,
                         num_unlabeled_frames=num_unlabeled_frames,
                         frame_sample_settings=frame_sample_settings,
                         curriculum_settings=curriculum_settings,
                         processing=processing, burn_in=burn_in, name=name,
                         sampler_state_file=sampler_state_file)
