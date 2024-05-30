import ltr.data.processing_utils as prutils
from ltr.utils import TensorDict
import torch, bisect, json, os, random, shutil
# from ltr.utils.vis_ops import vis_patches


class SparseMiddleware:
    """
     The Middleware class for training trackers in a sparsely-supervised manner
     SparseMiddleware class is a middleware connecting the teacher inference stage and student training stage.

     Typically, it bridges the teacher and the student with some necessary operations, including:

         1. Fetching the tracking results from teacher, and transform the results to coordinate in original images
         2. Filter the tracking results by clues, e.g. the IoU prediction scores
         3. Calculate pseudo annos for unsupervised pairs based on the tracking results of the teacher
         4. Record some necessary information (e.g. coarse boxes in unlabeled frames), ready for sampler state update
         5. Update the sampler state at the end of each training epoch in sparsely-supervised training stage
     """

    def __init__(self, settings, *args, **kwargs):
        """
        args:
            settings - settings for tracker training, at least including:

            - confidence_threshold - A threshold for filtering tracking failures by IoU prediction head
            - device - Device of tensor operations, typically "cuda"
            - print_aux - Auxiliary Boolean tag, indicating whether to print auxiliary information in middleware
            - state_saving_interval - The epoch interval for saving sampler state file as a middle state (stop and resume)
            - magic_number - The magic number for indicating the sampler state file name
        """
        super().__init__(*args, **kwargs)
        self.settings = settings

        # Important hyperparameters for filtering, cropping and augmenting the unsupervised pairs
        self.search_sz = settings.search_sz
        self.confidence_threshold = settings.curriculum_settings['confidence_threshold']

        self.device = settings.device
        self.print_aux = settings.printing_auxiliary_info
        self.state_saving_interval = settings.state_saving_interval

        # Here epoch_state_info is initialized to record all intermediate tracking results by teacher in this epoch
        self.epoch_state_info = dict()
        # Sampler state file records the existing coarse boxes of targets recorded in previous epochs
        self.sampler_state_file = os.path.join(
            settings.env.var_dir, 'records', "sampler_state_{}.json".format(settings.magic_number))

    def record_relay_results(self, middle_info, pseudo_label):
        """
        Entrance of SparseMiddleWare, constructing materials (unsupervised pairs) for training the student

        args:
            middle_info - A dict containing necessary information loaded from dataloader, containing at least:
                'key_record_keys', 'unlabeled_record_keys': Useful info for those frames, e.g. sequence/frame id
                'unlabeled_box_extract', 'unlabeled_resize_factors': Info for recovering the box in unlabeled frames.
                'aug_box_extract', 'aug_resize_factors': Info for the augmented patch prepared for training

            pseudo_label - A TensorDict of intermediate tracking results, loaded from SparseActors, containing:
                   "best_bbox": pseudo labels taken by teacher inference in search image coordinate
                   "best_score": score of the selected box
                   "best_conf": the IoU prediction score for filtering tracking failure

            data_unsup - The TensorDict of original data loaded by dataloader.
                   Please refer to processing.py for its detailed structure.
                   Typically, if a tracking failure happens, we use labeled templates loaded to fill the blank.
        """

        # Transmit related tensor to cuda
        for prefix in ['aug', 'unlabeled']:
            for suffix in ['box_extract', 'resize_factors']:
                key = "{}_{}".format(prefix, suffix)
                middle_info[key] = middle_info[key].to(self.device)

        # Select and filter training instances by the predicted IoU scores
        decoded_teacher_tracking = [self._record_pseudo_label(_box_extract, _resize_factor, _key_record_keys,
                                                      _unlabeled_record_keys, _bbox_predict, _conf_predict)
                            for (_box_extract, _resize_factor, _key_record_keys,
                                 _unlabeled_record_keys, _bbox_predict, _conf_predict)
                            in zip(middle_info['unlabeled_box_extract'].squeeze(1),
                                   middle_info['unlabeled_resize_factors'],
                                   middle_info['key_record_keys'], middle_info['unlabeled_record_keys'],
                                   pseudo_label['best_bbox'], pseudo_label['best_conf'])]

        bboxes_recovered, success_tags = zip(*decoded_teacher_tracking)
        # Generate new annotation for augmented patches (unsupervised pair)
        new_aug_anno = [self._transform_aug_label(_bbox_rec, _box_extract, _resize_factor)
                        for (_bbox_rec, _box_extract, _resize_factor)
                        in zip(bboxes_recovered, middle_info['aug_box_extract'].squeeze(1),
                               middle_info['aug_resize_factors'])]

        # Middleware Results
        pseudo_label = {"aug_anno": torch.stack(new_aug_anno, dim=0),
                        'aug_success': torch.stack(success_tags, dim=0)}
        return pseudo_label

    def _record_pseudo_label(self, box_extract, resize_factor, key_record_key,
                             unlabeled_record_key, bbox_predict, conf_predict):
        """
        Filter training instances, and collect necessary components for cropping patches later
        """

        # Recover predicted bounding boxes in search areas to boxes in original images
        bbox_recovered = prutils.recover_bbox_to_original(bbox_predict, box_extract, resize_factor, self.search_sz)

        # Doing statistics for successful forward tracking attempts
        successful_pseudo_label = 0

        # KEY RECORDS: [sequence_key, key_frame_id]
        seq_key, key_path_id, key_fid = key_record_key[0]
        # UNLABELED RECORDS: [sequence_key, unlabeled_frame_id]
        seq_key, unlabeled_path_id, unlabeled_frame_id = unlabeled_record_key[0]
        # Bounding box prediction from teacher network
        pred_bbox = bbox_recovered.tolist()

        # Use and record unlabeled frames and pseudo labels if IoU prediction confidence is high enough.
        successful_tracking = conf_predict >= self.confidence_threshold

        # The following operations depend on how many tracking trials above are successful
        if successful_tracking:

            # Record intermediate results as the coarse location of moving object
            # Doing statistics
            successful_pseudo_label += 1
            if self.print_aux:
                print("SUCCESS: Video: {}, Key F_id: {}, Unlabeled F_id: {}, IoU prediction: {:04f}, ".format(
                    seq_key, key_fid, unlabeled_frame_id, float(conf_predict)))

            # If the sequence has not been encountered in this single epoch, create new dict
            if seq_key not in self.epoch_state_info:
                self.epoch_state_info[seq_key] = dict()
                create_keys = ['path_ids', 'relay_frame_ids', 'bboxes', 'confidence']
                for k in create_keys:
                    self.epoch_state_info[seq_key][k] = []

            # Insert the iteration information into the epoch recording dict
            k_vs_insert = {'path_ids': unlabeled_path_id,'relay_frame_ids': unlabeled_frame_id,
                           'bboxes': pred_bbox, 'confidence': round(float(conf_predict), 4)}
            for (k, v) in k_vs_insert.items():
                self.epoch_state_info[seq_key][k].append(v)

        else:
            if self.print_aux:
                print("FAILURE: Video: {}, Key F_id: {}, Unlabeled F_id: {}, IoU prediction: {:04f}, ".format(
                    seq_key, key_fid, unlabeled_frame_id, float(conf_predict)))

        return bbox_recovered, successful_tracking

    def _transform_aug_label(self, bbox_rec, box_extract, resize_factor):

        transformed_label = prutils.transform_image_to_crop(bbox_rec, box_extract, resize_factor, self.search_sz)
        return torch.clamp_min(transformed_label, min=0.0)

    def update_sampler_state(self, epoch=None):
        """
        Update the sampler state based on the information collected in an epoch (self.epoch_state_info)
        This process is often performed at the end of each training epoch in the sparsely-supervised training stage

        args:
            epoch - Current training epoch
        """

        # Load current recorded sampler state
        prev_sampler_state = json.load(open(self.sampler_state_file, 'r'))

        # Merge existing locations and info collected in the current epoch
        for seq_key in self.epoch_state_info.keys():
            path_ids = self.epoch_state_info[seq_key]['path_ids']
            relay_frame_ids = self.epoch_state_info[seq_key]['relay_frame_ids']
            recorded_bboxes = self.epoch_state_info[seq_key]['bboxes']
            recorded_confidence = self.epoch_state_info[seq_key]['confidence']

            for (p_id, f_id, bbox, conf) in zip(path_ids, relay_frame_ids, recorded_bboxes, recorded_confidence):

                # bbox precision to 0.1, or saving location file storage
                bbox_round = [round(digit, 1) for digit in bbox]

                # Boolean tag of whether to conduct insertion
                insert_tag = False
                # Boolean tag of whether to replace existing coarse location
                replace_tag = False

                # Find the index to insert new relay frame
                insert_index = bisect.bisect(prev_sampler_state[seq_key][p_id]['relay_frame_ids'], f_id)

                if insert_index == 0:
                    # The new anchor frames is beyond the effect range of currently recorded anchor frames
                    insert_tag = True
                else:
                    left_fid = prev_sampler_state[seq_key][p_id]['relay_frame_ids'][insert_index - 1]
                    if left_fid == f_id:
                        replace_tag = True
                    else:
                        insert_tag = True

                if replace_tag and insert_index not in prev_sampler_state[seq_key][p_id]['key_frame_ids'] \
                        and conf > prev_sampler_state[seq_key][p_id]['confidence'][insert_index - 1]:
                    # Replace the original coarse location with new ones (don't replace key frame info)
                    k_vs_replace = {'bboxes': bbox_round,
                                    'confidence': conf}
                    for (k, v) in k_vs_replace.items():
                        prev_sampler_state[seq_key][p_id][k][insert_index - 1] = v

                elif insert_tag:
                    # Insert current f_id iff the new anchor frames is beyond the effect range of recorded anchor frames
                    k_vs_insert = {'relay_frame_ids': f_id,
                                   'bboxes': bbox_round,
                                   'confidence': conf}
                    for (k, v) in k_vs_insert.items():
                        prev_sampler_state[seq_key][p_id][k].insert(insert_index, v)

        # Empty the current epoch info
        self.epoch_state_info.clear()

        # Replace the current sampler state file
        json.dump(prev_sampler_state, open(self.sampler_state_file, 'w'), indent=4, sort_keys=True)

        # Cache sampler state for certain epochs, for stop training and resume training
        if epoch is not None and self.state_saving_interval is not None and epoch % self.state_saving_interval == 0:
            sampler_state_file_epoch = os.path.join(
                self.settings.env.var_dir, 'records',
                "sampler_state_{}_{:04d}.json".format(self.settings.magic_number, epoch))
            shutil.copy(self.sampler_state_file, sampler_state_file_epoch)
