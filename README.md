# SPOT: Sparsely-Supervised Object Tracking

:herb: **[Sparsely-Supervised Object Tracking]()**

Jilai Zheng, Wenxi Li, Chao Ma and Xiaokang Yang

IEEE Transactions on Image Processing (TIP 2024)

## Introduction

This repository implements sparsely-supervised deep tracker SPOT, 
which learns to track objects from very limited annotations (typically less than 5) in each training video. 
 
Main contributions of our SPOT framework can be summarized as follows:

- We propose a new sparse labeling setting for training visual trackers, as a practical tradeoff between conventional fully-supervised and unsupervised tracking.
- We incorporate the novel transitive consistency scheme into the teacher-student learning framework to train sparsely-supervised deep trackers. 
- We utilize three effective schemes, including IoU filtering, asymmetric augmentation, and temporal calibration, for robust sparsely-supervised learning. 
- We validate the proposed framework on large-scale video datasets. With two orders of magnitude fewer labels, our sparsely-supervised trackers perform on par with their fully-supervised counterparts.
- We study how to properly exploit a vast number of videos under a limited labeling budget, and demonstrate the potential of exploiting purely unlabeled videos for additional performance gain.

## Tutorial

Please refer to [TUTORIAL.md](docs/TUTORIAL.md) for how to use this repo.

## Performance

Please refer to [PERFORMANCE.md](docs/PERFORMANCE.md) for performance on SOT benchmarks. 

## Citation

If any parts of our paper and codes are helpful to your work, please generously citing:
 
```BibTeX
@inproceedings{zheng-tip2024-spot,
    title={Sparsely-Supervised Object Tracking},
    author={Zheng, Jilai and Li, Wenxi and Ma, Chao and Yang, Xiaokang},
    booktitle={IEEE Transactions on Image Processing (TIP)},
    year={2024}
}
```

## Reference
 
  We refer to the following repositories when implementing the SPOT framework. 
  Thanks for their great work.
 
 - [visionml/pytracking](https://github.com/visionml/pytracking)
 - [chenxin-dlut/TransT](https://github.com/chenxin-dlut/TransT)
 - [researchmm/TracKit](https://github.com/researchmm/TracKit)
 - [facebookresearch/unbiased-teacher](https://github.com/facebookresearch/unbiased-teacher)

## Contact
 
  Feel free to contact me if you have any questions.
 
 - Jilai Zheng, email: [zhengjilai@sjtu.edu.cn](https://github.com/zhengjilai)

