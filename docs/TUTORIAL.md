# SPOT Tutorial

Tutorial Page for Sparsely-Supervised Object Tracking (SPOT).


## Basic Features

The SPOT framework inherits [LTR](https://github.com/visionml/pytracking) for training, 
and integrates [pysot_toolkit](https://github.com/STVIR/pysot) for benchmark testing and evaluation. 

We also incorporate [Ocean](https://github.com/researchmm/TracKit) 
and [TransT](https://github.com/chenxin-dlut/TransT) into our SPOT framework. 
Both models can be used as the base tracker for sparsely-supervised training.


### Environment

Create new anaconda environment.
```
conda create -n SPOT python=3.8
conda activate SPOT
```

Install Pytorch and CUDA toolkit.
```
# CUDA 10.2
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch -c nvidia
# CUDA 11.1
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=11.1 -c pytorch -c nvidia
```

Install other necessary packages.
```
conda install matplotlib pandas tqdm
pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown
conda install cython scipy
sudo apt-get install libturbojpeg libgeos-dev
pip install pycocotools jpeg4py wget yacs py-part shapely==1.6.4.post2 mpi4py colorama
```


### Preparation

Dataset preparations before training, testing and evaluation (very important!). 

First, the local file (`config/local.py`) should be revised only once for each deployment trial. 
The following example takes the LaSOT dataset as an example.
```
# Directories for training datasets
self.train_dataset_dirs = {
    'LaSOT': '/data/dataset/lasot/LaSOTBenchmark',
}

# Directories for testing datasets
self.test_dataset_dirs = {
    'LaSOT': '/data/dataset/lasot/LaSOTTesting',
}
```

Then, download all annotation files (e.g. `LaSOT.json`) for testing benchmarks 
from [Google Drive (seq_files)](https://drive.google.com/drive/folders/1GBxAIIK4Tsgu7_17_iyGCrvc1gBcZAB-?usp=drive_link), 
and place them into the project folder `pysot_toolkit/seq_files/`.

Finally, conduct system preparations before running everything (also very important!). 
Without the `ulimit` command, the dataloader may break down during training.  
```
source activate SPOT
ulimit -SHn 51200
export PYTHONPATH=$(pwd):$PYTHONPATH
```


### Test

Use `scripts/test.py` for testing. Remember to specify the dataset (e.g. LaSOT),
model architecture (e.g. transt, ocean), and the wanted model checkpoint (e.g. spot_transt_002.pth).
Pretrained SP-TransT and SP-Ocean checkpoints can be found in 
[Google Drive (snapshot)](https://drive.google.com/drive/folders/1GBxAIIK4Tsgu7_17_iyGCrvc1gBcZAB-?usp=drive_link).
```
python scripts/test.py --dataset LaSOT --arch transt --resume your_model_name.pth.tar
```
Be careful that the code logic automatically add a base folder `var` before the model file path.  

### Eval

We integrate pysot_toolkit for evaluating models on benchmarks.
Install necessary package for evaluation (run only once).
```
cd ./pysot_toolkit/toolkit/utils/ && python setup.py build_ext --inplace
```

Evaluate performance on certain dataset (e.g. LaSOT), with result files generated in `var/results`.
```
python scripts/eval.py --tracker_path ./var/results/ --dataset LaSOT -t your_model_name
```


### Train

First, select and revise the specific config file with your wanted network architecture (e.g. TransT and Ocean). 
All config files have been placed in `ltr/train_settings/spot/`. 

Take SP-TransT-3 (sparsely-supervised TransT with 3 labels on each training video) as example, 
where its corresponding file is named as `spot_transt_003.py`. 
Please refer to subsection [Customizable config files](#customizable-config-files) for details. 
In most cases, the config files provided by default are already enough. 

Second, download or resample the key frame files. 
Please refer to subsection [Label sampling](#label-sampling) below for details.

Finally, you can start training the sparsely-supervised trackers (e.g SP-TransT-3).
```
python -u scripts/run_training.py spot spot_transt_003
```
Training SP-Ocean needs additional preparations. 
Please check subsection [Ocean support](#ocean-support) below for details.


### Onekey for train/test/eval (recommended)

We provided a onekey script for training, testing and evaluation. 

For SP-TransT-3, you can start all these phases with the following script.
```
python scripts/onekey.py --train --test --eval --module spot --arch transt --variant spot_transt_003 \
        --start_epoch 325 --end_epoch 400 --dataset LaSOT --gpus 0,1,2,3 --threads 4 --ts_tag t
```

For SP-Ocean-3, the script is typically as follows.
```
python scripts/onekey.py --train --test --eval --module spot --arch ocean --variant spot_ocean_003 \
        --start_epoch 275 --end_epoch 350 --dataset LaSOT --gpus 0,1,2,3 --threads 4 --ts_tag t
```


## Advanced Features

Here introduces some advanced features in SPOT.

### Label sampling

The label sampling results in training videos (i.e., labeled frames) may affect the overall performance. 
As a result, we provide the pre-sampled labeled frames in each video for fair comparison. 

The pre-sampling results (e.g. `LaSOT_train_kf_003.json` that samples 3 labeled frames in each LaSOT video) 
are all provided in [Google Drive (sampling_results)](https://drive.google.com/drive/folders/1GBxAIIK4Tsgu7_17_iyGCrvc1gBcZAB-?usp=drive_link). 
Please download all these files and place them in `sampling/sampling_results/`. 
The SPOT framework will automatically detect these files, and treat these sampled frames as labeled ones. 

Without the downloaded sampling result files, the framework can also sample from scratch. 
The following script is necessary if you would like to resample labeled frames. 
```
# Take k = 3 and the full training set as an example
cd sampling/vanilla_sampling
python key_frame_sampling.py --dataset GOT-10k --split vottrain --num 3 --overwrite
python key_frame_sampling.py --dataset LaSOT --split train --num 3 --overwrite
python key_frame_sampling.py --dataset TrackingNet --split 0,1,2,3 --num 3 --overwrite --merge
```

### Ocean support

SP-Ocean requires additional package (i.e., deform_conv) installed. 
The following command is necessary iff. you need to train/test the Ocean tracker.
```
python scripts/setup.py develop
```

For training SP-Ocean, a ResNet-style pretrained backbone is additionally needed. 
Please download `imagenet_pretrain.model` 
from [Google Drive (pretrain)](https://drive.google.com/drive/folders/1GBxAIIK4Tsgu7_17_iyGCrvc1gBcZAB-?usp=drive_link), 
and place the downloaded backbone file in `ltr/pretrain/`. 
The framework will automatically detect and load it when you start to train SP-Ocean. 


### Customizable config files

Customizable config file determines everything important during training and testing. 
All these config files are placed in folder `ltr/train_settings/spot/`. 
We have already provided template config files for SP-TransT-k and SP-Ocean-k (k=2,3,5). 
In most cases, these default config files provided are already enough. 

If you would like to revise the config file, we actually provide very detailed comments on the files. 
Please carefully check them and corresponding source codes before you start a new training attempt.


### Suspend and resume for training

During training, we store checkpoints for both the teacher and student every T epochs.

T is actually a editable parameter as `settings.ckp_saving_interval=25`.
You can revise T in the customizable config files in folder `ltr/train_settings/spot/`. 

If you killed the training process and want to resume that some time later, 
the framework will automatically detect the latest stored checkpoints (both teacher and student).
These checkpoints are placed in folder `var/checkpoints/ltr/spot/`, and 
this would restart the training process from that certain (i.e. the latest) epoch. 

To fully recover the training state in the sparsely-supervised training phase, 
a sampler state file (snapshot for sampler state) is also required. Check below for more details.

### Sampler state

The sampler state is a dict file storing all explored frames and the coarse locations of the target in each frame. 
This file will be initialized from the beginning, and accumulated with new predictions throughout the training phase.

This file is stored in `var/records`, and named as `sampler_state_{}.json`. 
Here the brace will be filled with the parameter `settings.magic_number` in customizable config files. 

At the end of each training epoch, the sampler state will be updated by new predictions in this epoch.
Additional snapshots for sampler state is saved every `settings.state_saving_interval` epochs (100 by default). 
This feature is designed for suspend and resume training in the sparsely-supervised training phase.  

NOTE: With the same magic number, the framework will detect and load the corresponding sampler state, 
and continue exploring the video from that certain recorded state. 
So make sure you use a unique magic number never used before, 
for each time you would like to start a new training attempt. 