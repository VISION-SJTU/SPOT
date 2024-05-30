# SPOT Performance

Performance Page for Sparsely-Supervised Object Tracking (SPOT).

## Performance

Performance of SPOT trackers on recent SOT benchmarks with different architecture and supervision settings. 

We use a mixture of the LaSOT, TrackingNet and GOT-10k training sets as a full training set, 
in order to report the performance on majority of benchmarks. 
In contrast, we follow the GOT10k protocol to train models on the GOT-10k training set only,
in order to report the performance on the GOT-10k testing set. 

|   Arch  | Label Sparsity | Labels (full/got) | LaSOT Suc. | TrackingNet Suc. | GOT10k AO | NFS30 Suc. | UAV123 Suc. | OTB100 Suc. |
| :-------: |:--------------:|:-----------------:|:----------:|:----------------:|:---------:|:----------:|:-----------:|:-----------:|
| TransT     |      k=2       |     36K / 19K     |    59.2    |       78.0       |   64.8    |     -      |      -      |      -      | 
| TransT     |      k=3       |     55K / 28K     |    62.3    |       79.0       |   66.3    |    64.9    |    61.6     |    68.2     | 
| TransT     |      k=5       |     91K / 47K     |    61.7    |       79.4       |   66.4    |     -      |      -      |      -      | 
| Ocean      |      k=2       |     36K / 19K     |    53.9    |       73.9       |   58.7    |     -      |      -      |      -      |  
| Ocean      |      k=3       |     55K / 28K     |    54.1    |       74.8       |   60.0    |    58.7    |    60.7     |    67.4     | 
| Ocean      |      k=5       |     91K / 47K     |    55.4    |       75.3       |   57.6    |     -      |      -      |      -      | 

All checkpoints and raw results for SP-TransT and SP-Ocean with k=2,3,5 can be found 
in [Google Drive (snapshot/results)](https://drive.google.com/drive/folders/1GBxAIIK4Tsgu7_17_iyGCrvc1gBcZAB-?usp=drive_link).


