# [AAAI 2025] Surgical Workflow Recognition and Blocking Effectiveness Detection in Laparoscopic Liver Resections with Pringle Maneuver

<div align=center>
<img src="assets/Pipeline.png">
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Abstract

> Pringle maneuver (PM) in laparoscopic liver resection aims to reduce blood loss and provide a clear surgical view by intermittently blocking blood inflow of the liver, whereas prolonged PM may cause ischemic injury. To comprehensively monitor this surgical procedure and provide timely warnings of ineffective and prolonged blocking, we suggest two complementary AI-assisted surgical monitoring tasks: workflow recognition and blocking effectiveness detection in liver resections. The former presents challenges in real-time capturing of short-term PM, while the latter involves the intraoperative discrimination of long-term liver ischemia states. To address these challenges, we meticulously collect a novel dataset, called PmLR50, consisting of 25,037 video frames covering various surgical phases from 50 laparoscopic liver resection procedures. Additionally, we develop an online baseline for PmLR50, termed PmNet. This model embraces Masked Temporal Encoding (MTE) and Compressed Sequence Modeling (CSM) for efficient short-term and long-term temporal information modeling, and embeds Contrastive Prototype Separation (CPS) to enhance action discrimination between similar intraoperative operations. Experimental results demonstrate that PmNet outperforms existing state-of-the-art surgical workflow recognition methods on the PmLR50 benchmark. Our research offers potential clinical applications for the laparoscopic liver surgery community.


## 🔥🔥🔥 News!!
* Dec 12, 2024: 🤗 Our work has been accepted by AAAI 2025! Congratulations!
* Feb 21, 2025: 🚀 Code and dataset have been released! [Dataset Link](https://docs.google.com/forms/d/e/1FAIpQLSf33G5mdwXeqwabfbXnEboMpj48iCNlQBAY_up4kLuZiqCPUQ/viewform?usp=dialog)


## PmLR50 Dataset and PmNet
### Installation
* Environment: CUDA 12.5 / Python 3.8
* Device: Two NVIDIA GeForce RTX 4090 GPUs
* Create a virtual environment
```shell
git clone https://github.com/RascalGdd/PmNet.git
cd PmNet
conda env create -f PmNet.yml
conda activate Pmnet
```
### Prepare your data
Download processed data from [PmLR50](https://docs.google.com/forms/d/e/1FAIpQLSf33G5mdwXeqwabfbXnEboMpj48iCNlQBAY_up4kLuZiqCPUQ/viewform?usp=dialog);
The final structure of datasets should be as following:

```bash
data/
    └──PmLR50/
        └──frames/
            └──01
                ├──00000000.jpg
                ├──00000001.jpg
                └──...
            ├──...    
            └──50
        └──phase_annotations/
            └──01.txt
            ├──02.txt
            ├──...
            └──50.txt
        └──blocking_annotations/
            └──01.txt
            ├──02.txt
            ├──...
            └──50.txt
        └──bbox_annotations/
            └──01.json
            ├──02.json
            ├──...
            └──50.json
```
Then, process the data with [generate_labels_pmlr.py](https://github.com/RascalGdd/PmNet/blob/main/datasets/data_preprosses/generate_labels_pmlr.py) to generate labels for training and testing.

### Training
We provide the script for training [train.sh](https://github.com/RascalGdd/PmNet/blob/main/train.sh) and testing [test.sh](https://github.com/RascalGdd/PmNet/blob/main/test.sh).

run the following code for training

```shell
sh train.sh
```
and run the following code for testing

```shell
sh test.sh
```
The checkpoint of our model is provided [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155229775_link_cuhk_edu_hk/EZVHcTmQBY1Mv1zTSLEtu0cBKTA7zTNURaG65gWWloqFmg?e=Zudo2X).

### More Configurations

We list some more useful configurations for easy usage:

|        Argument        |  Default  |                Description                |
|:----------------------:|:---------:|:-----------------------------------------:|
|  `--nproc_per_node`   |  2  |    Number of nodes used for training and testing    |
|       `--batch_size`       |   8    |   The batch size for training and inference   |
|     `--epochs`     | 50  |      The max epoch for training      |
|    `--save_ckpt_freq`    |    10    |     The frequency for saving checkpoints during training     |
|    `--nb_classes`     |    5     |     The number of classes for surgical workflows      |
| `--data_strategy` |    online    |    Online/offline mode       |
|     `--num_frames`     |    20    | The number of consecutive frames used  |
|     `--sampling_rate`   |    8  | The sampling interval for comsecutive frames |
|        `--enable_deepspeed`        |    True  |   Use deepspeed to accelerate  |
|  `--dist_eval`   |   False   |    Use distributed evaluation to accelerate    |
|  `--load_ckpt`   |   --   |    Load a given checkpoint for testing    |

## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [TMRNet](https://github.com/YuemingJin/TMRNet)
- [Surgformer](https://github.com/isyangshu/Surgformer/)
- [TimeSformer](https://github.com/facebookresearch/TimeSformer)

## Citation 
If you find our work useful in your research, please consider citing our paper:

    @inproceedings{guo2025surgical,
      title={Surgical Workflow Recognition and Blocking Effectiveness Detection in Laparoscopic Liver Resection with Pringle Maneuver},
      author={Guo, Diandian and Si, Weixin and Li, Zhixi and Pei, Jialun and Heng, Pheng-Ann},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={39},
      number={3},
      pages={3220--3228},
      year={2025}
    }
