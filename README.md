# How to Improve Representation Alignment and Uniformity in Graph-based Collaborative Filtering?
[![Static Badge](https://img.shields.io/badge/Python-v3.9+-blue)](https://img.shields.io/badge/PyTorch-v1.13+.0-blue) [![Static Badge](https://img.shields.io/badge/PyTorch-v1.13+-brightgreen)](https://img.shields.io/badge/PyTorch-v1.13+.0-brightgreen)


Welcome to the official code repository for the **ICWSM 2024** paper paper "How to Improve Representation Alignment and Uniformity in Graph-based Collaborative Filtering?". The code structure adapts from [SELFRec](https://github.com/Coder-Yu/SELFRec).

## Introduction
Our framework performs self-supervised contrastive learning on the user and item representations from the perspective of label-irrelevant alignment and uniformity, in addition to lable-relevant representation alignment and uniformity. With representations less dependent on label supervision, our framework therefore captures more label-irrelevant data structures and patterns, leading to more generalized representation alignment and uniformity.

## Getting Started

1. **Preparation:** 
   - Create a directory named `results/` in the project's root directory to store output files.

2. **Configurations:**
   - Navigate to `conf/AUPlus.conf` to adjust model settings, including hyper-parameters and dataset specifications. The `mode` parameter can be set as follows:
     - `0`: The default AU<sup>+</sup> model.
     - `1`: The AU<sup>+</sup>-AU variant, where augmented views are restrained with the alignment and uniformity losses.
     - `2`: The AU<sup>+</sup>-SGL (edge drop) variant, where augmented views are generated via edge drop in SGL.

3. **Run the Model:**
  From the root directory, run:
    ```
    python main.py --model=AUPlus
    ```

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{ouyang2024auplus
    title={How to Improve Representation Alignment and Uniformity in Graph-based Collaborative Filtering?},
    author={Zhongyu Ouyang and Chunhui Zhang and Shifu Hou and Chuxu Zhang and Yanfang Ye},
    booktitle={International AAAI Conference on Web and Social Media},
    year={2024}
}
```
