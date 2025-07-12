
# Human-Object Interaction from Human-Level Instructions (ICCV 2025)

This is the official implementation for the ICCV 2025 [paper](https://arxiv.org/abs/2406.17840). For more information, please check the [project webpage](https://hoifhli.github.io/).

![CHOIS Teaser](teaser.png)

### Environment Setup
> Note: This code was developed on Ubuntu 20.04 with Python 3.8, CUDA 12.4, and PyTorch 1.11.0.

Clone the repo.
```
git clone https://github.com/zhenkirito123/hoifhli_release.git
cd hoifhli_release/
```
Create a virtual environment using Conda and activate the environment. 
```
conda env create -f environment.yml
conda activate hoifhli_env 
```


### Prerequisites 
Please download [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) and put the model to ```data/smpl_all_models/```. The file structure should look like this:

```
data/
├── smpl_all_models/
│   ├── smplx/
│   │   ├── SMPLX_FEMALE.npz
│   │   ├── SMPLX_MALE.npz
│   │   ├── SMPLX_NEUTRAL.npz
│   │   ├── SMPLX_FEMALE.pkl
│   │   ├── SMPLX_MALE.pkl
│   │   ├── SMPLX_NEUTRAL.pkl

```

### Sampling long sequence for OMOMO objects



### Citation
```
@article{wu2024human,
  title={Human-object interaction from human-level instructions},
  author={Wu, Zhen and Li, Jiaman and Xu, Pei and Liu, C Karen},
  journal={arXiv preprint arXiv:2406.17840},
  year={2024}
}
```

### Related Repos
We adapted some code from other repos in data processing, learning, evaluation, etc. We would like to thank the authors and contributors of these repositories for their valuable work and resources.
```
https://github.com/lijiaman/chois_release
https://github.com/otaheri/GRAB
https://github.com/nghorbani/human_body_prior
https://github.com/PKU-EPIC/DexGraspNet
```