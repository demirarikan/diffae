# Forked implementation of Diffusion Autoencoders for Computational Surgineering practical course in TUM

Information about the original paper:

A CVPR 2022 (ORAL) paper ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html), [site](https://diff-ae.github.io/), [5-min video](https://youtu.be/i3rjEsiHoUU)):

```
@inproceedings{preechakul2021diffusion,
      title={Diffusion Autoencoders: Toward a Meaningful and Decodable Representation}, 
      author={Preechakul, Konpat and Chatthee, Nattanat and Wizadwongsa, Suttisak and Suwajanakorn, Supasorn},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
      year={2022},
}
```

## Description
This project uses Diffusion Autoencoders (DiffAE) to convert from CT labelmap generated ultrasound images into realistic ultrasound images.
This project was done as a part of Computational Surgineering practical course at TUM WS23/24

## Usage
Check the following notebooks for examples on how to load and use the models:
- us_autoencoding.ipynb
- us_encoding.ipynb
- us_manipulation.ipynb

For reference notebooks from the original implementation have been renamed but left untouched otherwise. The naming scheme is: zzz_example_XYZ.ipynb


### Prerequisites

The original repo has been implemented and tested on Python 3.8.10. So it is highly advised to use that Python version. 
Easiest was to create a Python environment with a specific version is to use conda. More details about that can be found [here](https://docs.conda.io/en/latest/).

After setting up the environment install go to [Pytorch website](https://pytorch.org/get-started/previous-versions/#v181) and install the version 1.8.1 (with cuda if it applies).

Because the required version of pytroch-fid package has a bug in it, it cannot be directly installed with pip. For this reason a fixed versin of the packge has been included in the repo. Just run the following command to install it while in the correct path.

`pip install pytorch-fid-0.2.0`

For the rest of the requirements see `requirements.txt`

```
pip install -r requirements.txt
```

### Checkpoints
Checkpoints can be found [here](https://drive.google.com/drive/folders/1ks-PABGPQ5oVvzmLtxxChSdQvr1QU529?usp=sharing).

### Datasets
Datasets for TUM computational surgineering students can be found on SLURM. Contact the supervisors to gain access.
- Simulated cropped images: /home/data/farid/simulated_images_cs_Demir_Yiched_Daniel/source_train_cropped
- Real cropped images: /home/data/farid/simulated_images_cs_Demir_Yiched_Daniel/target_train_cropped

## Training
For training the network the us_training.py script can be used. Example usage:

python us_training.py --login your_wandb_key_here --datatype mixed --dataset path/to/simulated/dataset path/to/real/dataset

The real and sim datatype arguments are artefacts from initial testing of the network and can be removed

To change the training parameters check out templates.py and create your own based on the examples. 
