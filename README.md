# [PFG, 2024] DUDES: Deep Uncertainty Distillation using Ensembles for Semantic Segmentation

This repo is the official implementation of [DUDES: Deep Uncertainty Distillation using Ensembles for Semantic Segmentation](https://link.springer.com/article/10.1007/s41064-024-00280-4), which was accepted for publication in the [PFG - Journal of Photogrammetry, Remote Sensing and Geoinformation Science](https://link.springer.com/journal/41064).

## Motivation
In this work, we apply student-teacher distillation to accurately approximate predictive uncertainties with a single forward pass while maintaining simplicity and adaptability. Experimentally, DUDES accurately captures predictive uncertainties without sacrificing performance on the segmentation task and indicates impressive capabilities of highlighting wrongly classified pixels and out-of-domain samples through high uncertainties on the Cityscapes and Pascal VOC 2012 dataset.

## Requirements
### Environment
First, clone this repo:
```shell
git clone https://github.com/StevenLandgraf/DUDES.git
cd DUDES/
```

Second, create a Docker Image with the provided Dockerfile:
```shell
docker build --rm -t <image_name> .
```

Third, start a Docker Container with the Docker Image:
```shell
docker run --name <container_name> [--docker_options] --volume /your_path_to/DUDES:/workspace <image_name> bash
docker attach <container_name>
```

Finally, install additional requirements:
```shell
pip install -r requirements.txt
```

Now, you should be good to go!

### Datasets
Create a data folder:
```shell
mkdir data
cd data/
```

#### Cityscapes
Download the dataset with wget:
```shell
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EWoa_9YSu6RHlDpRw_eZiPUBjcY0ZU6ZpRCEG0Xp03WFxg\?e\=LtHLyB\&download\=1 -O cityscapes.zip
unzip cityscapes.zip
```

#### Pascal VOC 2012 Dataset
Download the dataset with wget:
```shell
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EcgD_nffqThPvSVXQz6-8T0B3K9BeUiJLkY_J-NvGscBVA\?e\=2b0MdI\&download\=1 -O pascal.zip
unzip pascal.zip
```
