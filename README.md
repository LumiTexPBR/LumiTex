# Geometry-guided LVSM
The code is modified from [LACT-NVS](https://github.com/a1600012888/LaCT/tree/main/lact_nvs). We changed the code to support geometry-guided LVSM. To be specific, we add ccm map and normal map as additional conditioned, and replace the perspective projection's ray map with orthogonal projection's ray map.

## Environment Setup
Install the python dependencies:
```
pip install -r requirement.txt
```

Install `ffmpeg` to save rendering results as mp4 video:
```
sudo apt install ffmpeg
```
If ffmpeg can not be installed, changing `*_turntable.mp4` to `*_turntable.gif` in code and it will save in gif (but the size of video is larger).

## Training Script 

```
torchrun \
--nproc_per_node=8 \
--standalone \
train.py --config config/lact_l14_d768_ttt2x.yaml --actckpt
```
We provide an exmaple of dataset in /dataset_example. You can use it to test the code. For each object, we render 26 views of albedo, ccm and normal maps using orthogonal projection camera using a custom renderer based on nvdiffrast.



