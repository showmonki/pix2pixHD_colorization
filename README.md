# pix2pix HD model for colorization task
- original repo: https://github.com/NVIDIA/pix2pixHD
- usage: train own datasets for colorization task without prepare two dataset "_A" & "_B"
- main advantage: prepare datasets "datasets/<project_name>/" , with two folders "train_img" and "test_img"
- why convenient: when own dataset has lots of images, two folders gray and color will cost spaces

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- and also below required libraries.
```bash
pip install requirements.txt -r
```
- Training:
```bash
bash ./scripts/train_color.sh
# need change dataroot dir
# 2 epoch for time estimation
```

- Test:
```bash
bash ./scripts/test_color.sh
# need change dataroot dir
# 2 epoch for time estimation
```

## TODOs
- [ ] vgg loss
- [ ] fix test process GPU no used.