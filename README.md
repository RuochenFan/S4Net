## Dataset Preparation

You can download the dataset in pickle format from https://drive.google.com/open?id=1-Yn_9GMjeu-d8gLZ26t3bvH6yX_BMFfm. Or run make_saliency_dataset.py to prepare the dataset by yourself. The pickle file should be placed in ./data directory.

## Test

Our pretrained weights can be found in https://drive.google.com/open?id=1TeJw415uNGwmiOT1v5iLIxcGNFWGriW4, you can unzip it and place it into ./logs.

Simply run:

```
cd experiment
python3 test_seg.py
```

## Train

Download ImageNet pretrained weights for FPN from https://drive.google.com/open?id=12LDpUybjnbcoO3dAwYpS6tryzx29-Viu, unzip and place it into ./data.

This training scripts can run on multi-GPU mode. You can set GPU ids in ./experiment/config.py.

The training process is quite easy, just run:

```
cd experiment
python3 train_multi_gpu.py
```

