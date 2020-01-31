## Equisetum classification

### SSD model
paper：[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)
The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).

We use a pytorch implementation for SSD as our base model. The original repo is [here](https://github.com/amdegroot/ssd.pytorch)

### Dataset
Please download our Equisetum dataset [here](). Put it under the main folder.

In our Equisetum dataset, there are three json file for training, val and test set, seperately. You can also split the dataset by yourself with data_processing/split_dataset
