## Equisetum classification

### Dataset
Please download our Equisetum dataset [here](). Put it under the main folder.

In our Equisetum dataset, there are three json files for training, validation and test set, respectively. You can also split the dataset by yourself with **utils/split_datasets.py**

### Node detection
In this part, we use 
SSD model
We use SSD to detect nodes from Equisetum images.

paper：[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf).  
The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).

We use a pytorch implementation for SSD as our base model, and the original repo is [here](https://github.com/amdegroot/ssd.pytorch).

### 