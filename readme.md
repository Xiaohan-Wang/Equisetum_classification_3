## Equisetum classification

### Dataset
Please download our Equisetum dataset [here](). Put it under the main folder.

In our Equisetum dataset, there are three json files for training, validation and test set, respectively. You can also split the dataset by yourself with **split_datasets.py** in the folder.

### Node detection
We use SSD to detect nodes from Equisetum images.

paper：[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf).  
The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).

We use a pytorch implementation for SSD as our base model, and the original repo is [here](https://github.com/amdegroot/ssd.pytorch).

[Note]: Change the code directory in **config**.

A pretrained node detection model is in **node_detection/weights/Equisetum**. To train the model from scratch, run **node_detection/train.py**.

You can use the model to predict nodes for test images by **node_detection/get_predictions.py**. This file should generate a json file stored in **node_detection/results/predictions** containing all the detected bounding box for each image in the test set. 

With **visualization.py**, you can visualize the prediction result by having the predicted bounding box on original image. The result si outputed to **results/visualization**.
### 

