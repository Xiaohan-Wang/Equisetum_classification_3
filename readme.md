## Equisetum classification

### Dataset
Please download our Equisetum dataset [here](). Put **Images**, **Annotations**, and the three json files under **Equisetum** folder.

In our Equisetum dataset, the three json files are used as training, validation and test set, respectively. You can also split the dataset by yourself with **split_datasets.py** in the folder.

### Node detection
1. We use SSD to detect nodes from Equisetum images.  
paper：[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf).  
The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).  
We use a pytorch implementation for SSD as our base model, and the original repo is [here](https://github.com/amdegroot/ssd.pytorch).

2. Change the code directory in **config**.

3. A pretrained node detection model can be found [here](). Put the detection model under **node_detection/weights/Equisetum** folder. To train the model from scratch, run **node_detection/train.py**.

4. You can use the model to predict nodes for test images by **node_detection/get_predictions.py**. This file should generate a json file stored in **node_detection/results/predictions** containing all the detected bounding box for each image in the test set. 

5. With **visualization.py**, you can visualize the prediction result by having the predicted bounding box on original image. The result si outputed to **results/visualization**.
### 

