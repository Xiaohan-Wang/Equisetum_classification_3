## Equisetum classification

### Dataset
Please download our Equisetum dataset [here](). Put **Images** and **Annotations** under the **Equisetum** folder.

In our Equisetum dataset, the three json files are used as training, validation and test set, respectively. You can also split the dataset by yourself with `Equisetum/split_datasets.py`.

**[Note]**: Before running any code in the repo, make sure that you change parameter `code_dir` in `config.py` to your own repo directory. 
 
**[Note]**: To make the code work successfully, please run the mentioned scripts with `python -m Equisetum_classification_3.[folder].[script_name]`. For example, run `Equisetum/split_datasets.py` with `python -m Equisetum_classification_3.Equisetum.split_datasets`.

### Node detection
1. We use SSD to detect nodes from Equisetum images.  
Paper：[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf). Its official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).   
We use a pytorch implementation for SSD as our base model, and the original repo is [here](https://github.com/amdegroot/ssd.pytorch).

2. A pretrained node detection model can be found [here](https://drive.google.com/file/d/1pzNDBNb4r3Dj2_UVIlTJSsWh-pnyKxQQ/view?usp=sharing). Put the detection model under **node_detection/weights/Equisetum** folder. To train the model from scratch, run `node_detection/train.py`.

3. You can predict nodes for test images by `node_detection/get_predictions.py`. This file should generate three json files stored in **results/predictions**, which contain all the detected bounding box for each image in each set. 

4. With `node_detection/visualization.py`, you can visualize the prediction result by putting the predicted bounding box on original image. The results are saved in **results/visualization**.

### Equisetum Classification
1. `classification/get_features.py` is provided to extract useful features for classification task from the raw bounding box information. The output features can be found at **results/features**.

2. With the extracted features, `classification/classifier.py` trains a linear regression classifier to finish the Equisetum classification task. It should report the classification accuracy and generate a confusion table stored in **results/classification**.


