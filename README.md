# Navigating weight prediction with diet diary
This repo is the official implementation for the paper: [Navigating weight prediction with diet diary](https://www.arxiv.org/abs/2408.05445). [[website]](https://yxg1005.github.io/weight-prediction)

## Introduction
Current research in food analysis primarily concentrates on tasks such as food recognition, recipe retrieval and nutrition estimation from a single image. Nevertheless, there is a significant gap in exploring the impact of food intake on physiological indicators (e.g., weight) over time. This paper addresses this gap by introducing the **DietDiary** dataset, which encompasses daily dietary diaries and corresponding weight measurements of real users. Furthermore, we propose a novel task of weight prediction with a dietary diary that aims to leverage historical food intake and weight to predict future weights.

## Dataset Overview
We introduce a novel dataset, **DietDiary**, specifically for analyzing weight in relation to food intake. DietDiary encompasses diet diary of three meals over a period of time, accompanied by daily weight measurement. This example shows data records for two participants with different weight fluctuation trends in DietDiary. The records leading to weight gain are highlighted in red. The dataset is publicly available at [Google Drive](https://drive.google.com/drive/folders/1XYkdJAlY-PIPd3MQWNnX9jlOvOs2RZ36?usp=sharing) now.
![](.//pics//dataset-example.png)

## Getting Started
### 1. Install
Install [Pytorch](https://pytorch.org/get-started/locally/), [CLIP](https://github.com/openai/CLIP) and necessary dependencies.
```python
pip install -r requirements.txt
```

### 2. Data Preparation
Open the  [Google Drive](https://drive.google.com/drive/folders/1XYkdJAlY-PIPd3MQWNnX9jlOvOs2RZ36?usp=sharing) and download files as following:
```python
Weight_Prediction
├── dataset
│   ├── data.csv
│   ├── predict_ingr.json
```
The above two documents are already included in this repo.  

```python
feature path
├── LTSF-img-npy              #image feature of different settings
├── LTSF-txt-npy              #ingredients(from users) feature of different settings
├── LTSF-txt-from-img-npy     #ingredients(from ingredients prediction model) feature of different settings
```
Please download the three feature documents mentioned above to your directory, and update the *--feature_path* argument in **run_longExp.py** to your feature path.

```python
image path
├── DietDiary.zip
```
Please download and unzip the DietDiary dataset to your directory, and update the *--image_root* argument in **run_longExp.py** to your image path.

### 3. Training and Evaluation
We provide implementation of *NLienar/iTransformer/PatchTST* under the folder `./RUN_SH/`. You can reproduce the results as the following examples.

To train and evaluate the baseline *iTransformer* model:
```python
bash RUN_SH/iTransformer/S.sh
```



