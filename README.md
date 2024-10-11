# Navigating weight prediction with diet diary
This repo is the official implementation for the oral paper published on ACM Multimedia 2024: [Navigating weight prediction with diet diary](https://www.arxiv.org/abs/2408.05445). [[website]](https://yxg1005.github.io/weight-prediction)

## Introduction
Current research in food analysis primarily concentrates on tasks such as food recognition, recipe retrieval and nutrition estimation from a single image. Nevertheless, there is a significant gap in exploring the impact of food intake on physiological indicators (e.g., weight) over time. This paper addresses this gap by introducing the **DietDiary** dataset, which encompasses daily dietary diaries and corresponding weight measurements of real users. Furthermore, we propose a novel task of weight prediction with a dietary diary that aims to leverage historical food intake and weight to predict future weights. We hope this can offer valuable insights for individuals aiming to monitor their diet and manage their weight and health effectively over the long term.

<p align="center">
  <img src=".//pics//task-overview-1.png" alt="Weight Prediction Model" />
</p>

## Dataset Overview
We introduce a novel dataset, **DietDiary**, specifically for analyzing weight in relation to food intake. DietDiary encompasses diet diary of three meals over a period of time, accompanied by daily weight measurement. This example shows data records for two participants with different weight fluctuation trends in DietDiary. The records leading to weight gain are highlighted in red. The dataset is publicly available at [Google Drive](https://drive.google.com/drive/folders/134Y0rgylxAP37DFOx6whhOoz45abA4sr?usp=sharing) now.
![](.//pics//dataset-example.png)

## Getting Started
### 1. Install
Install [Pytorch](https://pytorch.org/get-started/locally/), [CLIP](https://github.com/openai/CLIP) and necessary dependencies.
```python
pip install -r requirements.txt
```

### 2. Data Preparation
* Open the [Google Drive](https://drive.google.com/drive/folders/134Y0rgylxAP37DFOx6whhOoz45abA4sr?usp=sharing) and download data.
```python
mkdir ./dataset
```
Please put the **data.csv** and **predict_ingr.json** in the `./dataset`.
Please download and unzip the DietDiary.zip to your directory, and update the *--image_root* argument in `run_longExp.py` to your image path.

* (Optional) Download the feature files from [feature_files](https://drive.google.com/drive/folders/1EdRdQysgzQetPp7WSSnmUDrTyGGV-4wz?usp=sharing). Then put **LTSF-img-npy**, **LTSF-txt-npy** and **LTSF-txt-from-img-npy** to `./features`. 
```python
mkdir ./features
```
If do not download, the code will create `./feature` automatically and extract features, and then save them to this path.

### 3. Training and Evaluation
We provide implementation of *NLienar/iTransformer/PatchTST* under the folder `./RUN_SH/`. You can reproduce the results as the following examples. Logs will be stored in `logs/`. Prediction results wiil be stored in `results/`.

To train and evaluate the baseline *iTransformer* model:
```python
sh RUN_SH/iTransformer/S.sh
```
To train and evaluate the *NLinear* model with diet information:
```python
# NLinear model with diet information from images
sh RUN_SH/NLinear/image.sh

# NLinear model with diet information from ingredients(users)
sh RUN_SH/NLinear/text.sh

# NLinear model with diet information from ingredients(ingredients prediction model)
sh RUN_SH/NLinear/lmm.sh

# NLinear model with diet information from both images and ingredients(users)
sh RUN_SH/NLinear/txt_img_early_fusion.sh
```

#### Notice
* For ablation study of number of meals, simply change *breakfast/lunch/supper* argument in `RUN_SH/model_name/.sh` to 1 (to include the meal) or 0 (to exclude the meal).
* For ablation study of hyper-parameter of 𝜆, set *Lambda* argument in `RUN_SH/model_name/.sh`.
* For only evaluation, set *is_training* argument in `RUN_SH/model_name/.sh` as 0.

## Results
Compared to models that do not incorporate food intake information, our method consistently achieves superior performance over NLinear and iTransformer across all evaluated settings.

![](.//pics//exp-result.png)

## Citation
If you find this repository useful for your work, please consider citing it as follows:
```python
@inproceedings{gui2024navigating,
  title={Navigating Weight Prediction with Diet Diary},
  author={Gui, Yinxuan and Zhu, Bin and Chen, Jingjing and Ngo, Chong-Wah and Jiang, Yu-Gang},
  booktitle={ACM Multimedia 2024}
}
```

## Acknowlegement
We sincerely thank the authors of [NLinear](https://github.com/cure-lab/LTSF-Linear), [iTransformer](https://github.com/thuml/iTransformer), [PatchTST](https://github.com/yuqinie98/PatchTST) and [FoodLMM](https://github.com/YuehaoYin/FoodLMM) for their valuable code and efforts.
