# Navigating weight prediction with diet diary
Official implementation for "Navigating Weight Prediction with Diet Diary" (MM 2024 Oral)

## Dataset Overview

## Usage 
1. The dataset will be released soon! 

2. Train and evaluate the model. We provide main experiments under the folder ./RUN_SH/. Under this folder. You can reproduce the results as the following examples:

```
# Weight prediction using "NLinear" model with food images as diet information
bash ./RUN_SH/NLinear/image.sh

# Weight prediction using "NLinear" model with fusioning food images and text from users as diet information
bash ./RUN_SH/NLinear/txt_img_early_fusion.sh

# Weight prediction using "iTransformer" model with ingredients from ingredient prediction model as diet information
bash ./RUN_SH/iTransformer/lmm.sh
