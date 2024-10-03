# Navigating weight prediction with diet diary
This repo is the official implementation for the paper: [Navigating weight prediction with diet diary](https://www.arxiv.org/abs/2408.05445). [[website]](https://yxg1005.github.io/weight-prediction)

## Introduction
Current research in food analysis primarily concentrates on tasks such as food recognition, recipe retrieval and nutrition estimation from a single image. Nevertheless, there is a significant gap in exploring the impact of food intake on physiological indicators (e.g., weight) over time. This paper addresses this gap by introducing the **DietDiary** dataset, which encompasses daily dietary diaries and corresponding weight measurements of real users. Furthermore, we propose a novel task of weight prediction with a dietary diary that aims to leverage historical food intake and weight to predict future weights.

## Dataset Overview
![ss](Weight_Prediction/pics/dataset-example.png)
We introduce a novel dataset, DietDiary, specifically for analyzing weight in relation to food intake. DietDiary encompasses diet diary of three meals over a period of time, accompanied by daily weight measurement. This example shows data records for two participants with different weight fluctuation trends in DietDiary. The records leading to weight gain are highlighted in red.
