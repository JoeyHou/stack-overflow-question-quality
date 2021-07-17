# Analysis and Modelling of Stack Overflow Question Quality
------
## 0. Overview
This is the GitHub repository to analyze and to model the quality of StackOverflow questions. It contains three different models (XGBoost, LSTM, and Pre-trained BERT) and the designed usage for those model is on [AWS Sagemaker](https://aws.amazon.com/sagemaker/). 

## 1. Projcet Orgnization
------------
    ├── README.md   		<- This file
    ├── cache					<- Cache data files (created during excution) 
    ├── data
    │   ├── raw            <- Raw data from Kaggle (see data section for details)
    │   └── processed      <- Processed data (created during excution) 
    ├── notebooks          <- Jupyter notebooks.
    ├── src_bert           <- Source code for BERT model
    ├── src_lstm           <- Source code for LSTM model
    ├── report.pdf         <- Final report
    └── proposal.pdf       <- Project proposal
    

--------
## 2. Project Excution
- **Environment Setup:** The entire project is designed to be excuted on [AWS Sagemaker](https://aws.amazon.com/sagemaker/). To replicate the project, please create an AWS notebook instance and select `clone from public GitHub repository` during creation.
- **Required Software:** 
    - `NLTK`
    - `SageMaker`
    - `Scikit-Learn`
- **Processing Pipeline**:
	- Data Transformation: clean the raw data, save to `cache` folder or processed `folder`.
	- EDA (Exploratory Data Analysis): Analyze data distribution & frequent word, etc.
	- Modeling: building models, including baesline, sklearn (XGBoost), LSTM, HuggingFace Model.
    - Result Analysis: analyzing model prediction results.
