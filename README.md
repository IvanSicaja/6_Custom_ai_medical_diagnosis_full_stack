Used dataset you can download from the following links:
https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection
https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

This dataset are downloaded and edited with scripts:
unifyImageSizeFromFolderAndSave.py
makeCSVDatasetFromImageFolders.py

Edited datasets are saved and used for training:
Br35H.csv
melanoma_cancer.csv
chest_xray.csv

For training are used following scripts:
1.0_trainCNN.py
1.1_trainCNN_CatchComplexPatterns.py
1.2_trainCNN_CatchComplexPatterns_DA_EarlyStopping.py
1.2_trainCNN_CatchComplexPatterns_DA_EarlyStopping_128x128.py
1.2_trainCNN_CatchComplexPatterns_DA_EarlyStopping_MoreLayers_128x128.py
1.3_trainCNN_CatchComplexPatterns_DA_EarlyStopping_ClassBalanced_ReducedLearningRate.py
1.4_trainCNN_MediumComplexity_DA_EarlyStopping_Droput_L2.py
1.5_trainCNN_smallerNumberOfFilters.py

For database creation are used following scripts:
create_db.py
db_models.py

The script which runs entire app is:
main.py

Please note that sharing project files is strictly prohibited. This project is intended solely for educational and non-commercial purposes.