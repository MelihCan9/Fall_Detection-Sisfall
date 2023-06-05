# Fall Detection with SisFall Dataset

This project aims to detect falls using the SisFall dataset, which consists of simulated fall events. The dataset contains accelerometer data obtained from tri-axial accelerometers. This project implements SVM, XGBoost, and NN (Neural Network) models for fall detection and using the SisFall dataset. Each model is trained on the dataset with optimum features and hyperparameters and then evaluated on a test dataset to assess its performance.

## Dataset: 
### SisFall: A Fall and Movement Dataset
SisFall dataset: A Fall and Movement Dataset. Created by: A. Sucerquia, J.D. López, J.F. Vargas-Bonilla SISTEMIC, Faculty of Engineering, Universidad de Antiquia UDEA. Detailed information about this dataset can be found on this website: http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/. Reference paper: Sucerquia A, López JD, Vargas-Bonilla JF. SisFall: A Fall and Movement Dataset. Sensors (Basel). 2017;17(1):198. Published 2017 Jan 20. doi:10.3390/s17010198.

PS: If you are having a problem accessing a dataset via its original link, you can get the raw data from: https://github.com/JiayangLai/SisFallDatasetAnnotation (SisFall_dataset_csv).


## Data Preprocessing

The project uses the Python programming language and libraries such as pandas, numpy and os to process the dataset. The data preprocessing steps involve the following:

1. Reading the raw data files in the folder where the dataset is stored.
2. Downsampling the data and calculating various properties of the accelerometers(ADXL345).
3. Computing statistical features (minimum, maximum, mean, etc.) and derived features.
4. Splitting the data into training and testing sets.
5. Standardizing the data.

## Contributing

If you would like to contribute to this project, you can fork the GitHub repository and make your changes. You can share your modifications through pull requests.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or suggestions, you can contact me at [mchamurcu@hotmail.com]


