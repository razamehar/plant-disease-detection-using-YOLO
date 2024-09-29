# Plant Disease Detection Using YOLO
This project aims to develop a robust plant disease detection system using advanced machine learning techniques, primarily leveraging YOLO (You Only Look Once) for object detection. The workflow includes data preprocessing, feature extraction, non-negative matrix factorization (NMF), fuzzy clustering, and model training.

## Project Overview
This project focuses on the detection of plant diseases through image analysis. By utilizing machine learning algorithms, we can identify and classify diseases present in various plants based on their visual characteristics.

## Data Preparation
The dataset used for this project is PlantDoc, which contains images of healthy and diseased plants. Images are categorized into different classes based on the type of plant and the specific disease.

### Steps:
- Data Resizing: All images are resized to a uniform dimension suitable for model training.
- Normalization: Image pixel values are normalized to a range suitable for deep learning models.

## Feature Extraction
We utilize a pretrained VGG16 model to extract features from the images, enhancing the model's ability to recognize patterns related to plant diseases. The extracted features are stored in a structured format for subsequent analysis.

### Process:
- Load the VGG16 model with pretrained weights.
- Extract features from each image in the dataset and flatten the resulting arrays for further processing.

## Non-Negative Matrix Factorization (NMF)
NMF is employed to decompose the feature set into parts that represent the hidden patterns within the images, helping in identifying shared features such as shapes and colors.

### Key Points:
- An optimal number of components is determined through reconstruction errors.
- Components are analyzed and interpreted to derive meaningful insights into the underlying data.

### Observations:
- Component Analysis: Each component reveals distinct characteristics associated with different plant diseases.
- Action Items: Recommendations are made based on the features identified, such as ensuring dataset balance and augmenting underrepresented classes.

## Fuzzy Clustering
Fuzzy clustering techniques are applied to find optimal groupings within the dataset.

### Metrics for Optimal Clusters:
- Fuzzy Partition Coefficient (FPC)
- Partition Entropy (PE)
- Xie-Beni Index

These metrics suggest the most suitable number of clusters for our dataset, leading to refined classifications based on membership values.

### Outcomes:
- Each cluster corresponds to a specific focus area (e.g., plant disease focus, plant morphology).
- Further action is suggested based on cluster characteristics.

## Data Augmentation
To improve model robustness, data augmentation techniques are employed. This includes various transformations such as flipping, rotating, cropping, and brightness adjustments to enrich the dataset.

### Procedure:
- Define augmentation sequences to create diverse training examples.
- Apply augmentations while ensuring the integrity of bounding box annotations.

## Modeling
The YOLO model is initialized and trained on the processed dataset.

### Configuration:
- Update data.yaml to specify training, validation, and test paths.
- Remove underrepresented classes to enhance model training.

### Training Parameters:
- Epochs: 100
- Batch Size: 16

## Results
Upon training, the model is evaluated using key metrics:

### Mean Average Precision (mAP) at IoU thresholds:
- mAP50: 0.471
- mAP50-95: 0.363

## Conclusions
This project demonstrates the effectiveness of leveraging deep learning models for plant disease detection. Through careful data preparation, feature extraction, and model training, we can build a system capable of accurately identifying plant diseases, which can significantly aid agricultural practices.

## License
This project is licensed under the Raza Mehar License. See the LICENSE.md file for details.

## Contact
For any questions or clarifications, please contact Raza Mehar at [raza.mehar@gmail.com], Pujan Thapa at [iampujan@outlook.com] or Syed Najam Mehdi at [najam.electrical.ned@gmail.com].