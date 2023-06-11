# Multi-Class Land Use Classification

🌍🔍 This project focuses on land use classification using a multi-class classification approach. The dataset used for training and evaluation is the UCMerced Land Use dataset, which contains satellite images of 21 different land use categories, including agricultural areas, airports, forests, golf courses, and more.

## Dataset

📦🖼️ The UCMerced Land Use dataset consists of satellite images divided into three sets: training, validation, and test. Each set contains images from 21 different land use categories. The dataset is organized in a directory structure with separate folders for each category within each set.

## Data Preparation

🔧🔍 The images are loaded and preprocessed using the TensorFlow `ImageDataGenerator` class. Data augmentation techniques such as rotation, shifting, zooming, and flipping are applied to the training data to increase its variability. The images are resized to 256x256 pixels and normalized to values between 0 and 1. The data is divided into batches and prepared as `tf.data.Dataset` objects for efficient processing during training.

## Model Architecture

🏗️🧱 Two models are explored in this project: a CNN model and a transfer learning model using the MobileNet architecture.

### CNN Model

📐🧱 The CNN model architecture consists of several convolutional and max pooling layers, followed by fully connected layers. ReLU activation is used after each convolutional layer, and dropout regularization is applied to reduce overfitting. The final layer uses softmax activation for multi-class classification. The model is trained using the Adam optimizer and categorical cross-entropy loss.

### Transfer Learning Model

🔄🧠 The transfer learning model utilizes the MobileNet architecture, which is pretrained on the ImageNet dataset. The MobileNet base model is loaded without the top layers, and new layers specific to the land use classification task are added. The base model is frozen, and only the new layers are trained. The model is compiled with the Adam optimizer and trained on the dataset.

## Model Evaluation

📈📊 Both models are trained and evaluated on the training, validation, and test datasets. The training and validation accuracies, losses, and evaluation metrics such as test accuracy are recorded. The performance of the models is visualized using accuracy and loss curves over epochs. Additionally, a confusion matrix and classification report are generated to assess the models' performance on individual land use classes.

## Results

✅🔍 The CNN model and the transfer learning model both achieve good accuracy on the land use classification task. The models demonstrate the ability to differentiate between different land use categories based on satellite images. The transfer learning model, leveraging the pre-trained MobileNet architecture, achieves comparable performance to the CNN model.

## Usage

⚙️🔬 To replicate the results of this project, follow these steps:

1. Download the UCMerced Land Use dataset from the [dataset link](https://polimi365-my.sharepoint.com/:u:/g/personal/10104160_polimi_it/EWA1ekjfRepPt5P9cpqjdycBbFFUOtlcLG8yyasZ8sFVjA?e=8cabgP).
2. Organize the dataset into the directory structure as described in the project.
3. Set up the required libraries and dependencies (TensorFlow, NumPy, Matplotlib, etc.).
4. Execute the provided code to load and preprocess the dataset.
5. Train the CNN model and/or the transfer learning model using the provided code.
6. Evaluate the models using the test dataset and analyze the results.
7. Modify the models or experiment with different hyperparameters to improve performance if desired.

## Conclusion

🌱📊 Land use classification is an important task with numerous applications in urban planning, environmental monitoring, and more. This project demonstrates the effectiveness of CNN models and transfer learning for land use classification based on satellite images. The models achieve good accuracy and provide insights into the classification process.
