# Data-Science-Portfolio
This is a  repository of the projects I worked on or currently working on. All the current projects are written in Python (Jupyter Notebook). Click on the projects to see full analysis and code.

Please contact me on [Linkedin](https://www.linkedin.com/in/yamil-velazquez92/) if you are looking to hire a data scientist / data analyst.

## Projects:

###  [CIFAR Image Classification using CNN and Keras](https://github.com/Lekamaster/Data-Science-Portfolio/tree/main/CNN_CIFAR)
<b>Objective</b>: 

The objective of this project was to build and train a Convolutional Neural Network (CNN) to classify images from the CIFAR dataset, and to evaluate its performance in terms of accuracy and learning behavior.

<b>Key Steps</b>

<b>*Data Collection</b>

Imported the required libraries and loaded the CIFAR image dataset, which consists of small RGB images belonging to multiple object classes.

<b>*Data Preprocessing</b>

- Normalized image pixel values to improve model convergence

- Prepared labels in the appropriate format for multi-class classification

- Split the data into training and test sets

<b>*Model Definition</b>

- Designed a Convolutional Neural Network (CNN) architecture composed of:

- Convolutional layers with ReLU activation

- MaxPooling layers for spatial dimensionality reduction

- Fully connected (Dense) layers

- Dropout layers to reduce overfitting

- Softmax output layer for multi-class classification

<b>*Model Compilation</b>

- Loss function: Categorical Crossentropy

- Optimizer: Adam

- Metric: Accuracy

- A model summary was generated to inspect the architecture and parameter count.

<b>*Model Training</b>

Trained the CNN model using the training dataset over a defined number of epochs, monitoring both training and validation performance.

<b>*Evaluation and Analysis</b>

- Evaluated the model using the test dataset

- Visualized training and validation accuracy and loss curves

- Analyzed the model’s ability to learn relevant visual features from CIFAR images



###  [Fifa 2022 Tweets Sentiment Analysis](https://github.com/Lekamaster/Data-Science-Portfolio/tree/main/FIFA-Sentiment-Analysis)
<b>Objective</b>: 

The goal of this analysis was to understand the sentiment surrounding FIFA 2022 first day.

<b>Key Steps</b>:

<b>*Data Collection</b>: 

Import library and load dataset.

<b>*Data Cleaning and Preprocessing</b>: 

Handled missing values, removed duplicates, and applied text cleaning techniques to enhance the quality of the dataset.

<b>*Exploratory Data Analysis (EDA)</b>:

Visualized sentiment distribution to gain insights into the overall mood of Twitter users regarding FIFA 2022.
Explored the most frequently mentioned words and visualized in a WordCloud.

<b>*Time Series Analysis</b>:

Examined the temporal trends of sentiment to identify any significant patterns or spikes during specific hours.

<b>*Modeling</b>:

Utilized Machine Learning techniques to perform sentiment prediction on the tweet text.

-Naive Bayes

-Vader

-Xgboost

-Random Forest

