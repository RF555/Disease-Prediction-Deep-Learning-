# Disease Prediction - A Deep Learning Project in PyTorch

# The Dataset

In this project we will be using the dataset of 4962 samples with 132 features ( each gets a prognosis of one of 41
diseases The dataset was obtained
from [https://www.kaggle.com](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning/data).

# Description of the project

In this project, I aim to develop a classification model using a Neural Network. The goal is to predict one of 41
disease s based on a set of 132 symptoms obtained from my dataset. The project will involve data preprocessing, model
training, evaluation, and comparison between Softmax Regression and a Neural Network with 3 hidden layers model.

# Model

For the implementation, I will use PyTorch to build a Neural Network model capable of capturing complex relationships
within the data. Additionally, I will implement simpler Softmax regression.

## Initial Attempts

The main improvements I found myself correcting during the building of the Neural Network:

* Use one hot encoding s o th at the label matrix of our data will be in the shape of (4962, 41).
* Trying different depths to the network and sizes of the hidden layers
* Trying different optimizers (ended up with Adam).

# Comparison

## Neural Network
•	The loss decreases significantly over epochs, which indicates that the Neural Network learns effectively from the data.
•	The accuracy of the Neural Network on the test set is near-perfect indicating its ability to generalize well to unseen data.
•	The perfect accuracy on the training set suggests that the model may be overfitting to the training data, as the model may have memorized the training data rather than learning generalizable patterns.
## SoftMax
•	The loss decreases gradually over epochs, indicating that the model is learning from the training data.
# Improvements
•	To address overfitting in the Neural Network, regularization techniques can be applied.
•	Hyperparameter tuning, such as adjusting the learning rate or batch size, may further optimize the performance of both models.
