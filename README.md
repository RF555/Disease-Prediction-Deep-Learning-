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

|                                      | Softmax         | Neural Network  |
|--------------------------------------|-----------------|-----------------|
| Loss (epoch 10)                      | 4.7218          | 1.4645          |
| Loss (epoch 20)                      | 2.6952          | 0.0251          |
| Loss (epoch 30)                      | 1.0956          | 0.0037          |
| Loss (epoch 40)                      | 3.2880          | 0.0025          |
| Loss (epoch 50)                      | 3.0375          | 0.0019          |
| Loss (epoch 60)                      | 2.2564          | 0.0005          |
| Loss (epoch 70)                      | 1.4565          | 0.0002          |
| Loss (epoch 80)                      | 2.9494          | 0.0005          |
| Loss (epoch 90)                      | 1.2315          | 0.0001          |
| Accuracy on the<br/> ***test set***  | 88.01007556675% | 99.94962216624% |
| Accuracy on the <br/>***train set*** | 88.47833389318% | 100.0%          |

## Neural Network

* The loss decreases significantly over epochs, which indicates that the Neural Network learns effectively from the
  data.
* The accuracy of the Neural Network on the ***test set*** is near-perfect indicating its ability to generalize well to
  unseen
  data.
* The perfect accuracy on the ***train set*** suggests that the model may be overfitting to the train data, as the
  model
  may have memorized the train data rather than learning generalizable patterns.

## SoftMax

* The loss decreases gradually over epochs, indicating that the model is learning from the train data.

# Improvements

* To address overfitting in the Neural Network, regularization techniques can be applied.
* Hyperparameter tuning, such as adjusting the learning rate or batch size, may further optimize the performance of both
  models.
