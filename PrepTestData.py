import torch
# import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import one_hot

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''


# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__
class GetDataset(Dataset):

    def __init__(self, X_, Y_):
        # Initialize data, download, etc.
        # read with numpy or pandas
        # XY_data = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.num_samples = X_.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.tensor(X_, dtype=torch.float32)  # size [num_samples, num_features]
        self.y_data = torch.tensor(Y_, dtype=torch.float32)  # size [num_samples, num_classes]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.num_samples


def getTrainAndTest():
    whole_data = pd.read_csv('Data/Disease Prediction Dataset.csv')
    # here the LAST column is the class label, the rest are the features
    # Split data into features and target variable
    XX = whole_data.drop(columns=["prognosis"])
    yy = pd.get_dummies(whole_data["prognosis"])  # (onehot encoding the prognosis)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.4, random_state=42)
    # Standardize features (optional but recommended)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # X_train_numpy = X_train.to_numpy()
    # X_test_numpy = X_test.to_numpy()
    y_train_numpy = y_train.to_numpy()
    y_test_numpy = y_test.to_numpy()
    # return GetDataset(X_train_numpy, y_train_numpy), GetDataset(X_test_numpy, y_test_numpy)
    return GetDataset(X_train, y_train_numpy), GetDataset(X_test, y_test_numpy)


def trainModel(data_loader, model, criterion, optimizer, num_epochs, print_rate):
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            # Forward pass and loss
            y_predicted = model(inputs)
            loss = criterion(y_predicted, labels)

            # Backward pass and update
            loss.backward()
            optimizer.step()

            # zero grad before new step
            optimizer.zero_grad()

        if (epoch + 1) % print_rate == 0:
            print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

    print('Finished Training\n\n')


def accuracyTest(model, data_loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for inputs_, labels_ in data_loader:
            outputs = model(inputs_)

            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            predicted = one_hot(predicted, num_classes=41)
            # print(f'predicted.size(): {predicted.size()}')
            # print(f'labels.size(): {labels_.size()}\n')

            n_samples += labels_.size(0)

            for i_ in range(predicted.size(0)):
                label = labels_[i_]
                pred = predicted[i_]
                # print("label.size():\t", label.size())
                # print("pred.size(): \t", pred.size())
                # print("label: ", label)
                # print("pred: ", pred)
                # label_i = torch.argmax(label)
                # pred_i = torch.argmax(pred)
                # print(f'label_{i}:  {label_i}')
                # print(f'pred_{i}:  {pred_i}')
                if label.size() == pred.size():
                    equals = True
                    for j_label, j_pred in zip(label, pred):
                        if j_label != j_pred:
                            equals = False
                            # print(f'break!!\n\n')
                            break
                    if equals:
                        n_correct += 1

            # print(f'n_samples: {n_samples}')
            # print(f"n_correct: {n_correct}\n")

        acc = 100.0 * n_correct / n_samples
        return acc
