import numpy as np
import pickle
import torch as t
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
#from torch.optim.lr_scheduler import MultiplicativeLR
import matplotlib.pyplot as plt
from itertools import chain
#from scipy.stats import pearsonr
#import seaborn as sns

batch_factor = 2
batch_size = 32 * batch_factor
lr = 5e-4 * batch_factor
b1, b2 = 0.7, 0.999
weight_decay = 1.6e-3 #0.0012
n_epochs = 100
train_frac = 0.75
conv_channels = 4
board_size = (18, 11)
drop_top_n = 4
drop_bottom_n = 1

grade_dict = {'6A':1, '6A+':2, '6B':3, '6B+':4, '6C':5, '6C+':6, '7A':7, '7A+':8, '7B':9, '7B+':10, '7C':11, '7C+':12, '8A':13, '8A+':14, '8B':15, '8B+':16}

# Convert grade n from ints to 1D tensors with ones in the first n positions and zeros elsewhere.
# Ex. 3 -> [1,1,1,0,0,...]
# This allows for ordinal regression in the loss function
# i.e. loss is greater for predictions further from the true grade
grade_dict_vec = {}
for grade in grade_dict:
    vec = t.zeros(len(grade_dict) - drop_bottom_n - drop_top_n)
    vec[0:grade_dict[grade] - drop_bottom_n] = 1
    grade_dict_vec[grade] = vec

num_grades = len(grade_dict) - drop_bottom_n - drop_top_n

def prediction2grade(prediction):
    #Convert ordinal predictions to grades
    return (prediction > 0.5).cumprod(axis=1).sum(axis=1).tolist()

def chain_lists(lists) -> np.ndarray:
    return np.array(list(chain(*lists)))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    #p represents the data, q represents the model predictions
    p, q = p[q != 0], q[q != 0]
    return np.sum(p * np.log2(p / q))

class Data(Dataset):
    def __init__(self):
    
        with open('problems1.pkl', 'rb') as f:
            data1 = pickle.load(f)

        with open('problems2.pkl', 'rb') as f:
            data2 = pickle.load(f)

        data = {**data1, **data2}

        self.names = data.keys()
        self.grades, self.start_holds, self.mid_holds, self.end_holds, self.all_holds = [], [], [], [], []
        for name in self.names:
            problem = data[name]
            if problem[0] in grade_dict.keys():
                if grade_dict[problem[0]] in (max(grade_dict.values()) - np.array(range(0, drop_top_n))):
                    continue
                if grade_dict[problem[0]] in range(1, drop_bottom_n + 1):
                    continue
                self.grades.append(problem[0])
                self.start_holds.append(problem[1])
                self.mid_holds.append(problem[2])
                self.end_holds.append(problem[3])
                self.all_holds.append(problem[4])

        self.all_holds_split_channels = t.Tensor([[self.start_holds[i], self.mid_holds[i], self.end_holds[i]] for i in range(len(self.start_holds))])

        self.start_holds = t.Tensor(self.start_holds)
        self.mid_holds = t.Tensor(self.mid_holds)
        self.end_holds = t.Tensor(self.end_holds)
        self.all_holds = t.Tensor(self.all_holds)

        self.all_holds_neg_ends = self.mid_holds - self.start_holds - self.end_holds

        self.grades_numeric = [grade_dict[grade] for grade in self.grades]
        self.grades = [grade_dict_vec[grade] for grade in self.grades]
                
        self.len = len(self.grades)
                
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.all_holds_split_channels[index], self.grades[index])


# Random split of data
dataset = Data()

# sns.violinplot(x=dataset.grades_numeric)
# plt.show()
# assert 0

# grades = [int(grade.sum().item()) for grade in dataset.grades]
# hist = plt.hist(grades, bins=np.array(range(min(grades), max(grades)+2))-0.5, rwidth=0.9)
# plt.show()

train_len = int(train_frac * dataset.len)
valid_len = dataset.len - train_len

train_data, valid_data = random_split(dataset, (train_len, valid_len))

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Input size: (batch_size, 3, 18, 11)
        self.b1 = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5)
                )
        # (batch_size, 32, 9, 6)
        self.b2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5)
                )
        # (batch_size, 64, 9, 6)
        self.b3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5)
                )
        # (batch_size, 128, 5, 3)
        self.b4 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5)
                )
        # (batch_size, 256, 5, 3)
        # Flatten -> (batch_size, 256*5*3)
        self.b5 = nn.Sequential(
                    nn.Linear(256*5*3, num_grades),
                    nn.Sigmoid()
                )
        # (batch_size, num_grades)
        

    def forward(self, problem):
        inter = self.b1(problem)
        inter = self.b2(inter)
        inter = self.b3(inter)
        inter = self.b4(inter)
        return self.b5(inter.flatten(start_dim=1))
        

def ordinal_regression_loss(prediction, target):
    return t.pow(nn.MSELoss(reduction='none')(prediction, target).sum(axis=1), 2).mean()

model = Model()

#Stochastic gradient descent optimizer
optimizer = Adam(model.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)

#Change the learning rate dyamically
# lmbda = lambda batch: 1.018
# scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

def evaluate():
    valid_loss, predictions, actual = [], [], []

    #Training mode
    model.eval()
    for data in valid_loader:

        #Split batch into input (boulder problem) and targets (grade)
        problem = data[0]
        grade = data[1]

        #Forward Propagation
        prediction = model(problem)

        #Loss computation
        loss = ordinal_regression_loss(prediction, grade)

        valid_loss.append(loss.item())
        predictions.append(prediction2grade(prediction))
        actual.append(prediction2grade(grade))

    return valid_loss, chain_lists(predictions), chain_lists(actual)

def train():
    train_loss, predictions, actual = [], [], []
    #lr = []

    #Training mode
    model.train()
    for data in train_loader:

        #Split batch into input (boulder problem) and targets (grade)
        problem = data[0]
        grade = data[1]

        #Zero the gradients
        optimizer.zero_grad()

        #Forward Propagation
        prediction = model(problem)

        #Loss computation
        loss = ordinal_regression_loss(prediction, grade)

        #Backpropagation
        loss.backward()

        #Weight optimization
        optimizer.step()

        train_loss.append(loss.item())
        predictions.append(prediction2grade(prediction))
        actual.append(prediction2grade(grade))

        #lr.append(scheduler.get_last_lr())
        #scheduler.step()


    return train_loss, chain_lists(predictions), chain_lists(actual)


#Training and validation loop
for epoch in range(n_epochs): #An epoch is a run of the entire training dataset
    train_loss, predictions_train, actual_train = train()

    # plt.scatter(lr, train_loss)
    # plt.xscale('log')
    # plt.xlim(1e-4, 2e-3)
    # plt.show()

    valid_loss, predictions_valid, actual_valid = evaluate()

    grade_diff_valid = predictions_valid - actual_valid
    grade_diff_train = predictions_train - actual_train

    bins = np.array(list(range(1, 18))) - 0.5
    p = np.histogram(actual_valid, bins, density=True)[0]
    q = np.histogram(predictions_valid, bins, density=True)[0]

    print('Epoch: %i' % epoch)
    print('  Std of training error distribution:    %.3f' % np.std(grade_diff_train))
    print('  Std of validation error distribution:  %.3f' % np.std(grade_diff_valid))
    print('  Mean training loss:                    %.3f' % np.mean(train_loss))
    print('  Mean validation loss:                  %.3f' % np.mean(valid_loss))
    print('  Validation KL divergence:              %.3f' % kl_divergence(p, q))
    n_correct = len(predictions_valid[predictions_valid == actual_valid])
    print('  Correct predictions:                   %i' % n_correct)
    proportion_correct = n_correct/len(predictions_valid)
    print('  Proportion correct:                    %.3f' % proportion_correct)

    # bins = np.array(range(min(grade_diff), max(grade_diff)+2)) - 0.5
    # plt.hist(grade_diff, align='mid', rwidth=.9, bins=bins)
    # plt.title('epoch: %d' % (epoch+1))
    # plt.show()

actual_errors = actual_valid[actual_valid != predictions_valid]
predictions_errors = predictions_valid[actual_valid != predictions_valid]


print('Number of problems: %i' % len(actual_valid))
print('Number of incorrect predictions: %i' % len(actual_errors))
print('Number of correct predictions: %i' % (len(actual_valid) - len(actual_errors)))
#corr, _ = pearsonr(actual_valid, predictions_valid)
#print('Correlation coefficient: %.3f' % corr)

bins = np.array(range(min(grade_diff_valid), max(grade_diff_valid)+2)) - 0.5
plt.hist(grade_diff_valid, rwidth=.9, bins=bins, density=True)
plt.title('Difference in actual and predicted grades')
plt.show()

bins = np.array(range(min(actual_errors), max(actual_errors) + 2)) - 0.5
plt.hist2d(actual_errors, 
           predictions_errors,
           bins=bins,
           cmin=0.00001,
           density=True,
           zorder=1)
ticks = np.array(range(min(actual_errors), max(actual_errors)+2))
plt.xticks(ticks)
plt.yticks(ticks)
plt.xlabel('Actual Grade (1=6A, 16=8B+)')
plt.ylabel('Predicted Grade (1=6A, 16=8B+)')
plt.grid(alpha=0.2, zorder=0)
plt.colorbar(label='Proportion of errors')
plt.gca().invert_yaxis()
plt.show()