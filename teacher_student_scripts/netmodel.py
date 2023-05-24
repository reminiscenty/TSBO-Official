import torch
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt

def load_array(data_arrays, batch_size, weights=None):  # @save
    dataset = data.TensorDataset(*data_arrays)
    if weights is not None:
        sampler = data.WeightedRandomSampler(weights, len(dataset))
        return data.DataLoader(dataset, batch_size, sampler=sampler)
    else:
        return data.DataLoader(dataset, batch_size)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.0001)
        nn.init.normal_(m.bias, std=0.0001)


class Accumulator:  # @save

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




class TeacherModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.net = nn.Sequential()
    def print_weights(self):
        self.apply(print_weights)
    def forward(self,x):
        return self.net(x)
    
    def pred_determinstic(self,x):
        return self.net(x)

    def evaluate_loss(self, data_iter):  #@save
        self.eval()  
        metric = Accumulator(2)  
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(self.compute_loss(self(X), y), y.numel())
        return metric[0] / metric[1]

    def compute_loss(self, input, target):
        return self.loss(input, target.view(input.shape))

    def train_model(self, train_iter, test_iter, lr, epochs, is_figure=False):
        self.train()
        train_loss_list, test_loss_list = [], []
    #     teacher_model.net.apply(print_weights)
        teacher_trainer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=.3)
        for epoch in range(epochs):
            # teacher model
            # teacher_trainer
    #         teacher_model.net.train()
            for x, y in train_iter:
                y_hat = self(x)
                loss = self.compute_loss(y_hat, y)
                teacher_trainer.zero_grad()
                loss.backward()
                teacher_trainer.step()
            if epoch %1000==0:
                train_loss = self.evaluate_loss(train_iter)
                train_loss_list.append(train_loss)
    #         teacher_model.net.eval()
                test_loss = self.evaluate_loss(test_iter)
                test_loss_list.append(test_loss)

                print(f'epoch: {epoch}, train teacher_loss: {train_loss}, test teacher_loss:{test_loss}')
        if is_figure:
            fig = plt.figure(figsize=(6, 8))
            plt.plot(train_loss_list, 'b')
            plt.plot(test_loss_list, 'r')
            plt.legend(['train teacher_loss', 'test teacher_loss'])
            plt.title(f'learning rate={lr}')
            return fig


    def train_epoch(self, train_iter, updater):  # @save
        self.train()
        metric = Accumulator(1)
        for X, y in train_iter:
            y_hat = self(X)
            l = self.compute_loss(y_hat, y)
            # print(l)
            updater.zero_grad()
            l.sum().backward()
            updater.step()
            metric.add(float(l.sum()))
        return metric[0]


class MLPModel(TeacherModel):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        m = nn.Sequential(
            # nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.net = m
        self.loss = torch.nn.MSELoss()


class VMLPModel(TeacherModel):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim=latent_dim        
        self.net = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.Linear(32, 2)
        )
        self.loss = torch.nn.GaussianNLLLoss()
        self.softplus = torch.nn.Softplus()
        self.varmin = 1e-6
    def compute_loss(self, input, target):
        if input.ndim==1:
            input = input.reshape(-1,2)
        return self.loss(input[:, 0], target.reshape(-1), self.decode_variance(input[:, 1]))
    def decode_variance(self, input):
        return self.softplus(input)+self.varmin
    def forward(self, x):
        y_pred = self.net(x)
        if y_pred.ndim==1:
            return y_pred.reshape(1,-1)
        else:
            return y_pred
        
    def pred_determinstic(self, x:torch.Tensor):
        y_pred = self(x)
        if y_pred.ndim==1:
            return y_pred[0].reshape(-1,1)
        else:
            return y_pred[:,0].reshape(-1, 1)

def train_epoch_ts_model(train_iter, teacher_model, teacher_trainer,
                         student_model, student_likelihood, student_trainer, n_unlabeled, ):
    if isinstance(teacher_model, torch.nn.Module):
        teacher_model.train()
    if isinstance(student_model, torch.nn.Module):
        student_model.train()

    for X, Y in train_iter:
        n_labeled = X.shape[0]


def print_weights(m):
    if type(m) == nn.Linear:
        print(m.weight.shape, m.bias.shape)
        print(m.weight, m.weight.grad)
        print(m.bias, m.bias.grad)

class TopologyTeacher(TeacherModel):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        # self.encoder = vae.encoder
        self.latent_dim = latent_dim
        self.net = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.Linear(32, 1)
        )
        # self.loss = torch.nn.BCEWithLogitsLoss()
    # def forward(self,x):
    #     return self.net(self.encoder(x))


class ShapeTeacher(TeacherModel):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        # self.encoder = vae.encoder
        self.latent_dim = latent_dim
        self.net = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
class ExprTeacher(TeacherModel):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        # self.encoder = vae.encoder
        self.latent_dim = latent_dim
        self.net = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
class ChemTeacher(TeacherModel):
    def __init__(self, latent_dim) -> None:
        super().__init__()
        # self.encoder = vae.encoder
        self.latent_dim = latent_dim
        self.net = nn.Sequential(nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
