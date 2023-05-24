
import argparse, gpytorch
import os, sys
from utils import pretrain_teacher_student, train_teacher_student
from netmodel import MLPModel
from gp_models import ExactGPModel
import torch
from pathlib import Path
from math import log
from gev import GEV
import numpy as np
import matplotlib.pyplot as plt

def f(x,x_obj):
    assert torch.is_tensor(x)        
    return torch.exp(-torch.norm(x-x_obj, dim=-1)).view(-1,1)
class ArgsTest:
    a = 0
if __name__=='__main__':
    # ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
    # print(ROOT_PROJECT)
    # sys.path.insert(0, ROOT_PROJECT)
    n = 101

    plt.show()
    device = torch.device(0)
    xLabel = torch.tensor([
        [0.5,0.2],
        [0.2,0.5],
        [0.5,0.5],
        [0.4,0.4],
        [0.3, 0.45],
        [0.45,0.3]],dtype=torch.float,device=device)
    xLabel = torch.rand(100, 2,dtype=torch.float,device=device)*0.5
    # print(xLabel)
    xObj = torch.tensor([0.7,0.7],dtype=torch.float,device=device)
    yObj = f(xObj,xObj)
    yLabel = f(xLabel,xObj)
    print(xLabel.shape, yLabel.shape)
    xBound = torch.tensor([[0.,1],
                            [0,1]],dtype=torch.float,device=device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=2)
    
    args = parser.parse_args()
    args.batch_size = 16
    args.device = device
    args.gaussian_lr = 1e-3
    args.unlabel_logstd_value = 0.1
    args.teacher = MLPModel(args.latent_dim).to(device)
    args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-3)
    # args.gaussian_lr=1e-4
    args.unlabel_logstd_value = log(0.1)
    args.student = ExactGPModel(xLabel,yLabel).to(device)
    args.student_loss = gpytorch.mlls.ExactMarginalLogLikelihood(
                            args.student.likelihood, args.student)
    args.student_optimizer = torch.optim.Adam(args.student.parameters(), lr=0.01)
    # args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-3)
    args.student_initial_train_epochs = 2000
    args.teacher_initial_train_epochs = 2000
    args.initial_train = True
    # args.gevmcmc = True
    args.gev_lr = 1e-3
    args.gev_params = [0.1, 0., 1.]
    args.lambda_0 = 1e-2
    args.lambda_m = 5*1e-2
    args.p = GEV(args)
    args.unlabel_loss = 'gaussian'
    args.joint_train_epochs = 20
    args.noised_unlabel = False
    args.ts_train_count = 0
    args.unlabel_ratio = 10
    # args.topk = 10
    args.weighted = True
    a = 0
    # pretrain_teacher_student(args, xLabel, yLabel, xLabel, yLabel, xBound,a)
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(0, 1, n)
    x2, x1 = np.meshgrid(x1,x2)
    x = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1)),axis=-1)
    y = f(torch.tensor(x,device=device),xObj).view(n,n).cpu().numpy()
    # print(x1.shape,x2.shape,y.shape)
    # print(x1.reshape(-1).reshape(n,n)==x1)
    fig, ax = plt.subplots()

    c = ax.pcolormesh(x1, x2, y, cmap='RdBu', vmin=0, vmax=1)
    ax.set_title('pcolormesh')
    # set the limits of the plot to the limits of the data
    ax.axis([0, 1, 0, 1])
    fig.colorbar(c, ax=ax)
    plt.savefig('a.pdf')

