from math import nan, inf
from locale import normalize
from turtle import forward
import torch
import argparse
from typing import Union
class ObjDistribution(torch.nn.Module):
    def setup(self):
        return True
    def get_pdf(self):
        return True

class GEV(ObjDistribution):
    topy:torch.Tensor
    lr:float
    def __init__(self,args) -> None:
        super().__init__()
        # self.params = torch.tensor(args.gev_params,dtype=torch.float,device=args.device)
        # assert len(self.params)==3
        self.xi = torch.tensor(args.gev_params[0],dtype=torch.float,device=args.device, requires_grad=True)
        self.mu = torch.tensor(args.gev_params[1],dtype=torch.float,device=args.device, requires_grad=True)
        self.sigma = torch.tensor(args.gev_params[2],dtype=torch.float,device=args.device, requires_grad=True)
        self.trainer = torch.optim.SGD([self.xi, self.mu, self.sigma],lr=args.gev_lr)
        # self.trainer = torch.optim.SGD([self.xi],lr=args.gev_lr)
        self.init_count = 0
        # self.topy = args.topy
        self.lr = args.gev_lr

    def review(self):
        # if self.xi >0.:
        #     self.xi = torch.tensor(0.).to(self.xi)
        if self.sigma <0.:
            self.sigma = torch.tensor(0.01).to(self.sigma)
    def pdf(self, y_normed:torch.Tensor):
        if self.xi !=0.:
            return (1+self.xi*y_normed)**(-1/self.xi-1)*torch.exp(-(1+self.xi*y_normed)**(-1/self.xi))
        else:
            return torch.exp(-y_normed)*torch.exp(-torch.exp(-y_normed))
    def logpdf(self, y_normed:torch.Tensor):
        if self.xi !=0.:
            return (-1/self.xi-1)*torch.log((1+self.xi*y_normed))+(-(1+self.xi*y_normed)**(-1/self.xi))
        else:
            return (-y_normed)+(-torch.exp(-y_normed))

    def cdf(self, y_normed:torch.Tensor):
        if self.xi !=0.:
            return torch.exp(-(1+self.xi*y_normed)**(-1/self.xi))
        else:
            return torch.exp(-torch.exp(-y_normed))
    def logcdf(self, y_normed:torch.Tensor):
        if self.xi !=0.:
            return (-(1+self.xi*y_normed)**(-1/self.xi))
        else:
            return (-torch.exp(-y_normed))
    def normalize(self, y:torch.Tensor):
        return (y-self.mu)/self.sigma
    def nll(self, topy):
        loss = self.logpdf(topy[0])
        for i in range(1,topy.shape[0]):
            loss += self.logpdf(topy[i])
        return -loss
    def forward(self, y):
        y_normed = self.normalize(y)
        likelihood = self.pdf(y_normed)
        if torch.isnan(likelihood) or torch.isinf(likelihood):
            likelihood = self.pdf(torch.tensor(1.,device=y.device, dtype=torch.float))
        return torch.max(torch.tensor(1e-4,device=y.device),likelihood)
    def nllloss(self, y):
        return self.nll(y)
    def init_params(self, topy, lr):
        if self.init_count ==0:
            self.init_count = 1
            self.sigma = topy[0] - topy[-1]
            self.sigma = self.sigma.clone().detach().requires_grad_(True)
            if self.sigma==0.:
                self.sigma = torch.tensor(1.,requires_grad=True,device=topy.device)
            self.mu = topy[-1].clone().detach()
            self.mu.requires_grad = True
            self.xi = torch.tensor(-.1,requires_grad=True,device=topy.device)
            print(self.sigma)
            self.trainer = torch.optim.SGD([self.xi, self.mu, self.sigma],lr=lr)
            self.topy = topy.clone().detach()
            self.lr=lr
        
    def reset_params(self):
        print('reset gev params')
        topy = self.topy
        self.sigma = topy[0] - topy[-1]
        self.sigma = self.sigma.clone().detach().requires_grad_(True)
        if self.sigma==0.:
            self.sigma = torch.tensor(1.,requires_grad=True,device=topy.device)
        self.mu = topy[-1].clone().detach()
        self.mu.requires_grad = True
        self.xi = torch.tensor(-.1,requires_grad=True,device=topy.device)
        self.trainer = torch.optim.SGD([self.xi, self.mu, self.sigma],lr=self.lr)

    def train_epochs(self, topy:Union[torch.Tensor,None]=None, epochs=100):
        # topy = torch.cat([topy, y[idx[-1]].view(1)])\
        print('gev train started')
        if topy is not None:
            dist = (topy.view(self.topy.shape)-self.topy)**2
            if dist.sum()==0.:
                print('no new topy value, return')
                return
            else:
                print('new topy value found, train gev')
                self.topy = topy.clone().detach()
        # self.topy.requires_grad=True
        topy_normed = self.normalize(self.topy)
        loss_old = self.nllloss(topy_normed)
        # print(topy_normed, self.topy)
        for i in range(epochs):
            topy_normed = (self.topy-self.mu)/self.sigma
            self.trainer.zero_grad()
            loss = self.nllloss(topy_normed)
            if torch.isnan(loss) or torch.isinf(loss):
                print('gev loss:%.3f, old loss:%.3f, xi:%.3f, mu:%.3f, sigma:%.3f'%(loss.item(),loss_old.item(), self.xi.item(),self.mu.item(),self.sigma.item()))
                print(f'loss is {loss.item()}, reset params')
                self.reset_params()
                topy_normed = self.normalize(self.topy)
                loss = self.nllloss(topy_normed)
                print('gev loss:%.3f, old loss:%.3f, xi:%.3f, mu:%.3f, sigma:%.3f'%(loss.item(),loss_old.item(), self.xi.item(),self.mu.item(),self.sigma.item()))
            assert not (torch.isnan(loss) or torch.isinf(loss)), print('gev loss:%.3f, old loss:%.3f, xi:%.3f, mu:%.3f, sigma:%.3f'%(loss.item(),loss_old.item(), self.xi.item(),self.mu.item(),self.sigma.item()))
            if torch.abs(self.xi)>=0.5:
                self.xi.requires_grad = False
            if torch.abs(self.sigma)>=10:
                self.sigma.requires_grad = False
            
            # Early termination
            if 1- loss/loss_old <1e-5 and i !=0:
                break
            else:
                loss.backward()
                self.trainer.step()
            loss_old = loss
        print('gev loss:%.3f, old loss:%.3f, xi:%.3f, mu:%.3f, sigma:%.3f'%(loss.item(),loss_old.item(), self.xi.item(),self.mu.item(),self.sigma.item()))
        print('gev train finished')
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    device = torch.device('cuda:3')
    args.device = device
    args.gev_lr = 1e-3
    args.gev_params = [-0.1, 0., 1.]
    args.topy=30
    args.p = GEV(args)
    y = torch.tensor([5.4380, 4.7098, 4.6977, 4.5172, 4.4664, 4.3508, 4.2026, 4.2011, 4.1915,\
        4.0020, 3.9249, 3.8917, 3.8895, 3.8861, 3.8049, 3.7814, 3.7453, 3.6788,\
        3.5361, 3.5152, 3.5021, 3.4525, 3.4450, 3.3692, 3.3555, 3.3492, 3.3257,\
        3.2955, 3.2954, 3.2606], device=args.device)
    args.p.init_params(y, args.gev_lr)
    args.p.train_epochs(epochs=1)
