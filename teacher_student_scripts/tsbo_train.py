
from math import log, sqrt
import random, argparse
import numpy as np
from scipy.fftpack import shift
import torch
from pyDOE import lhs
import gpytorch
from botorch.utils.transforms import normalize, unnormalize
from matplotlib import pyplot as plt
import os
import sys
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import shutil
from typing import Dict, Any, Optional
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from pathlib import Path
sys.path.insert(0,str(Path(os.path.realpath(__file__)).parent))
from gp_models import train_step, ExactGPModel, GPModel
import netmodel 
from gev import GEV

from mcmc import mcmc_different_init




def set_random_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

def add_tsbo_args(parser:ArgumentParser):
    tsbo_group = parser.add_argument_group("tsbo")
    tsbo_group.add_argument(
        "--self_training",
        action='store_true',
        help="Train teacher student model with unlabelled data",
    )
    tsbo_group.add_argument(
        "--unlabel_gauss_var",
        type=float,
        default=0.1,
        help="Train teacher student model with unlabelled data",
    )
    tsbo_group.add_argument(
        "--gaussian_lr",
        type=float,
        default=1e-3,
        help="Learning rate of optimized gaussian for unlabeled data",
    )
    tsbo_group.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of top k function values",
    )
    tsbo_group.add_argument(
        "--x_unlabeled_given",
        action='store_true',
        help="Use unlabelled data in dataset",
    )
    tsbo_group.add_argument(
        "--is_validation",
        action='store_true',
        help="Whether to specify validation method other than top K",
    )
    tsbo_group.add_argument(
        "--validation_type",
        type=str,
        default='random',
        help="How to select validation set",
    )
    tsbo_group.add_argument(
        "--unlabel_sample_type",
        type=str,
        default='gaussian',
        help="gaussian,random,gev",
    )
    tsbo_group.add_argument(
        "--student_fit_lr",
        type=float,
        default=1e-3,
        help="Student's learning rate",
    )
    tsbo_group.add_argument(
        "--unlabel_loss_ratio",
        type=float,
        default=1e-3,
        help="Feedback loss weight",
    )
    tsbo_group.add_argument(
        "--joint_train_epochs",
        type=int,
        default=20,
        help="Bi-level train epoches of the teacher-student model",
    )
    tsbo_group.add_argument(
        "--unlabel_loss",
        type=str,
        default='MSE',
        help="Feedback loss type",
    )
    tsbo_group.add_argument(
        "--ts_version",
        type=int,
        default=1,
        help="1: teacher-student (determinstic) 2: only teacher(deterministic) 3: only teacher(probalistic) 4: teacher-student (probalistic)",
    )
    tsbo_group.add_argument(
        "--additional_noise_std",
        type=float,
        default=0.01,
        help="Additional noises to labels",
    )
    tsbo_group.add_argument(
        "--is_additional_noise",
        action='store_true',
        help="Additional noises to labels",
    )

    return parser


def setup_args(args, task):
    '''
    set up teacher, student, gev
    '''
    device = torch.device('cuda:'+str(args.cuda))
    args.device = device
    args.unlabel_gauss_std = sqrt(args.unlabel_gauss_var)
    if task =='expr':
        args.teacher = netmodel.ExprTeacher(args.latent_dim).to(device)
        args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-3)
    elif task =='topology':
        args.teacher = netmodel.TopologyTeacher(args.latent_dim).to(device)
        args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-4)
    elif task =='chem':
        args.teacher = netmodel.ChemTeacher(args.latent_dim).to(device)
        args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-4)
    if args.ts_version ==3 or args.ts_version==4:
        args.teacher = netmodel.VMLPModel(args.latent_dim).to(device)
        if task =='expr':
            args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-3)
        else:
            args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-4)
    x = torch.tensor(0.,device=device).view(1,1)
    y = torch.tensor(0.,device=device).view(1,1)
    args.student = GPModel(x,y,lr=0.1)
    args.student_initial_train_epochs = 2000
    args.teacher_initial_train_epochs = 2000
    args.initial_train = True
    args.gev_lr = 1e-3
    args.gev_params = [0.1, 0., 1.]
    args.lambda_0 = 1e-2
    args.lambda_m = 5*1e-2
    args.gevtop = GEV(args)
    args.gevbot = GEV(args)
    args.noised_unlabel = False
    args.ts_train_count = 0
    args.unlabel_ratio = 10
    args.weighted = True

    
    args.x_unlabeled_given = False
    if args.ts_version == 4:
        args.student_model_fit = ExactGPModel(x, y,
                                              gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                                                noise=torch.zeros_like(y),learn_additional_noise=True,noise_constraint=gpytorch.constraints.GreaterThan(1e-4))).to(args.device)
    else:
        args.student_model_fit = ExactGPModel(x, y).to(args.device)

    args.student_likelihood_fit = args.student_model_fit.likelihood
    args.student_loss_logp_fit = gpytorch.mlls.ExactMarginalLogLikelihood(
                            args.student_model_fit.likelihood, args.student_model_fit)
    args.student_trainer_fit = torch.optim.Adam(args.student_model_fit.parameters(), lr=args.student_fit_lr)
    args.validation_ratio = 0.3
    args.train_y_noise = None

    if args.is_additional_noise:
        args.additional_noise = np.random.randn(1000)*args.additional_noise_std

def add_tsparams_to_path(args, result_path):
    if args.is_additional_noise:
        result_path +='-noisestd_'+str(args.additional_noise_std)
    if args.self_training:
        result_path +='-self_training'
        result_path += '-'+args.unlabel_sample_type+'_sample-training_epoch-'+str(args.joint_train_epochs)
    else:
        return result_path
    if args.unlabel_sample_type=='gaussian':
        result_path+='-lr_'+str(args.gaussian_lr)+'-var_'+str(args.unlabel_gauss_var)
    result_path+='-topk_'+str(args.topk)
    if args.is_validation:
        result_path+='-valid_'+args.validation_type
    result_path+='-student_fit_lr_'+str(args.student_fit_lr)+'-unlabel_loss_ratio_'+str(args.unlabel_loss_ratio)+'-'+args.unlabel_loss
    result_path+='-ts_version_'+str(args.ts_version)
    return result_path

def gaussian_rsample_warp(x_raw, mean, std):
    return x_raw*std +mean

def pretrain_teacher_student(args, train_x, train_y, x_bounds):
    train_x_normed = normalize(train_x, x_bounds)
    teacher_model: netmodel.TeacherModel = args.teacher
    if args.ts_version==1 or args.ts_version==4:
        
        student_model_fit:ExactGPModel = args.student_model_fit
        student_likelihood_fit = args.student_likelihood_fit
        student_loss_logp_fit = args.student_loss_logp_fit
        student_trainer_fit =  torch.optim.Adam(args.student_model_fit.parameters(), lr=0.01)
        student_model_fit.update_train_data(train_x_normed, torch.flatten(train_y))
        student_model_fit.train()
        student_likelihood_fit.train()
        print(type(student_likelihood_fit))
        for i in range(args.student_initial_train_epochs):
            output = student_model_fit(train_x_normed)
            loss = -student_loss_logp_fit(output, torch.flatten(train_y))
            student_trainer_fit.zero_grad()
            loss.backward()
            student_trainer_fit.step()
            if (i+1)%50 == 0:
                if type(student_likelihood_fit) is gpytorch.likelihoods.GaussianLikelihood:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                            i + 1, args.student_initial_train_epochs, loss.item(),
                            student_model_fit.covar_module.base_kernel.lengthscale.item(),
                            student_likelihood_fit.noise.item()
                        ))
                elif type(student_likelihood_fit) is gpytorch.likelihoods.FixedNoiseGaussianLikelihood:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                            i + 1, args.student_initial_train_epochs, loss.item(),
                            student_model_fit.covar_module.base_kernel.lengthscale.item(),
                            student_likelihood_fit.second_noise.item()
                        ))
    
    teacher_model.train()
    teacher_trainer = args.teacher_optimizer
    train_iter_label = netmodel.load_array((train_x, train_y), args.batch_size)


    for i in range(args.teacher_initial_train_epochs):        
        for Z_labeled, Y_labeled in train_iter_label:
            teacher_trainer.zero_grad()
            Y_labeled_hat = teacher_model(Z_labeled)
            loss = teacher_model.compute_loss(Y_labeled_hat, Y_labeled)
            loss.backward()
            teacher_trainer.step()
        if (i+1)%50 == 0:
            with torch.no_grad():
                train_y_hat = teacher_model.pred_determinstic(train_x)
                teacher_train_loss = torch.nn.MSELoss(reduction='mean')(train_y_hat, train_y)
                print('%d/%d,teacher mse loss:%.6f, teacher loss:%.6f'%(i+1,args.teacher_initial_train_epochs,teacher_train_loss.item(), loss.item()))
            # early termination if loss is small enough
            if teacher_train_loss < 1e-3:
                break
            
def softplus_warp(x):
    return torch.nn.Softplus()(x)+1e-6

def joint_train_teacher_student(args, train_x, train_y, valid_x, valid_y, x_bounds, datamodule):
    train_x_normed = normalize(train_x, x_bounds)
    valid_x_normed = normalize(valid_x, x_bounds)
    idx_top = torch.argsort(train_y, dim=0, descending=True)
    topk_x_normed = torch.cat([train_x_normed[idx_top[i]] for i in range(args.topk)],0)
    train_w = torch.ones_like(train_y,device=args.device,dtype=torch.float)
    train_w = train_w.clone().detach().view(-1, 1).to(train_x)
    teacher_model:netmodel.TeacherModel = args.teacher
    student_model:GPModel = args.student
    teacher_trainer = args.teacher_optimizer
    student_model.set_train_data(train_x_normed, train_y)
    student_model.train()
    topk_y = torch.cat([train_y[idx_top[i]] for i in range(args.topk)],0)
    train_iter = netmodel.load_array((train_x, train_y), args.batch_size)
    epoch_displayinfo = int(args.joint_train_epochs)
    loss_unlabel = 0.
    is_gaussian_distribution = args.unlabel_sample_type=='gaussian'

    student_model_fit:ExactGPModel = args.student_model_fit
    student_likelihood_fit = args.student_likelihood_fit
    student_loss_logp_fit = args.student_loss_logp_fit
    student_trainer_fit = args.student_trainer_fit

    if args.x_unlabeled_given:
        X_unlabeled = args.X_unlabeled
        X_unlabeled_normed = normalize(X_unlabeled, x_bounds)
    else:
        if args.unlabel_sample_type=='gaussian':
            unlabel_num = args.unlabel_ratio*args.topk
            unlabel_mean = topk_x_normed[0].clone().detach().view(1,-1)
            unlabel_mean_variable = unlabel_mean.clone().detach()
            unlabel_mean_variable.requires_grad=True
            unlabel_std = args.unlabel_gauss_std*torch.ones_like(unlabel_mean).to(unlabel_mean)
            unlabel_std_variable = unlabel_std.clone().detach()
            unlabel_std_variable.requires_grad=True
            X_unlabeled_normed_raw = torch.normal(0., 1., size=(unlabel_num, unlabel_mean.shape[1])).to(unlabel_mean)
            X_unlabeled_normed = gaussian_rsample_warp(X_unlabeled_normed_raw, unlabel_mean, unlabel_std)
            gaussian_trainer = torch.optim.Adam([unlabel_mean_variable, unlabel_std_variable], lr=args.gaussian_lr)
            X_unlabeled = unnormalize(X_unlabeled_normed, x_bounds)

        elif args.unlabel_sample_type=='gev':
            X_unlabeled_normed = mcmc_different_init(args.gevtop,args, teacher_model, train_y, train_x_normed, x_bounds, args.unlabel_ratio)
            X_unlabeled = unnormalize(X_unlabeled_normed, x_bounds)
        else: # elif args.unlabel_sample_type=='random':
            X_unlabeled_normed = torch.rand(args.unlabel_ratio*args.topk,args.latent_dim,device=args.device,dtype=torch.float)
            X_unlabeled = unnormalize(X_unlabeled_normed, x_bounds)

    for epoch in range(args.joint_train_epochs):

        #train teacher model label loss
        teacher_model.train()
        for X_labeled, Y_labeled in train_iter:

            teacher_trainer.zero_grad()
            Y_labeled_hat = teacher_model(X_labeled)
            loss_label = teacher_model.compute_loss(Y_labeled_hat, Y_labeled)
            loss_label.backward()
            teacher_trainer.step()
        if args.ts_version==1 or args.ts_version==4:
            with torch.no_grad():
                Y_unlabeled = teacher_model(X_unlabeled)
            student_model_fit.update_train_data(X_unlabeled_normed, Y_unlabeled)
            student_model_fit.train()
            student_likelihood_fit.train()
            output = student_model_fit(X_unlabeled_normed)
            loss = -student_loss_logp_fit(output, student_model_fit.train_targets)
            student_trainer_fit.zero_grad()
            loss.backward()
            student_trainer_fit.step()
            student_model.set_train_data(X_unlabeled_normed, Y_unlabeled)
            Y_unlabeled = teacher_model(X_unlabeled)
            if type(teacher_model) is netmodel.VMLPModel:
                student_model.update_params(student_model_fit, teacher_model.decode_variance(Y_unlabeled[:,1]))
            else:
                student_model.update_params(student_model_fit, 0)
        
            student_model.eval()
            if args.is_validation:
                loss_unlabel = args.unlabel_loss_ratio*student_model.pred_nll(valid_x_normed,valid_y,args.unlabel_loss)
            else:
                loss_unlabel = args.unlabel_loss_ratio*student_model.pred_nll(topk_x_normed,topk_y,args.unlabel_loss)
            teacher_trainer.zero_grad()
            loss_unlabel.backward()
            teacher_trainer.step()
            
            if is_gaussian_distribution:
                teacher_model.eval()
                student_model.eval()
                X_unlabeled_normed_variable = gaussian_rsample_warp(X_unlabeled_normed_raw, unlabel_mean_variable, unlabel_std_variable)
                X_unlabeled_variable = unnormalize(X_unlabeled_normed_variable, x_bounds)
                Y_unlabeled_variable = teacher_model(X_unlabeled_variable)
                student_model.set_train_data(X_unlabeled_normed_variable, Y_unlabeled_variable)
                loss_unlabel = args.unlabel_loss_ratio*student_model.pred_nll(topk_x_normed,topk_y,args.unlabel_loss)
                gaussian_trainer.zero_grad()
                loss_unlabel.backward()
                gaussian_trainer.step()
                unlabel_mean = unlabel_mean_variable.clone().detach()
                unlabel_std = unlabel_std_variable.clone().detach()
                X_unlabeled_normed = gaussian_rsample_warp(X_unlabeled_normed_raw, unlabel_mean, unlabel_std)
                X_unlabeled = unnormalize(X_unlabeled_normed, x_bounds)



        if epoch % epoch_displayinfo == epoch_displayinfo-1:
            teacher_model.eval()
            with torch.no_grad():
                train_y_hat = teacher_model.pred_determinstic(train_x)
                teacher_train_loss = torch.nn.MSELoss()(train_y_hat, train_y.reshape(train_y_hat.shape))
                np.set_printoptions(precision=3)
            print('epoch:%d/%d, t train loss:%.3f, t unlabel loss:%.3f '
                's lengthscale:%.3f '
                's noise:%.3f '% 
                (       epoch+1, args.joint_train_epochs, teacher_train_loss, loss_unlabel, 
                        torch.exp(student_model.loglengthscale),
                        torch.exp(student_model.lognugget)*student_model.scale)) #,' unlabel cov:%.3f'%unlabel_cov)
            if is_gaussian_distribution:
                print('unlabel mean:', unlabel_mean.cpu().numpy())
                print('unlabel std:', unlabel_std.cpu().numpy())
    with torch.no_grad():
        Y_unlabeled = teacher_model(X_unlabeled)
    with torch.no_grad():
        return X_unlabeled_normed, teacher_model.pred_determinstic(X_unlabeled).reshape(-1,1)


def train_teacher_student(args, train_x:torch.Tensor, train_y:torch.Tensor, valid_x:torch.Tensor, valid_y:torch.Tensor, x_bounds, datamodule):
    gevtop:GEV = args.gevtop

    # remove recursive samples
    train_xy = torch.cat([train_x, train_y], dim=1)
    train_xy = torch.unique(train_xy, dim=0)
    train_x = train_xy[:,:-1]
    if train_x.ndim==1:
        train_x = train_x.reshape(-1,1)
    train_y = train_xy[:,-1].reshape(-1,1)
    if args.ts_train_count == 0:
        idx_top = torch.argsort(train_y, dim=0, descending=True)
        topk_y = torch.cat([train_y[idx_top[i]] for i in range(args.topk)],0)
        gevtop.init_params(torch.flatten(topk_y.detach()), args.gev_lr)
        gevtop.train()
        gevtop.train_epochs()
        pretrain_teacher_student(args, train_x, train_y, x_bounds)
        args.ts_train_count = 1
        args.n_init = train_x.shape[0]
    if args.is_validation:
        if args.validation_type == 'fixed':
            unlabel_n = int(args.n_init*args.validation_ratio)
            valid_x = train_x[:unlabel_n]
            valid_y = train_y[:unlabel_n]
            train_x = train_x[unlabel_n:]
            train_y = train_y[unlabel_n:]
        elif args.validation_type == 'labelall':
            valid_x = train_x
            valid_y = train_y
        elif args.validation_type == 'random':
            label_n = train_x.shape[0]
            unlabel_n = int(label_n*args.validation_ratio)
            indices = random.sample(range(label_n),unlabel_n)
            indices_train = [i not in indices for i in range(label_n)]
            valid_x = train_x[indices,:]
            valid_y = train_y[indices,:]
            train_x = train_x[indices_train,:]
            train_y = train_y[indices_train,:]



    return joint_train_teacher_student(args, train_x, train_y, valid_x, valid_y, x_bounds, datamodule)

class UnlabelDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x=x
        self.y=y
        self.len = len(y)
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
class UnlabelDataMoudle(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    def setup(self, dset):
        self.dset = dset
    def train_dataloader(self):
        return DataLoader(self.dset, batch_size=self.batch_size)

def get_bounds(x, y, tkwargs):
    # To account for outliers
    bounds = torch.zeros(2, x.shape[1], **tkwargs)
    bounds[0] = torch.quantile(x, .0005, dim=0)
    bounds[1] = torch.quantile(x, .9995, dim=0)
    ybounds = torch.zeros(2, y.shape[1], **tkwargs)
    ybounds[0] = torch.quantile(y, .0005, dim=0)
    ybounds[1] = torch.quantile(y, .9995, dim=0)
    ydelta = .05 * (ybounds[1] - ybounds[0])
    ybounds[0] -= ydelta
    ybounds[1] += ydelta

    # make sure best sample is within bounds
    y_train_std = y.add(-y.mean()).div(y.std())
    i = torch.argmax(y_train_std.flatten())
    bounds[1] = torch.maximum(x[i], bounds[1])
    bounds[0] = torch.minimum(x[i], bounds[0])
    # bounds = put_max_in_bounds(x, y_train_std, bounds)

    # print(f"Data bound of {bounds} found...")
    delta = .05 * (bounds[1] - bounds[0])
    bounds[0] -= delta
    bounds[1] += delta
    # print(f"Using data bound of {bounds}...")
    return bounds, ybounds
