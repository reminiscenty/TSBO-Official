
from math import log
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
# from pathlib import Path
# ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
# print(ROOT_PROJECT)
# sys.path.insert(0, ROOT_PROJECT)

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, BASE_DIR)
from gp_models import train_step, ExactGPModel
import netmodel as netmodel
from gev import GEV

from mcmc import mcmc_different_init




def set_random_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

def add_tsbo_args(parser:ArgumentParser):
    tsbo_group = parser.add_argument_group("tsbo")
    tsbo_group.add_argument('--random_sample', action='store_true')
    tsbo_group.add_argument(
        "--self_training",
        action='store_true',
        help="Train teacher student model with unlabelled data",
    )
    tsbo_group.add_argument(
        "--unlabel_retrain",
        action='store_true',
        help="Train teacher student model with unlabelled data",
    )
    tsbo_group.add_argument(
        "--unlabel_retrain_type",
        type=str,
        default='lhs',
        help="Train teacher student model with unlabelled data",
    )
    tsbo_group.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Train teacher student model with unlabelled data",
    )
    return parser


def setup_args(args, task):
    '''
    set up teacher, student, gev
    '''
    device = torch.device('cuda:'+str(args.cuda))
    args.device = device
    args.gaussian_lr = 1e-3
    args.unlabel_logstd_value = 0.1
    if task =='expr':
        args.teacher = netmodel.ExprTeacher(args.latent_dim).to(device)
        args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-3)
        # args.gaussian_lr=1e-4
        args.unlabel_logstd_value = log(0.1)
    elif task =='topology':
        args.teacher = netmodel.TopologyTeacher(args.latent_dim).to(device)
        args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-4)
        # args.gaussian_lr=1e-4
        args.unlabel_logstd_value = log(0.1)
    elif task =='chem':
        args.teacher = netmodel.ChemTeacher(args.latent_dim).to(device)
        args.teacher_optimizer = torch.optim.Adam(args.teacher.parameters(), lr=1e-4)
        # args.gaussian_lr=1e-4
        args.unlabel_logstd_value = log(0.1)
        
    x = torch.tensor(0.,device=device)
    y = torch.tensor(0.,device=device)
    args.student = ExactGPModel(x,y).to(device)
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

def gaussian_rsample_warp(x_raw, mean, logstd):
    return x_raw*torch.exp(logstd) +mean

def pretrain_teacher_student(args, train_x, train_y, test_x, test_y, x_bounds, datamodule):
    train_x_normed = normalize(train_x, x_bounds)
    test_x_normed = normalize(test_x, x_bounds)
    # assert False, print(train_x[0], x_bounds, train_x_normed)
    # train_w = datamodule.data_weighter.weighting_function(torch.flatten(train_y).cpu().numpy())
    # test_w = datamodule.data_weighter.weighting_function(torch.flatten(test_y).cpu().numpy())
    # train_w = torch.tensor(train_w).view(-1, 1).to(train_x)
    # test_w = torch.tensor(test_w).view(-1, 1).to(train_x)
    teacher_model = args.teacher
    student_model = args.student
    student_likelihood = args.student.likelihood
    student_loss_logp = args.student_loss
    student_trainer = args.student_optimizer
    student_model.set_train_data(train_x_normed, torch.flatten(train_y), strict=False)
    student_model.train()
    student_likelihood.train()
    for _ in range(args.student_initial_train_epochs):
        # print(torch.flatten(train_y), train_z_normed)
        output = student_model(train_x_normed)
        loss = -student_loss_logp(output, torch.flatten(train_y))
        student_trainer.zero_grad()
        loss.backward()
        student_trainer.step()
    teacher_model.train()
    teacher_trainer = args.teacher_optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    # train_iter_z = netmodel.load_array((train_z, train_y), args.batch_size, train_w)
    train_iter_z = netmodel.load_array((train_x, train_y), args.batch_size)
    for i in range(args.teacher_initial_train_epochs):
        for Z_labeled, Y_labeled in train_iter_z:
            teacher_trainer.zero_grad()
            Y_labeled_hat = teacher_model.net(Z_labeled)
            loss_label = criterion(Y_labeled_hat, Y_labeled)
            teacher_l = loss_label
            teacher_l.backward()
            teacher_trainer.step()
        if (i+1)%500 == 0:
            with torch.no_grad():
                # teacher_train_loss = torch.nn.MSELoss(reduction='none')(teacher_model.net(train_z), train_y)
                # teacher_train_loss = torch.mean(teacher_train_loss*train_w.view(teacher_train_loss.shape))
                # teacher_test_loss = torch.nn.MSELoss(reduction='none')(teacher_model.net(test_z), test_y)
                # teacher_test_loss = torch.mean(teacher_test_loss*test_w.view(teacher_test_loss.shape))
                train_y_hat = teacher_model.net(train_x)
                test_y_hat = teacher_model.net(test_x)
                teacher_train_loss = torch.nn.MSELoss(reduction='mean')(train_y_hat, train_y)
                teacher_test_loss = torch.nn.MSELoss(reduction='mean')(test_y_hat, test_y)
                # a = [[b.item(),c.item()] for b, c in zip(train_y_hat, train_y)]
                # print(a)
                print('%d/%d,teacher train loss:%.3f, teacher test loss:%.3f'%(i+1,args.teacher_initial_train_epochs,teacher_train_loss.item(), teacher_test_loss.item()))

def joint_train_teacher_student(args, train_x, train_y, test_x, test_y, x_bounds, datamodule):
    train_x_normed = normalize(train_x, x_bounds)
    test_x_normed = normalize(test_x, x_bounds)
    # train_w = datamodule.data_weighter.weighting_function(torch.flatten(train_y).cpu().numpy())
    # test_w = datamodule.data_weighter.weighting_function(torch.flatten(test_y).cpu().numpy())
    train_w = torch.ones_like(train_y,device=args.device,dtype=torch.float)
    test_w = torch.ones_like(test_y,device=args.device,dtype=torch.float)
    train_w = train_w.clone().detach().view(-1, 1).to(train_x)
    test_w = test_w.clone().detach().view(-1, 1).to(train_x)
    teacher_model = args.teacher
    student_model = args.student
    teacher_trainer = args.teacher_optimizer
    student_likelihood = args.student.likelihood
    student_loss_logp = args.student_loss
    student_trainer = args.student_optimizer
    student_model.set_train_data(train_x_normed, torch.flatten(train_y), strict=False)
    student_model.train()
    student_likelihood.train()
    idx = torch.argsort(train_y, dim=0, descending=True)
    topk_x = torch.cat([train_x[idx[i]] for i in range(args.topk)],0)
    topk_x_normed = torch.cat([train_x_normed[idx[i]] for i in range(args.topk)],0)
    print('topk_x.shape:',topk_x.shape)
    topk_y = torch.cat([train_y[idx[i]] for i in range(args.topk)],0)
    topk_w = torch.cat([train_w[idx[i]] for i in range(args.topk)],0)
    train_iter = netmodel.load_array((train_x, train_y), args.batch_size)
    criterion = torch.nn.MSELoss(reduction='mean')
    teacher_param_n = sum(p.numel() for _, p in teacher_model.net.named_parameters())
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    teacher_loss_gnll = torch.nn.GaussianNLLLoss()
    epoch_displayinfo = int(args.joint_train_epochs)
    gnll_loss = torch.nn.GaussianNLLLoss(reduction='mean')
    loss_unlabel = 0.
    is_gaussian_distribution = args.random_sample
    if args.unlabel_loss =='mse':
        criterion_unlabel = torch.nn.MSELoss(reduction='none')
    else:
        assert args.unlabel_loss =='gaussian'
        criterion_unlabel = torch.nn.GaussianNLLLoss(reduction='none')
    if is_gaussian_distribution:
        unlabel_num = args.unlabel_ratio*args.topk
        unlabel_mean = topk_x_normed[0].clone().detach().view(1,-1)
        unlabel_mean_variable = unlabel_mean.clone().detach()
        unlabel_mean_variable.requires_grad=True
        unlabel_logstd = args.unlabel_logstd_value*torch.ones_like(unlabel_mean).to(unlabel_mean)
        unlabel_logstd_variable = unlabel_logstd.clone().detach()
        unlabel_logstd_variable.requires_grad=True
        X_unlabeled_normed_raw = torch.normal(0., 1., size=(unlabel_num, unlabel_mean.shape[1])).to(unlabel_mean)
        X_unlabeled_normed = gaussian_rsample_warp(X_unlabeled_normed_raw, unlabel_mean, unlabel_logstd)
        gaussian_trainer = torch.optim.Adam([unlabel_mean_variable, unlabel_logstd_variable], lr=args.gaussian_lr)
    # X_unlabeled = unnormalize(X_unlabeled_normed, x_bounds)

    else:
        X_unlabeled_normed = mcmc_different_init(args, teacher_model.net, train_y, train_x_normed, x_bounds, args.unlabel_ratio)
    if args.x_unlabeled_given:
        X_unlabeled_normed = args.x_unlabeled
    X_unlabeled = unnormalize(X_unlabeled_normed, x_bounds)
    for epoch in range(args.joint_train_epochs):
        # initialize loss
        teacher_train_loss, teacher_test_loss = torch.tensor(
            0.0, device=args.device), torch.tensor(0.0, device=args.device)
        student_train_loss, student_test_loss = torch.tensor(
            0.0, device=args.device), torch.tensor(0.0, device=args.device)

        #train teacher model label loss
        # teacher_model.train()
        # for X_labeled, Y_labeled in train_iter:
        #     # sample unlabeled data
        #     # Z_labeled_normed = normalize(Z_labeled, z_bounds)
        #     # Y_unlabeled += torch.empty(Y_unlabeled.shape,device=device).normal_(mean=0., std=unlabel_noise_std)



        #     # compute gradients of teacher model related to student model


        #     # teacher_trainer.step()
            
        #     teacher_trainer.zero_grad()
        #     Y_labeled_hat = teacher_model.net(X_labeled)
        #     assert Y_labeled_hat.shape == Y_labeled.shape
        #     loss_label = criterion(Y_labeled_hat, Y_labeled)
        #     # print(teacher_l.shape)
        #     loss_label.backward()
        #     teacher_trainer.step()
        #     # print('training loss: ', loss_label.item())

        with torch.no_grad():
            Y_unlabeled = teacher_model.net(X_unlabeled)

        # train student model
        student_model.train()
        student_likelihood.train()
        train_step(student_model, 
                student_loss_logp, student_trainer, X_unlabeled_normed, 
                torch.flatten(Y_unlabeled), is_update=True, epochs=1)
        
        # train teacher model unlabel loss
        teacher_model.train()
        student_model.eval()
        student_likelihood.eval()
        for _ in range(1):
            # if args.noised_unlabel:
            if False:
                g_t_un = torch.zeros(teacher_param_n, device=args.device)
                Y_unlabeled += torch.empty(Y_unlabeled.shape,device=args.device).normal_(mean=0., std=args.unlabel_noise_std)
                student_model.train()
                student_likelihood.train()
                h_right = train_step(student_model, 
                        student_loss_logp, student_trainer, X_unlabeled_normed, 
                        torch.flatten(Y_unlabeled), is_update=True)
                h_left = train_step(student_model, 
                        student_loss_logp, student_trainer, train_x_normed, 
                        torch.flatten(train_y), is_update=False)
                h = 1e-1 * cos(h_left, h_right) 


                # compute gradients of teacher model related to student model

                teacher_model.net.train()
                Y_unlabeled_hat = teacher_model.net(X_unlabeled)
                teacher_trainer.zero_grad()
                teacher_l = teacher_loss_gnll(Y_unlabeled_hat, Y_unlabeled, args.unlabel_noise_std*torch.ones(Y_unlabeled.shape, device=args.device))
                teacher_l.backward()
                
                j = 0
                for p in teacher_model.net.parameters():
                    if p.requires_grad:
                        lenth = p.numel()
                        g_t_un[j:j+lenth] = h * p.grad.detach().view(-1)
                        # p.grad *= h
                        j += lenth

                # teacher_trainer.step()
                
                teacher_trainer.zero_grad()
                j = 0
                # g_t = torch.empty(teacher_param_n).to(device)
                for p in teacher_model.net.parameters():
                    if p.requires_grad:
                        lenth = p.numel()
                        p.grad += g_t_un[j:j+lenth].view(p.grad.shape)
                        # g_t[j:j+lenth] = p.grad.detach().view(-1)
                        j += lenth

                # update teacher model
                teacher_trainer.step()
        
            else:
                Y_unlabeled = teacher_model.net(X_unlabeled) # with grad
                student_model.set_train_data(X_unlabeled_normed, torch.flatten(Y_unlabeled), strict=False)
                student_model.eval()
                student_likelihood.eval()
                output = student_model(topk_x_normed)
                if args.unlabel_loss =='mse':
                    loss_unlabel = criterion_unlabel(output.mean, torch.flatten(topk_y)).sum()
                else:
                    # loss_unlabel = criterion_unlabel(output.mean, torch.flatten(topk_y), output.variance)
                    # print(output.shape, torch.flatten(Y_unlabeled))
                    loss_unlabel = -student_loss_logp(output, torch.flatten(topk_y))
                # if args.weighted:
                #     loss_unlabel = torch.mean(topk_w.view(loss_unlabel.shape)*loss_unlabel)
                # else:
                #     loss_unlabel = torch.mean(loss_unlabel)
                print('joint loss:%.3f'% (loss_unlabel.item()))
                print('-------------------------------------------------------------------------------------------------------------')
                teacher_trainer.zero_grad()
                teacher_model.print_weights()
                print('-----------------------------------------------------------------------------------------------------------')
                loss_unlabel.backward()
                teacher_trainer.step()
                teacher_model.print_weights()
        #if gaussian unlabel, train hyper-params
        if is_gaussian_distribution:
            teacher_model.net.eval()
            student_model.eval()
            student_likelihood.eval()
            X_unlabeled_normed_variable = gaussian_rsample_warp(X_unlabeled_normed_raw, unlabel_mean_variable, unlabel_logstd_variable)
            X_unlabeled_variable = unnormalize(X_unlabeled_normed_variable, x_bounds)
            Y_unlabeled_variable = teacher_model.net(X_unlabeled_variable)
            student_model.set_train_data(X_unlabeled_normed_variable, torch.flatten(Y_unlabeled_variable), strict=False)
            output = student_model(topk_x_normed)
            if args.unlabel_loss =='mse':
                loss_unlabel = criterion_unlabel(output.mean, torch.flatten(topk_y))
            else:
                loss_unlabel = criterion_unlabel(output.mean, torch.flatten(topk_y), output.variance)
            if args.weighted:
                loss_unlabel = torch.mean(topk_w.view(loss_unlabel.shape)*loss_unlabel)
            else:
                loss_unlabel = torch.mean(loss_unlabel)
            gaussian_trainer.zero_grad()
            loss_unlabel.backward()
            gaussian_trainer.step()
            unlabel_mean = unlabel_mean_variable.clone().detach()
            unlabel_logstd = unlabel_logstd_variable.clone().detach()
            X_unlabeled_normed = gaussian_rsample_warp(X_unlabeled_normed_raw, unlabel_mean, unlabel_logstd)
            X_unlabeled = unnormalize(X_unlabeled_normed, x_bounds)



        if epoch % epoch_displayinfo == epoch_displayinfo-1:
            teacher_model.net.eval()
            student_model.eval()
            student_likelihood.eval()
            student_model.set_train_data(X_unlabeled_normed, torch.flatten(Y_unlabeled), strict=False)
            with torch.no_grad():
                train_y_hat = teacher_model.net(train_x)
                test_y_hat = teacher_model.net(test_x)
                teacher_train_loss = torch.nn.MSELoss(reduction='none')(train_y_hat, train_y)
                assert teacher_train_loss.shape == train_w.shape, print(teacher_train_loss.shape, train_w.shape)
                teacher_train_loss = torch.mean(teacher_train_loss*train_w)
                teacher_test_loss = torch.nn.MSELoss(reduction='none')(test_y_hat, test_y)
                assert teacher_test_loss.shape == test_w.shape, print(teacher_test_loss.shape, test_w.shape)
                teacher_test_loss = torch.mean(teacher_test_loss*test_w)
                # teacher_train_loss = torch.nn.MSELoss(reduction='mean')(train_y_hat, train_y)
                # teacher_test_loss = torch.nn.MSELoss(reduction='mean')(test_y_hat, test_y)
                pred = student_model(train_x_normed)
                student_train_loss = gnll_loss(pred.mean, torch.flatten(train_y), pred.variance)
                pred = student_model(test_x_normed)
                student_test_loss = gnll_loss(pred.mean, torch.flatten(test_y), pred.variance)
                a = [[b.item(),c.item(), d.item(), e[0].item()] for b, c, d, e in zip(topk_y, teacher_model.net(topk_x), student_model(topk_x_normed).mean, topk_x)]
                print(len(train_x))
                np.set_printoptions(precision=3)
                print(np.array(a))
                print(topk_x[0])
            print('epoch:%d/%d, t train loss:%.3f, t unlabel loss:%.3f '
                't test loss:%.3f '
                's train loss:%.3f '
                's test loss:%.3f '
                's lengthscale:%.3f '
                's noise:%.3f '% (epoch+1, args.joint_train_epochs, teacher_train_loss, loss_unlabel, teacher_test_loss,
                                student_train_loss, student_test_loss, student_model.covar_module.base_kernel.lengthscale,
                                student_model.likelihood.noise)) #,' unlabel cov:%.3f'%unlabel_cov)
            if is_gaussian_distribution:
                print('unlabel mean:', unlabel_mean.cpu().numpy())
                print('unlabel logstd:', unlabel_logstd.cpu().numpy())
    with torch.no_grad():
        Y_unlabeled = teacher_model.net(X_unlabeled)
    return X_unlabeled_normed, Y_unlabeled

def train_teacher_student(args, train_x, train_y, test_x, test_y, x_bounds, datamodule):
    p:GEV = args.p
    if args.ts_train_count == 0:
        idx = torch.argsort(train_y, dim=0, descending=True)
        topk_y = torch.cat([train_y[idx[i]] for i in range(args.topk)],0)
        p.init_params(torch.flatten(topk_y), args.gev_lr)
        p.train_epochs()
        pretrain_teacher_student(args, train_x, train_y, test_x, test_y, x_bounds, datamodule)
        args.ts_train_count = 1
    return joint_train_teacher_student(args, train_x, train_y, test_x, test_y, x_bounds, datamodule)

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

# def retrain_model_u(model, datamodule, save_dir, version_str, num_epochs, cuda, store_best=False,
#                   best_ckpt_path: Optional[str] = None):
#     # pl._logger.setLevel(logging.CRITICAL)
#     train_pbar = SubmissivePlProgressbar(process_position=1)

#     # Create custom saver and logger
#     tb_logger = TensorBoardLogger(
#         save_dir=save_dir, version=version_str, name=""
#     )
#     checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/val", )

#     # Handle fractional epochs
#     if num_epochs < 1:
#         max_epochs = 1
#         limit_train_batches = num_epochs
#     elif int(num_epochs) == num_epochs:
#         max_epochs = int(num_epochs)
#         limit_train_batches = 1.0
#     else:
#         raise ValueError(f"invalid num epochs {num_epochs}")

#     # Create trainer
#     trainer = pl.Trainer(
#         gpus=[cuda] if cuda is not None else 0,
#         max_epochs=max_epochs,
#         limit_train_batches=limit_train_batches,
#         limit_val_batches=1,
#         checkpoint_callback=True,
#         terminate_on_nan=True,
#         logger=tb_logger,
#         callbacks=[train_pbar, checkpointer],
#         gradient_clip_val=20.0,  # Model is prone to large gradients
#     )

#     # Fit model
#     trainer.fit(model, datamodule)

#     if store_best:
#         assert best_ckpt_path is not None
#         os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
#         shutil.copyfile(checkpointer.best_model_path, best_ckpt_path)
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
