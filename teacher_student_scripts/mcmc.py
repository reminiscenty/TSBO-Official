from botorch.utils.transforms import normalize, unnormalize
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import teacher_student_scripts.gev as gev
import random
from torch.distributions.multivariate_normal import MultivariateNormal
def mcmc(gev:gev.GEV,args, model, z0_normed, z_bounds, n, n_warmup,descending=True):
    z_new_normed_all = torch.zeros((n, z0_normed.shape[0])).to(z0_normed)
    i = 0
    z_normed = z0_normed.clone()
    with torch.no_grad():
        y = model.pred_determinstic(unnormalize(z_normed,z_bounds))
    # assert not torch.isnan(y), print(unnormalize(z_normed,z_bounds), z_normed, z_bounds)
    pa_max = torch.ones(1).to(z0_normed)
    while i<n+n_warmup:
        z_new_normed = proposal_dist(args, model,z_normed, z_bounds,descending=descending)
        with torch.no_grad():
            y_new = model.pred_determinstic(unnormalize(z_new_normed,z_bounds))
        if torch.isnan(gev(y_new)) or torch.isinf(gev(y_new)):
            gev.reset_params()
        pa = gev(y_new)/(gev(y)*torch.exp(proposal_dist(args, model,z_normed, z_bounds,z_new_normed,descending=descending)-proposal_dist(args, model, z_new_normed, z_bounds,z_normed,descending=descending)))
        pa = torch.min(pa_max,pa)
        a = random.uniform(0., 1.)
        # print(f'{i}/{n+n_warmup},','y:%.3f, y_new:%.3f, p(y):%.3f, p(y_new):%.3f, %.3f, %.3f'%(y, y_new, p(y), p(y_new), proposal_dist(args, model,z_normed, z_bounds,z_new_normed), proposal_dist(args, model, z_new_normed, z_bounds,z_normed)))
        if a <= pa:
            z_normed = z_new_normed
            y = y_new
            if i >=n_warmup:
                z_new_normed_all[i-n_warmup,:] = z_normed.clone()
            i += 1
    return z_new_normed_all.detach()
def mcmc_different_init(gev:gev.GEV,args, model, train_y, train_z_normed, z_bounds, n, n_warmup=30, descending=True):
    if descending:
        idx = torch.argsort(train_y, dim=0, descending=True)
        topk_y = torch.cat([train_y[idx[i]] for i in range(args.topk)],0)
    else:
        idx = torch.argsort(-train_y, dim=0, descending=True)
        topk_y = torch.cat([-train_y[idx[i]] for i in range(args.topk)],0)
    topk_z_normed = torch.cat([train_z_normed[idx[i]] for i in range(args.topk)],0)
    gev.train_epochs(torch.flatten(topk_y))
    z_all = torch.zeros((n*topk_z_normed.shape[0],topk_z_normed.shape[1])).to(train_z_normed)
    for i in range(topk_z_normed.shape[0]):
        z0 = topk_z_normed[i]
        z_all[i*n:(i+1)*n,:] = mcmc(gev,args, model, z0, z_bounds, n, n_warmup,descending=descending)
    return z_all

def proposal_dist(args, model, z1_normed, z_bounds, z2=None,descending=True):
    model.eval()
    z_normed = z1_normed.view(1,-1).detach().clone()
    z = unnormalize(z_normed, z_bounds)
    z.requires_grad = True
    if descending:
        y1 = model.pred_determinstic(z)
    else:
        y1 = -model.pred_determinstic(z)
    y1.backward()
    g = z.grad
    # print('g:',g)
    # print('z_normed:',z_normed)
    if torch.sum(g**2) != 0.:
        g = g/(torch.sqrt(torch.sum(g**2)))
    # print('g:',g)
    # print(y1)
    covariance_matrix = args.lambda_0**2*torch.eye(z.shape[1]).to(z)+(args.lambda_m**2-args.lambda_0**2)*torch.matmul(g.view(-1, 1),g.view(1,-1))
    m = MultivariateNormal(torch.flatten(z_normed), covariance_matrix)
    if z2 is None:
        #sample data
        z_new_normed = m.rsample()
        return z_new_normed #unnormalize(z_new_normed, z_bounds)
    else:
        #get log prob
        # return torch.exp(m.log_prob(normalize(z2, z_bounds)))
        return m.log_prob(z2)
   
