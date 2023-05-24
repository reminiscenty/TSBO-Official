import math
from tkinter.messagebox import NO
import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskGP
# We will use the simplest form of GP model, exact inference
import gpytorch.lazy

def vmv(leftVector: torch.Tensor,mat: torch.Tensor, rightVector: torch.Tensor):
    return leftVector.T @ mat @ rightVector
class ExactGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    def __init__(self, train_x, train_y,likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self._num_outputs  =1
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def update_train_data(self, x, y):
        if y.ndim==1:
            self.set_train_data(x, y, strict=False)
        elif y.ndim==2:
            self.set_train_data(x, y[:,0], strict=False)
            if y.shape[1] ==2:
                self.likelihood.noise = self.softplus(y[:,1]) +1e-6

class GPModel(torch.nn.Module):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, lr):
        super().__init__()
        self.device = train_x.device
        self.x = train_x
        if train_y.ndim==1:
            self.y = train_y
        else:
            self.y = train_y[:,0]
        self.mu = torch.zeros(1,device=self.device)
        self.scale = torch.zeros(1,device=self.device)
        self.lognugget = torch.zeros(1,device=self.device)
        self.loglengthscale = torch.zeros(1,device=self.device)
        self.fixed_noise = torch.zeros_like(self.y, device=self.device)
        self.kernel = 'rbf'
        self.nll = torch.nn.GaussianNLLLoss()
        self.lr = lr
        self.optimizer = torch.optim.Adam([self.lognugget,self.loglengthscale],lr=self.lr)
        self.n = self.x.shape[0]
        self.one = torch.ones(self.x.shape[0],1,device=self.device)
        self.K : torch.Tensor
        self.inv_K : torch.Tensor
        self.mu : torch.Tensor
        self.scale : torch.Tensor
        self.dist : torch.Tensor
        self.MSELoss = torch.nn.MSELoss()
        self.updateKernel()

    def getKernel(self,x1: torch.Tensor,x2: torch.Tensor):
        K = torch.zeros(x1.shape[0],x2.shape[0],device=self.device).to(x1)
        for i in range(x1.shape[0]):
            K[i,:] = torch.exp( -(torch.norm((x1[i]-x2), dim=1)**2) / (2* torch.exp(self.loglengthscale)**2))
        return K*self.scale

    def getDist(self,x1: torch.Tensor,x2: torch.Tensor):
        dist = torch.zeros(x1.shape[0],x2.shape[0],device=self.device).to(x1)
        for i in range(x1.shape[0]):
            dist[i] = torch.exp( -(torch.norm((x1[i]-x2), dim=1)**2) / (2* torch.exp(self.loglengthscale)**2))
        return dist*self.scale

    def updateKernel(self):
        self.K = self.getKernel(self.x,self.x) \
                    +torch.diag(((1e-4+torch.exp(self.lognugget))*torch.ones(self.x.shape[0],device=self.device)).view(-1) + self.fixed_noise)
        self.inv_K = torch.linalg.inv(self.K)
        # self.dist = self.getDist(self.x,self.x)

    def updateMuVar(self):
        self.mu = (self.one.T @ self.inv_K @ self.y)/ (self.one.T @ self.inv_K @self.one)
        self.scale = (self.y-self.mu).T @ self.inv_K @ (self.y-self.mu)/self.inv_K.shape[0]

    def set_train_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.x = train_x
        if train_y.ndim==1:
            self.y = train_y
        else:
            self.y = train_y[:,0]
            if train_y.shape[1]==2:
                self.fixed_noise = train_y[:,1]/self.scale
            else:
                self.fixed_noise = 0
        self.updateKernel()
    
    def train_step(self):
        self.optimizer.zero_grad()
        loss = self.train_nll()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def updateTrainableParams(self,lengthscale: torch.Tensor,noise: torch.Tensor):
        self.loglengthscale = torch.log(lengthscale+1e-4)
        self.lognugget = torch.log(noise/(self.scale+1e-4)+1e-4)
    def updateMu(self,inv_K: torch.Tensor):
        self.mu = vmv(self.one,self.inv_K,self.y)/vmv(self.one,self.inv_K,self.one)
        # self.mu = mu.detach()
        # return mu

    def updateScale(self, inv_K: torch.Tensor):
        vector = self.y-self.mu
        self.scale = vmv(vector,inv_K,vector)/inv_K.shape[0]
        # self.variance = variance.detach()
        # return variance

    def train_nll(self):
        self.updateKernel()
        self.updateMuVar()
        _,nll = torch.linalg.slogdet(self.K)
        nll = nll + self.n*torch.log(self.scale).squeeze()
        return nll
    def setHparamsGrad(self):
        self.updateKernel()
        self.updateMuVar()
        tempRowVector = self.y.T@self.inv_K
        denomi = (tempRowVector@self.y)
        dot_K = self.K*self.dist
        # lengthscale grad
        g = self.n/2 * (tempRowVector @ dot_K @ tempRowVector.T) / denomi - 1/2*torch.diag(self.inv_K@dot_K).sum()
        self.loglengthscale.grad = -(torch.exp(self.loglengthscale) * g).detach().view(1)

        # noise grad
        g = self.n/2* (tempRowVector**2).sum() / denomi - 1/2*torch.diag(self.inv_K).sum()
        self.lognugget.grad = -(torch.exp(self.lognugget) * g).detach().view(1)
    def predict(self, input_x):
        k = self.getDist(self.x,input_x)
        self.mu = self.mu.detach()
        self.scale = self.scale.detach()
        # Mean prediction
        f = self.mu + k.T @ self.inv_K @ (self.y-self.mu*self.one)
        
        # Variance prediction
        var = (self.scale)*(1 - torch.diag(k.T @ self.inv_K @ k)+torch.exp(self.lognugget)).reshape(-1,1)
        return f,var
    def pred_nll(self, input_x, input_y,score_type='Mahalanobis'):
        k = self.getDist(self.x,input_x)
        input_K = self.getDist(input_x,input_x)
        # Mean prediction
        f = self.mu + k.T @ self.inv_K @ (self.y-self.mu)
        
        # Variance prediction
        # var = input_K-k.T @ self.inv_K @ k + torch.diag((torch.exp(self.lognugget)*self.scale*torch.ones(input_x.shape[0],device=self.device)).view(-1))
        var = input_K-k.T @ self.inv_K @ k + torch.diag((torch.exp(self.lognugget)*self.scale*torch.ones(input_x.shape[0],device=self.device)).view(-1))
        nll:torch.Tensor
        if score_type == 'NLL':
            _,nll = torch.linalg.slogdet(var)
            pred_error = input_y-f.reshape(input_y.shape)
            pred_error.reshape(-1, 1)
            nll = nll + vmv(pred_error,torch.linalg.inv(var),pred_error).squeeze()
        elif score_type == 'Mahalanobis':
            pred_error = input_y-f.reshape(input_y.shape)
            pred_error.reshape(-1, 1)
            nll = vmv(pred_error,torch.linalg.inv(var),pred_error).squeeze()
        else: #elif score_type == 'MSE':
            nll = self.MSELoss(f,input_y.view(f.shape))
        return nll
    def print_params(self):
        print('lengthscale: %.3f   noise: %.3f, mu:%.3f, scale:%.3f' % (
                torch.exp(self.loglengthscale).item(),
                (torch.exp(self.lognugget)*self.scale).item(),
                self.mu.item(),
                self.scale.item()
            ))
        
    def updateOptimizer(self):
        self.mu.requires_grad = True
        self.scale.requires_grad = True
        self.loglengthscale.requires_grad = True
        self.lognugget.requires_grad = True
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
    def update_params(self, gpmodel:ExactGPModel, fixed_noise):

        # update mu, scale, lengthscale, nugget (adding independent noise), and optional fixed dependent noise
        with torch.no_grad():
            self.mu = gpmodel.mean_module.constant
            self.scale = gpmodel.covar_module.outputscale
            self.loglengthscale = torch.log(gpmodel.covar_module.base_kernel.lengthscale)
            if type(gpmodel.likelihood) is gpytorch.likelihoods.GaussianLikelihood:
                self.lognugget = torch.log(gpmodel.likelihood.noise[0]/self.scale)
                self.fixed_noise = (torch.zeros_like(self.y, device=self.device))/ self.scale
            elif type(gpmodel.likelihood) is gpytorch.likelihoods.FixedNoiseGaussianLikelihood:
                self.lognugget = torch.log(gpmodel.likelihood.second_noise[0]/self.scale)
                self.fixed_noise = fixed_noise/self.scale
        if type(gpmodel.likelihood) is gpytorch.likelihoods.GaussianLikelihood:
            self.fixed_noise = (torch.zeros_like(self.y, device=self.device))/ self.scale
        elif type(gpmodel.likelihood) is gpytorch.likelihoods.FixedNoiseGaussianLikelihood:
            self.fixed_noise = fixed_noise/self.scale
       
        # update kernel invK
        self.updateKernel()

        # update optimizer
        self.updateOptimizer()





            


class ExactMarginalLogLikelihoodNew(ExactMarginalLogLikelihood):
    def forward(self, function_dist, target, *params):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        if not isinstance(function_dist, gpytorch.distributions.MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        for p in params:
            print(f'params:{p}')
        # print(f'function_dist:{function_dist}')
        # print(f'target.shape:{target.shape}')
        output = self.likelihood(function_dist, *params)
        # print(f'output:{output}')
        res = self.log_prob(output, target)
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        # print(num_data)
        return res.div_(num_data)

    def log_prob(self, output, value):
        '''

        :param output: gpytorch.distributions.MultivariateNormal
        :param value: 2 * N tensor, [0,:] left tensor, [1,:] right tensor
        :return: y^T*K^(-1)*y
        '''
        # if settings.fast_computations.log_prob.off():
        #     return super().log_prob(value)
        #
        # if self._validate_args:
        #     self._validate_sample(value)
        # print(f'traget.shape={value.shape}')
        # print(value)
        mean, covar = output.loc, output.lazy_covariance_matrix
        a = covar.logdet()
        # print(a)
        left_diff = value[0,:] - mean
        right_diff = value[1,:] #- mean
        # print(covar)
        # print(f'right={right_diff}, left={left_diff}')

        # Repeat the covar to match the batch shape of left_diff
        if left_diff.shape[:-1] != covar.batch_shape:
            # print('if True:')
            if len(left_diff.shape[:-1]) < len(covar.batch_shape):
                left_diff = left_diff.expand(covar.shape[:-1])
            else:
                padded_batch_shape = (*(1 for _ in range(left_diff.dim() + 1 - covar.dim())), *covar.batch_shape)
                covar = covar.repeat(
                    *(diff_size // covar_size for diff_size, covar_size in zip(left_diff.shape[:-1], padded_batch_shape)),
                    1,
                    1,
                )

        # Get log determininant and first part of quadratic form
        covar = covar.evaluate_kernel()
        # print(covar)
        inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=left_diff.unsqueeze(-1), logdet=True)
        # print(f'inv_quad={inv_quad}, loglet={logdet}')
        inv_quad = covar.inv_matmul(right_tensor=right_diff, left_tensor=left_diff.reshape(1, -1))
        logdet = covar.logdet()
        # print(f'inv_quad={inv_quad}, loglet={logdet}')

        # res = -0.5 * sum([inv_quad, logdet, left_diff.size(-1) * math.log(2 * math.pi)])
        res = -0.5 * sum([inv_quad, left_diff.size(-1) * math.log(2 * math.pi)])
        return res


def train_step(model, loss, trainer, x, y, is_update=True, epochs=1):
    if x is not None and y is not None:
        model.set_train_data(x, y, strict=False)
    for _ in range(epochs):
        trainer.zero_grad()
        output = model(x)
        l = -loss(output, y)
        l.backward()
        if is_update:
            trainer.step()
    # print([name for name, p in model.named_parameters()])
    return torch.as_tensor([p.grad.detach() for name, p in model.named_parameters()])
