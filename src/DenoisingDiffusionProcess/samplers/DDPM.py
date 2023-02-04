"""

    This file contains the DDPM sampler class for a diffusion process

"""
import torch
from torch import nn

from ..beta_schedules import *

class DDPM_Sampler(nn.Module):
    
    def __init__(self,
                 num_timesteps=1000,
                 schedule='linear'
                ):
        
        super().__init__()
        
        self.num_timesteps=num_timesteps
        self.schedule=schedule
        
        self.register_buffer('betas',get_beta_schedule(self.schedule,self.num_timesteps))
        self.register_buffer('betas_sqrt',self.betas.sqrt())
        self.register_buffer('alphas',1-self.betas)
        self.register_buffer('alphas_cumprod',torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt',(1-self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt',self.alphas.sqrt())
        self.register_buffer('alphas_sqrt_recip',1/(self.alphas_sqrt))
        
    @torch.no_grad()
    def forward(self,*args,**kwargs):   
        return self.step(*args,**kwargs)
    
    @torch.no_grad()
    def step(self,x_t,t,z_t):
        """
            Given approximation of noise z_t in x_t predict x_(t-1)
        """
        assert (t<self.num_timesteps).all()
        
        # 2. Approximate Distribution of Previous Sample in the chain 
        mean_pred,std_pred=self.posterior_params(x_t,t,z_t)
        
        # 3. Sample from the distribution
        z=torch.randn_like(x_t) if any(t>0) else torch.zeros_like(x_t)
        return mean_pred + std_pred*z
    
    def posterior_params(self,x_t,t,noise_pred):
        
        assert (t<self.num_timesteps).all()
        
        beta_t=self.betas[t].view(x_t.shape[0],1,1,1)
        alpha_one_minus_cumprod_sqrt_t=self.alphas_one_minus_cumprod_sqrt[t].view(x_t.shape[0],1,1,1)
        alpha_sqrt_recip_t=self.alphas_sqrt_recip[t].view(x_t.shape[0],1,1,1)
        
        mean=alpha_sqrt_recip_t*(x_t-beta_t*noise_pred / alpha_one_minus_cumprod_sqrt_t)
        std=self.betas_sqrt[t].view(x_t.shape[0],1,1,1)
        
        return mean, std