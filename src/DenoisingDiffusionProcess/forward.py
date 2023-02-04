"""

    This file contains implementations of the forward diffusion process

    Current Models:
    
    1) Gaussian Diffusion

"""
import torch
from torch import nn

from .beta_schedules import *

class ForwardModel(nn.Module):
    """
        (Forward Model Template)
    """
    
    def __init__(self,
                 num_timesteps=1000,
                 schedule='linear'
                ):
        
        super().__init__()
        self.schedule=schedule
        self.num_timesteps=num_timesteps
     
    @torch.no_grad()
    def forward(self, x_0, t):
        """
            Get noisy sample at t given x_0
        """
        raise NotImplemented
    
    @torch.no_grad()
    def step(self, x_t, t):
        """
            Get next sample in the process
        """
        raise NotImplemented   
        

class GaussianForwardProcess(ForwardModel):
    """
        Gassian Forward Model
    """
    
    def __init__(self,
                 num_timesteps=1000,
                 schedule='linear'
                ):
        
        super().__init__(num_timesteps=num_timesteps,
                         schedule=schedule
                        )
        
        # get process parameters
        self.register_buffer('betas',get_beta_schedule(self.schedule,self.num_timesteps))
        self.register_buffer('betas_sqrt',self.betas.sqrt())
        self.register_buffer('alphas',1-self.betas)
        self.register_buffer('alphas_cumprod',torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt',(1-self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt',self.alphas.sqrt())
     
    @torch.no_grad()
    def forward(self, x_0, t, return_noise=False):
        """
            Get noisy sample at t given x_0
            
            q(x_t | x_0)=N(x_t; alphas_cumprod_sqrt(t)*x_0, 1-alpha_cumprod(t)*I)
        """
        assert (t<self.num_timesteps).all()
        
        b=x_0.shape[0]
        mean=x_0*self.alphas_cumprod_sqrt[t].view(b,1,1,1)
        std=self.alphas_one_minus_cumprod_sqrt[t].view(b,1,1,1)
        
        noise=torch.randn_like(x_0)
        output=mean+std*noise        
        
        if not return_noise:
            return output
        else:
            return output, noise
    
    @torch.no_grad()
    def step(self, x_t, t, return_noise=False):
        """
            Get next sample in the process
            
            q(x_t | x_t-1)=N(x_t; alphas_sqrt(t)*x_0,betas(t)*I)
        """
        assert (t<self.num_timesteps).all()
        
        mean=self.alphas_sqrt[t]*x_t
        std=self.betas_sqrt[t]
        
        noise=torch.randn_like(x_0)
        output=mean+std*noise        
        
        if not return_noise:
            return output
        else:
            return output, noise