# MIT License

# Copyright (c) 2019 John Lalor <john.lalor@nd.edu> and Pedro Rodriguez <me@pedro.ai>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from py_irt.models import abstract_model
import pyro
import pyro.distributions as dist
import torch

import torch.distributions.constraints as constraints

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD

import pyro.contrib.autoguide as autoguide

import pandas as pd

from functools import partial

import numpy as np

import torch.nn as nn
import torch.nn.functional as F


@abstract_model.IrtModel.register("1pl_bhm")
class OneParamBHM(abstract_model.IrtModel):
    """1PL Hierarchical IRT model"""

    def __init__(
            self, *,
            priors: str,
            num_items: int,
            num_subjects: int,
            num_knowledges: int,
            verbose: bool = False,
            device: str = "cpu",
            vocab_size: int = None,
            dropout: float = None,
            hidden: int = None,
            **kwargs
    ):
        super().__init__(
            device=device, num_items=num_items, num_subjects=num_subjects, verbose=verbose
        )
        self.num_knowledges = num_knowledges
        self.priors = priors
        assert self.priors in ["pool", "unpool", "bhm"], f'Prior `{self.priors}` not implemented'
    
    
    def model_pool(self, subjects, items, knowledges, obs):
        with pyro.plate("subjects", self.num_subjects, device=self.device):
            ability = pyro.sample(
                "theta_i",
                dist.Normal(
                    torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
                ),
            )

        with pyro.plate("items", self.num_items, device=self.device):
            diff = pyro.sample(
                "beta_j",
                dist.Normal(
                    torch.tensor(0.0, device=self.device), torch.tensor(1.0e3, device=self.device)
                ),
            )

        with pyro.plate("observations", len(items), device=self.device):
            pyro.sample("obs", dist.Bernoulli(logits=ability[subjects] - diff[items]), obs=obs)
    
    
    def guide_pool(self, models, items, knowledges, obs):
        # register learnable params in the param store
        theta_loc_param = pyro.param(
            "theta_i_loc", torch.zeros(self.num_subjects, device=self.device)
        )
        theta_scale_param = pyro.param(
            "theta_i_scale",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        beta_loc_param = pyro.param("beta_j_loc", torch.zeros(self.num_items, device=self.device))
        beta_scale_param = pyro.param(
            "beta_j_scale",
            torch.empty(self.num_items, device=self.device).fill_(1.0e3),
            constraint=constraints.positive,
        )

        # guide distributions
        with pyro.plate("subjects", self.num_subjects, device=self.device):
            dist_theta = dist.Normal(theta_loc_param, theta_scale_param)
            pyro.sample("theta_i", dist_theta)
        with pyro.plate("items", self.num_items, device=self.device):
            dist_b = dist.Normal(beta_loc_param, beta_scale_param)
            pyro.sample("beta_j", dist_b)
    
    
    def model_unpool(self, subjects, items, knowledges, obs):
        with pyro.plate("subjects", self.num_subjects, device=self.device):
            with pyro.plate("knowledges", self.num_knowledges, device=self.device):
                ability = pyro.sample(
                    "theta_ik",
                    dist.Normal(
                        torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
                    ),
                )

        with pyro.plate("items", self.num_items, device=self.device):
            diff = pyro.sample(
                "beta_j",
                dist.Normal(
                    torch.tensor(0.0, device=self.device), torch.tensor(1.0e3, device=self.device)
                ),
            )

        with pyro.plate("observe_data", len(items), device=self.device):
            pyro.sample("obs", dist.Bernoulli(logits=ability[subjects, knowledges] - diff[items]), obs=obs)
    
    
    def guide_unpool(self, models, items, knowledges, obs):
        # register learnable params in the param store
        theta_loc_param = pyro.param(
            "theta_ik_loc", torch.zeros(self.num_subjects, self.num_knowledges, device=self.device)
        )
        theta_scale_param = pyro.param(
            "theta_ik_scale",
            torch.ones(self.num_subjects, self.num_knowledges, device=self.device),
            constraint=constraints.positive,
        )
        beta_loc_param = pyro.param("beta_j_loc", torch.zeros(self.num_items, device=self.device))
        beta_scale_param = pyro.param(
            "beta_j_scale",
            torch.empty(self.num_items, device=self.device).fill_(1.0e3),
            constraint=constraints.positive,
        )

        # guide distributions
        with pyro.plate("subjects", self.num_subjects, device=self.device):
            with pyro.plate("knowledges", self.num_knowledges, device=self.device):
                dist_theta = dist.Normal(theta_loc_param, theta_scale_param)
                pyro.sample("theta_ik", dist_theta)
        with pyro.plate("items", self.num_items, device=self.device):
            dist_b = dist.Normal(beta_loc_param, beta_scale_param)
            pyro.sample("beta_j", dist_b)


    def model_bhm(self, subjects, items, knowledges, obs):
        """Initialize a 1PL model with hierarchical priors"""
        mu_b = pyro.sample(
            "mu_b",
            dist.Normal(
                torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device)
            ),
        )
        u_b = pyro.sample(
            "u_b",
            dist.Gamma(
                torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)
            ),
        )
        mu_theta = pyro.sample(
            "mu_theta",
            dist.Normal(
                torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device)
            ),
        )
        u_theta = pyro.sample(
            "u_theta",
            dist.Gamma(
                torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)
            ),
        )
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))
        with pyro.plate("bs", self.num_items, device=self.device):
            diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))
        
        with pyro.plate("observe_data", len(items)):
            pyro.sample("obs", dist.Bernoulli(logits=ability[subjects] - diff[items]), obs=obs)


    def guide_bhm(self, subjects, items, knowledges, obs):
        """Initialize a 1PL guide with hierarchical priors"""
        loc_mu_b_param = pyro.param("loc_mu_b", torch.tensor(0.0, device=self.device))
        scale_mu_b_param = pyro.param(
            "scale_mu_b", torch.tensor(1.0e2, device=self.device), constraint=constraints.positive
        )
        loc_mu_theta_param = pyro.param("loc_mu_theta", torch.tensor(0.0, device=self.device))
        scale_mu_theta_param = pyro.param(
            "scale_mu_theta",
            torch.tensor(1.0e2, device=self.device),
            constraint=constraints.positive,
        )
        alpha_b_param = pyro.param(
            "alpha_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        beta_b_param = pyro.param(
            "beta_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        alpha_theta_param = pyro.param(
            "alpha_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        beta_theta_param = pyro.param(
            "beta_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        m_theta_param = pyro.param(
            "loc_ability", torch.zeros(self.num_subjects, device=self.device)
        )
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param(
            "scale_diff",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )

        # sample statements
        pyro.sample("mu_b", dist.Normal(loc_mu_b_param, scale_mu_b_param))
        pyro.sample("u_b", dist.Gamma(alpha_b_param, beta_b_param))
        pyro.sample("mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param))
        pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))

        with pyro.plate("thetas", self.num_subjects, device=self.device):
            pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))
        with pyro.plate("bs", self.num_items, device=self.device):
            pyro.sample("b", dist.Normal(m_b_param, s_b_param))

    def get_model(self):
        return getattr(self, "model_" + self.priors)

    def get_guide(self):
        return getattr(self, "guide_" + self.priors)

    def fit(self, models, items, responses, num_epochs):
        """Fit the IRT model with variational inference"""
        optim = Adam({"lr": 0.1})
        svi = SVI(self.get_model(), self.get_guide*(), optim, loss=Trace_ELBO())

        pyro.clear_param_store()
        for j in range(num_epochs):
            loss = svi.step(models, items, responses)
            if j % 100 == 0 and self.verbose:
                print("[epoch %04d] loss: %.4f" % (j + 1, loss))

        print("[epoch %04d] loss: %.4f" % (j + 1, loss))

    def export(self):
        # TODO
        return {}

    def predict(self, subjects, items, knowledges, params_from_file=None):
        """predict p(correct | params) for a specified list of model, item pairs"""
        predictive = pyro.infer.Predictive(self.get_model(), guide=self.get_guide(), num_samples=800)
        svi_samples = predictive(subjects, items, knowledges, obs=None)
        svi_obs = svi_samples["obs"].data.cpu().numpy().mean(axis=0, keepdims=False).astype(float)
        assert len(svi_obs) == len(subjects), f'len(svi_obs) = {svi_obs.shape}'
        return svi_obs

