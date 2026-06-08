
import torch
import gpytorch
from einops import rearrange
from PCGP import ConstraintsModifications


class PCGP_Kernel_0(gpytorch.kernels.Kernel):
    def __init__(self, parameter_modifications = {}, number_of_input_dimensions=3, num_tasks=8, **kwargs):
        super().__init__()
        self.num_tasks = num_tasks
        self.parameter_dict = {'nu0': None, 'amplitude': None, 'lengthscale': None}
        self.param_constraints = {}
        self.number_of_input_dimensions = number_of_input_dimensions
                           
        for param_name in self.parameter_dict:
            raw_name = f"raw_{param_name}"
            param = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.register_parameter(raw_name, param)   
            if param_name[:-2] == "amplitude" or param_name[:-2] == "lengthscale":
                self.register_constraint(raw_name, gpytorch.constraints.Positive())
                self.param_constraints[param_name] = gpytorch.constraints.Positive() 
                           
        for param_name, (value, requires_grad, constraint) in parameter_modifications.items():
            if param_name in self.parameter_dict:
                raw_name = f"raw_{param_name}"
                raw_param = getattr(self, raw_name)
                # handle constraints
                if constraint and requires_grad:
                    CM = ConstraintsModifications(constraint)
                    value = value if CM.is_fulfilled(value) else CM.init_val_from_constraint()
                    self.register_constraint(raw_name, constraint)
                    self.param_constraints[param_name] = constraint
                self.set_param(param_name, value)   
                raw_param.requires_grad_(requires_grad)

    def _set_param(self, param_name, value):
        raw_name = f"raw_{param_name}"
        param = getattr(self, raw_name)
        value = torch.as_tensor(value).to(param)
        if param_name in self.param_constraints:
            value = self.param_constraints[param_name].inverse_transform(value)
        self.initialize(**{raw_name: value})

    def get_param(self, param_name):
        raw_name = f"raw_{param_name}"
        raw_param = getattr(self, raw_name)
        if param_name in self.param_constraints:
            return self.param_constraints[param_name].transform(raw_param)
        return raw_param

    def get_raw_param(self, param_name):
        return getattr(self, f"raw_{param_name}")

    def set_param(self, param_name, value):
        if param_name in self.parameter_dict:
            self._set_param(param_name, value)
        else:
            raw_name = f"raw_{param_name}"
            value_tensor = torch.nn.Parameter(torch.tensor([value]))
            self.register_parameter(raw_name, value_tensor)


    def forward(self, x1, x2, diag=False, **params):
        N, M = x1.size(0), x2.size(0)
            # 1. Separate features from indices
            # We use .long() because indices must be integers for argsort/bincount
        idx1 = x1[:, -1].long() #indices must be integers for argsort/bincount"
        idx2 = x2[:, -1].long()
        data1 = x1[:, :-1]
        data2 = x2[:, :-1]
        if data1.dim() == 1: #make sure data is 2D to avoid dimension issues
            data1 = data1.unsqueeze(-1)
        if data2.dim() == 1:
            data2 = data2.unsqueeze(-1)
                    # 2. Sort indices and reorder data rows
        sort_idx1 = torch.argsort(idx1)
        sort_idx2 = torch.argsort(idx2)
            
        sorted_data1 = data1[sort_idx1]
        sorted_data2 = data2[sort_idx2]
            
            # 3. Get split sizes
        counts1 = torch.bincount(idx1).tolist()
        counts2 = torch.bincount(idx2).tolist()
        splits1 = torch.split(sorted_data1, counts1)
        splits2 = torch.split(sorted_data2, counts2)
        nu0 = self.get_param('nu0')
        amplitude = self.get_param('amplitude')
        lengthscale = self.get_param('lengthscale')
        def k00_fn(x, y):
            return -amplitude*(-2*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
        def k01_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k02_fn(x, y):
            return -amplitude*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
        def k03_fn(x, y):
            return amplitude*nu0*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2 - (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k04_fn(x, y):
            return amplitude*(-nu0*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + nu0*(-(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,2] + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*y[...,2]) + (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k05_fn(x, y):
            return -amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
        def k06_fn(x, y):
            return amplitude*nu0*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
        def k07_fn(x, y):
            return amplitude*nu0*(x[...,1] - y[...,1])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*y[...,2]/lengthscale**3
        def k10_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k11_fn(x, y):
            return -amplitude*(-2*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
        def k12_fn(x, y):
            return -amplitude*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
        def k13_fn(x, y):
            return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,2] - y[...,2])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k14_fn(x, y):
            return amplitude*(nu0*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2 + (lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*y[...,2]) - (x[...,0] - y[...,0])*(nu0*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1]) + nu0*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])*y[...,2] + nu0*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])*y[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])*y[...,2]))*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k15_fn(x, y):
            return -amplitude*nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
        def k16_fn(x, y):
            return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
        def k17_fn(x, y):
            return -amplitude*nu0*(x[...,0] - y[...,0])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*y[...,2]/lengthscale**3
        def k20_fn(x, y):
            return -amplitude*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
        def k21_fn(x, y):
            return -amplitude*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
        def k22_fn(x, y):
            return amplitude*(lengthscale - (x[...,2] - y[...,2])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**2
        def k23_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k24_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k25_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k26_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k27_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k30_fn(x, y):
            return amplitude*nu0*((lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale + (lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] - (x[...,2] - y[...,2])**2) - (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k31_fn(x, y):
            return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,2] - y[...,2])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k32_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k33_fn(x, y):
            return amplitude*nu0**2*(3*lengthscale**3 - lengthscale**2*(2*(x[...,1] - y[...,1])**2 + 5*(x[...,2] - y[...,2])**2) - lengthscale*((lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2 - 4*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*y[...,2] - 2*(lengthscale - (x[...,0] - y[...,0])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2 - (lengthscale - (x[...,2] - y[...,2])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2] + (2*lengthscale**3 - 2*lengthscale**2*(2*(x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + lengthscale*((lengthscale - (x[...,0] - y[...,0])**2)**2 + 4*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2) - (lengthscale - (x[...,0] - y[...,0])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] + (3*lengthscale**3 - 3*lengthscale**2*(x[...,1] - y[...,1])**2 + lengthscale*(-(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2 - (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2 + 2*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] + (6*lengthscale**3 - 18*lengthscale**2*(x[...,1] - y[...,1])**2 + 9*lengthscale*(lengthscale - (x[...,1] - y[...,1])**2)**2 - (3*lengthscale - (x[...,1] - y[...,1])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**6
        def k34_fn(x, y):
            return amplitude*nu0*(nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] - nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(-9*lengthscale**2 + lengthscale*(3*(x[...,0] - y[...,0])**2 + 2*(x[...,1] - y[...,1])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(-3*lengthscale**2 + lengthscale*((x[...,0] - y[...,0])**2 + 2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])**2)*x[...,2] + nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(2*lengthscale**2 - 4*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)**2) + nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(12*lengthscale**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*x[...,2]*y[...,2] + nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(12*lengthscale**2 - 6*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2))*y[...,2] - nu0*(2*lengthscale**3 - 2*lengthscale**2*(2*(x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + lengthscale*((lengthscale - (x[...,0] - y[...,0])**2)**2 + 4*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2) - (lengthscale - (x[...,0] - y[...,0])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - nu0*(3*lengthscale**3 - lengthscale**2*(2*(x[...,1] - y[...,1])**2 + 5*(x[...,2] - y[...,2])**2) - lengthscale*((lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2 - 4*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) - nu0*(3*lengthscale**3 - 3*lengthscale**2*(x[...,1] - y[...,1])**2 - lengthscale*((3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2 - 2*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] + nu0*(-(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*y[...,2] + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*x[...,2]*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*x[...,2]*y[...,2]) + (lengthscale - (x[...,0] - y[...,0])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2] - (6*lengthscale**3 - 18*lengthscale**2*(x[...,1] - y[...,1])**2 + 9*lengthscale*(lengthscale - (x[...,1] - y[...,1])**2)**2 - (3*lengthscale - (x[...,1] - y[...,1])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**6
        def k35_fn(x, y):
            return -amplitude*nu0**2*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(3*lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*x[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (x[...,2] - y[...,2])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
        def k36_fn(x, y):
            return -amplitude*nu0**2*(x[...,2] - y[...,2])*(-3*lengthscale**2 + lengthscale*(2*(x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) - (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2 + (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
        def k37_fn(x, y):
            return -amplitude*nu0**2*(x[...,1] - y[...,1])*((lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,2] - y[...,2])**2) + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2) + (14*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*x[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*y[...,2]/lengthscale**5
        def k40_fn(x, y):
            return amplitude*(nu0*(-(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - (lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2) + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,2] + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*x[...,2]) + (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k41_fn(x, y):
            return -amplitude*(-nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] - nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,2] - y[...,2])**2) + nu0*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) + nu0*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,2] + nu0*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*x[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k42_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k43_fn(x, y):
            return amplitude*nu0*(nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] + nu0*(lengthscale - (x[...,2] - y[...,2])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2] - nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(-3*lengthscale**2 + lengthscale*((x[...,1] - y[...,1])**2 + 2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2)*x[...,2] + nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(2*lengthscale**2 - 4*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)**2) + nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(8*lengthscale**2 - 2*lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2))*x[...,2]*y[...,2] + nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(12*lengthscale**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*x[...,2]*y[...,2] + nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(12*lengthscale**2 - 6*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2))*x[...,2] - nu0*(2*lengthscale**3 - 2*lengthscale**2*(2*(x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + lengthscale*((lengthscale - (x[...,0] - y[...,0])**2)**2 + 4*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2) - (lengthscale - (x[...,0] - y[...,0])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - nu0*(3*lengthscale**3 - lengthscale**2*(2*(x[...,1] - y[...,1])**2 + 5*(x[...,2] - y[...,2])**2) - lengthscale*((lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2 - 4*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + nu0*(-(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*x[...,2])*y[...,2] + (lengthscale - (x[...,0] - y[...,0])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - (3*lengthscale**3 - 3*lengthscale**2*(x[...,1] - y[...,1])**2 - lengthscale*((3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2 - 2*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] - (6*lengthscale**3 - 18*lengthscale**2*(x[...,1] - y[...,1])**2 + 9*lengthscale*(lengthscale - (x[...,1] - y[...,1])**2)**2 - (3*lengthscale - (x[...,1] - y[...,1])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**6
        def k44_fn(x, y):
            return -amplitude*(-nu0**2*(2*lengthscale**3 - 2*lengthscale**2*(2*(x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + lengthscale*((lengthscale - (x[...,0] - y[...,0])**2)**2 + 4*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2) - (lengthscale - (x[...,0] - y[...,0])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - nu0**2*(3*lengthscale**3 - lengthscale**2*(2*(x[...,0] - y[...,0])**2 + 5*(x[...,1] - y[...,1])**2) - lengthscale*((lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2 - 4*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2) + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - nu0**2*(3*lengthscale**3 - lengthscale**2*(2*(x[...,0] - y[...,0])**2 + 5*(x[...,2] - y[...,2])**2) - lengthscale*((lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2 - 4*(x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2) - nu0**2*(3*lengthscale**3 - lengthscale**2*(2*(x[...,1] - y[...,1])**2 + 5*(x[...,2] - y[...,2])**2) - lengthscale*((lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2 - 4*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) - nu0**2*(6*lengthscale**3 - 18*lengthscale**2*(x[...,2] - y[...,2])**2 + 9*lengthscale*(lengthscale - (x[...,2] - y[...,2])**2)**2 - (3*lengthscale - (x[...,2] - y[...,2])**2)**2*(x[...,2] - y[...,2])**2)*x[...,2]*y[...,2] - nu0*(3*lengthscale**3 - 3*lengthscale**2*(x[...,1] - y[...,1])**2 - lengthscale*((3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2 - 2*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] + nu0*(-nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*y[...,2] + 2*nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*x[...,2]*y[...,2] + 2*nu0*(lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*x[...,2]*y[...,2] + nu0*(lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,2] + nu0*(lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,2] + 2*(3*lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*x[...,2]*y[...,2]) + nu0*(-nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*x[...,2] - nu0*(lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2)*x[...,2] - nu0*(lengthscale - (x[...,2] - y[...,2])**2)*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*y[...,2] - nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(-9*lengthscale**2 + lengthscale*(3*(x[...,0] - y[...,0])**2 + 2*(x[...,1] - y[...,1])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] - nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(-3*lengthscale**2 + lengthscale*((x[...,0] - y[...,0])**2 + 2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])**2)*x[...,2] - nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(-3*lengthscale**2 + lengthscale*((x[...,1] - y[...,1])**2 + 2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2)*x[...,2] + 2*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(2*lengthscale**2 - 4*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)**2) + nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(8*lengthscale**2 - 2*lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2))*x[...,2]*y[...,2] + nu0*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*(12*lengthscale**2 - 6*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2))*x[...,2] + nu0*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*(12*lengthscale**2 - 6*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2))*y[...,2] + nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(12*lengthscale**2 - 6*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2))*x[...,2] + nu0*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(12*lengthscale**2 - 6*lengthscale*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2))*y[...,2] + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2] + (lengthscale - (x[...,2] - y[...,2])**2)*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2] + 2*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(12*lengthscale**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*x[...,2]*y[...,2]) - (6*lengthscale**3 - 18*lengthscale**2*(x[...,1] - y[...,1])**2 + 9*lengthscale*(lengthscale - (x[...,1] - y[...,1])**2)**2 - (3*lengthscale - (x[...,1] - y[...,1])**2)**2*(x[...,1] - y[...,1])**2)*x[...,2]*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**6
        def k45_fn(x, y):
            return amplitude*nu0*(nu0*((x[...,0] - y[...,0])*(-3*lengthscale**2 + 3*lengthscale*(x[...,2] - y[...,2])**2 + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2)*x[...,2] + (x[...,2] - y[...,2])*(-3*lengthscale**2 + lengthscale*(2*(x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])**2)) + (x[...,2] - y[...,2])*(-nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + nu0*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,2] + nu0*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1]) + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,2]))*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
        def k46_fn(x, y):
            return amplitude*nu0*(nu0*(x[...,1] - y[...,1])*(-3*lengthscale**2 + 3*lengthscale*(x[...,2] - y[...,2])**2 + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2)*x[...,2] + nu0*(x[...,2] - y[...,2])*(-3*lengthscale**2 + lengthscale*(2*(x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1])**2) + nu0*(x[...,2] - y[...,2])*(-(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*x[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*x[...,2] + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])) + (x[...,2] - y[...,2])*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*x[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
        def k47_fn(x, y):
            return amplitude*nu0*(nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1]) + nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])*x[...,2] + nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])*x[...,2] + nu0*(lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0]) + nu0*(lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0])*x[...,2] + nu0*(lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])*x[...,2] + nu0*(lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,0] - y[...,0])**2)*(x[...,0] - y[...,0]) + nu0*(lengthscale - (x[...,2] - y[...,2])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1]) + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])*x[...,2] + (nu0*(x[...,0] - y[...,0])*(2*lengthscale**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)**2) + nu0*(x[...,1] - y[...,1])*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2) + (x[...,1] - y[...,1])*(12*lengthscale**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)))*x[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*y[...,2]/lengthscale**5
        def k50_fn(x, y):
            return amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
        def k51_fn(x, y):
            return amplitude*nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
        def k52_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k53_fn(x, y):
            return amplitude*nu0**2*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(3*lengthscale + (3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,2] - y[...,2])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
        def k54_fn(x, y):
            return amplitude*nu0*(nu0*(-(x[...,0] - y[...,0])*(-3*lengthscale**2 + 3*lengthscale*(x[...,2] - y[...,2])**2 + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2)*y[...,2] + (x[...,2] - y[...,2])*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2) + (x[...,0] - y[...,0])**2*(x[...,1] - y[...,1])**2)*y[...,2] + (x[...,2] - y[...,2])*(3*lengthscale**2 - lengthscale*(3*(x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2)) - (x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(nu0*(3*lengthscale - (x[...,0] - y[...,0])**2)*y[...,2] + nu0*(3*lengthscale - (x[...,2] - y[...,2])**2) + (3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2]))*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
        def k55_fn(x, y):
            return amplitude*nu0**2*(lengthscale**2 - lengthscale*((x[...,0] - y[...,0])**2 + (x[...,2] - y[...,2])**2) + (x[...,0] - y[...,0])**2*(x[...,2] - y[...,2])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k56_fn(x, y):
            return -amplitude*nu0**2*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k57_fn(x, y):
            return -amplitude*nu0**2*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*y[...,2]/lengthscale**4
        def k60_fn(x, y):
            return -amplitude*nu0*(lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
        def k61_fn(x, y):
            return -amplitude*nu0*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**3
        def k62_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k63_fn(x, y):
            return -amplitude*nu0**2*(x[...,2] - y[...,2])*(3*lengthscale**2 - lengthscale*(3*(x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2 - (-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
        def k64_fn(x, y):
            return -amplitude*nu0*(nu0*(x[...,1] - y[...,1])*(-3*lengthscale**2 + 3*lengthscale*(x[...,2] - y[...,2])**2 + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])**2)*y[...,2] - nu0*(x[...,2] - y[...,2])*(3*lengthscale**2 - lengthscale*(3*(x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2) + nu0*(x[...,2] - y[...,2])*(-(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*y[...,2] + (3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])) + (x[...,2] - y[...,2])*(-3*lengthscale**2 + 3*lengthscale*(x[...,1] - y[...,1])**2 + (3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])**2)*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**5
        def k65_fn(x, y):
            return -amplitude*nu0**2*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0])*(x[...,1] - y[...,1])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k66_fn(x, y):
            return amplitude*nu0**2*(lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2) + (x[...,1] - y[...,1])**2*(x[...,2] - y[...,2])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)/lengthscale**4
        def k67_fn(x, y):
            return -amplitude*nu0**2*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*y[...,2]/lengthscale**4
        def k70_fn(x, y):
            return -amplitude*nu0*(x[...,1] - y[...,1])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*x[...,2]/lengthscale**3
        def k71_fn(x, y):
            return amplitude*nu0*(x[...,0] - y[...,0])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*x[...,2]/lengthscale**3
        def k72_fn(x, y):
            return torch.zeros_like(x[...,0]-y[...,0], device=x.device)
        def k73_fn(x, y):
            return amplitude*nu0**2*(x[...,1] - y[...,1])*(3*lengthscale**2 - lengthscale*((x[...,1] - y[...,1])**2 + 2*(x[...,2] - y[...,2])**2) + (-lengthscale + (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2 + (lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale + 2*(3*lengthscale - (x[...,1] - y[...,1])**2)*y[...,2] - (x[...,2] - y[...,2])**2) + (2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2)*y[...,2] + (12*lengthscale**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*x[...,2]/lengthscale**5
        def k74_fn(x, y):
            return -amplitude*nu0*(nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,1] - y[...,1]) + nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])*y[...,2] + nu0*(lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])*y[...,2] + nu0*(lengthscale - (x[...,1] - y[...,1])**2)*(lengthscale - (x[...,2] - y[...,2])**2)*(x[...,0] - y[...,0]) + nu0*(lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,2] - y[...,2])**2)*(x[...,2] - y[...,2])*y[...,2] - nu0*(x[...,0] - y[...,0])*(-3*lengthscale**2 + lengthscale*((x[...,0] - y[...,0])**2 + 2*(x[...,1] - y[...,1])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,1] - y[...,1])**2)*y[...,2] - nu0*(x[...,0] - y[...,0])*(-3*lengthscale**2 + lengthscale*((x[...,0] - y[...,0])**2 + 2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,0] - y[...,0])**2)*(x[...,2] - y[...,2])**2) + nu0*(x[...,0] - y[...,0])*(2*lengthscale**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)**2)*y[...,2] - nu0*(x[...,1] - y[...,1])*(-3*lengthscale**2 + lengthscale*((x[...,1] - y[...,1])**2 + 2*(x[...,2] - y[...,2])**2) + (lengthscale - (x[...,1] - y[...,1])**2)*(x[...,2] - y[...,2])**2) + nu0*(x[...,1] - y[...,1])*(2*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2)*y[...,2] + (lengthscale - (x[...,0] - y[...,0])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2)*(x[...,1] - y[...,1])*y[...,2] + (x[...,1] - y[...,1])*(12*lengthscale**2 - 6*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,1] - y[...,1])**2)*(3*lengthscale - (x[...,1] - y[...,1])**2))*y[...,2])*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*x[...,2]/lengthscale**5
        def k75_fn(x, y):
            return -amplitude*nu0**2*(x[...,0] - y[...,0])*(x[...,2] - y[...,2])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*x[...,2]/lengthscale**4
        def k76_fn(x, y):
            return -amplitude*nu0**2*(x[...,1] - y[...,1])*(x[...,2] - y[...,2])*(-4*lengthscale + (x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*x[...,2]/lengthscale**4
        def k77_fn(x, y):
            return amplitude*nu0**2*(4*lengthscale**2 - 4*lengthscale*(x[...,0] - y[...,0])**2 - 4*lengthscale*(x[...,1] - y[...,1])**2 + (lengthscale - (x[...,0] - y[...,0])**2)**2 + 2*(lengthscale - (x[...,0] - y[...,0])**2)*(lengthscale - (x[...,1] - y[...,1])**2) + (lengthscale - (x[...,1] - y[...,1])**2)**2)*torch.exp(-1/2*((x[...,0] - y[...,0])**2 + (x[...,1] - y[...,1])**2 + (x[...,2] - y[...,2])**2)/lengthscale)*x[...,2]*y[...,2]/lengthscale**4
        function_grid = [
            [k00_fn, k01_fn, k02_fn, k03_fn, k04_fn, k05_fn, k06_fn, k07_fn],
            [k10_fn, k11_fn, k12_fn, k13_fn, k14_fn, k15_fn, k16_fn, k17_fn],
            [k20_fn, k21_fn, k22_fn, k23_fn, k24_fn, k25_fn, k26_fn, k27_fn],
            [k30_fn, k31_fn, k32_fn, k33_fn, k34_fn, k35_fn, k36_fn, k37_fn],
            [k40_fn, k41_fn, k42_fn, k43_fn, k44_fn, k45_fn, k46_fn, k47_fn],
            [k50_fn, k51_fn, k52_fn, k53_fn, k54_fn, k55_fn, k56_fn, k57_fn],
            [k60_fn, k61_fn, k62_fn, k63_fn, k64_fn, k65_fn, k66_fn, k67_fn],
            [k70_fn, k71_fn, k72_fn, k73_fn, k74_fn, k75_fn, k76_fn, k77_fn],
        ]
                           
        rows = []
        for i, s1 in enumerate(splits1):
            row_blocks = []
            s1_expanded = s1.unsqueeze(1)
            for j, s2 in enumerate(splits2):
                s2_expanded = s2.unsqueeze(0)
                block = function_grid[i][j](s1_expanded, s2_expanded)
                row_blocks.append(block)
            rows.append(torch.cat(row_blocks, dim=1))
        sorted_matrix = torch.cat(rows, dim=0)
        # assemble kernel
        K = torch.empty((N, M), device=x1.device, dtype=sorted_matrix.dtype)
        K[sort_idx1[:, None], sort_idx2] = sorted_matrix
        if diag:
            return torch.diag(K)
        return K


class PCGP_Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  parameter_modifications = {}, number_of_input_dimensions = 3, num_tasks = 8, priors = None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.num_tasks = num_tasks
        self.number_of_input_dimensions = number_of_input_dimensions
        self.covar_module = (
                            PCGP_Kernel_0(parameter_modifications, number_of_input_dimensions=self.number_of_input_dimensions, num_tasks=self.num_tasks)
                            )
                        
        if priors:
            for name, prior in priors.items():
                self.register_prior(
                    name+"_prior",
                    prior,
                    lambda m: m.covar_module.get_param(name),
                    lambda m, val: m.covar_module._set_param(name, val),
                )

    def forward(self, x):
        mean_x = self.mean_module(x[:,0]) ##no need for task specific mean since it's zero
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 