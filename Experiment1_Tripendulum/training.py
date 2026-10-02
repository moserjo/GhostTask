import torch
import gpytorch
from PCGP import laplace_approx, LaplaceResult
from dataclasses import dataclass, field
import copy
import resource
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONVERGENCE_WINDOW = 50    # iterations the physical parameter has to stay stable over
CONVERGENCE_TOL = 1e-4     # 0.01%: peak-to-peak variation over that window, relative to its mean


def peak_rss():
    """Peak resident set size of this process so far, in bytes (Linux: ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def is_stable(trajectory, window=CONVERGENCE_WINDOW, tol=CONVERGENCE_TOL):
    """True if the last `window` values vary by less than `tol` (relative, peak-to-peak)."""
    if len(trajectory) < window:
        return False
    w = torch.stack(list(trajectory[-window:])).flatten().double()
    return bool(((w.max() - w.min()) / w.mean().abs().clamp_min(1e-12)) < tol)


@dataclass
class train_output:
    parameters_during_training:dict
    laplace:LaplaceResult
    loss:list
    final_parameters:dict = field(default_factory=dict)
    MAP_estimates:torch.Tensor = None
    parameter_index:dict = field(default_factory=dict)
    n_iter:int = 0
    converged:bool = False
    max_iter_reached:bool = False


def train(model, likelihood, parameters, train_x, train_y, training_iter=50, learning_rate = 0.1, laplace = True,
          convergence_parameters = (), max_iter = None,
          convergence_window = CONVERGENCE_WINDOW, convergence_tol = CONVERGENCE_TOL):
    """
    Trains the model with Adam.

    `training_iter` is the MINIMUM number of iterations. If `convergence_parameters` is given
    (the physical parameters the experiment reports), training continues past it until each of
    them is stable to within `convergence_tol` over the last `convergence_window` iterations,
    or until `max_iter` (default: 5x training_iter) is reached. With no `convergence_parameters`
    the loop runs exactly `training_iter` iterations, as before.
    """
    parameters_during_training = {key:[] for key in parameters}
    loss_landscape = []
    if max_iter is None:
        max_iter = 5*training_iter if convergence_parameters else training_iter
    max_iter = max(max_iter, training_iter)

    optimizer = torch.optim.Adam(model.named_parameters(), lr=learning_rate)
    marginal_log_likelihood  = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    converged = False
    i = 0
    while i < max_iter:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -marginal_log_likelihood(output, train_y)

            for key in parameters:
                    parameters_during_training[key].append(copy.deepcopy(model.covar_module.get_param(key).detach()))
            loss_landscape.append(loss.detach())

            if i%100==0 and not device=="cuda":
                print("iteration: ", i, "loss:", loss.item())
            loss.backward(retain_graph = True)
            optimizer.step()
            i += 1

            if i >= training_iter and convergence_parameters:
                converged = all(is_stable(parameters_during_training[key], convergence_window, convergence_tol)
                                for key in convergence_parameters)
                if converged:
                    break

    n_iter = i
    max_iter_reached = bool(convergence_parameters) and not converged
    print("iterations: ", n_iter, " converged: ", converged)

    model.train()
    likelihood.train()        
    optimizer.zero_grad()
    output = model(train_x)  
    final_loss = -marginal_log_likelihood(output, train_y)*train_y.shape[0] #necessary to get correct gradients
    print("Final loss: ", final_loss.item())

    #final parameter values, and the same MAP_estimates/parameter_index the Laplace step used to return
    final_parameters = {key: model.covar_module.get_param(key).detach().cpu() for key in parameters}
    parameters_with_gradient = [key for key in parameters if model.covar_module.get_param(key).requires_grad]
    parameter_index = {name: i for i, name in enumerate(parameters_with_gradient)}
    MAP_estimates = torch.squeeze(torch.tensor([model.covar_module.get_param(k).detach() for k in parameters_with_gradient]))

    laplace_return = None
    if laplace:
        laplace_return = laplace_approx(parameters, model, final_loss)
    
    loss_landscape = torch.stack(loss_landscape).detach().cpu().numpy() 
    for key in parameters_during_training:
        parameters_during_training[key] =  torch.stack(parameters_during_training[key]).detach().cpu().numpy()
    return train_output(parameters_during_training=parameters_during_training,
                        laplace = laplace_return,
                        loss=loss_landscape,
                        final_parameters = final_parameters,
                        MAP_estimates = MAP_estimates,
                        parameter_index = parameter_index,
                        n_iter = n_iter,
                        converged = converged,
                        max_iter_reached = max_iter_reached,
                  )
