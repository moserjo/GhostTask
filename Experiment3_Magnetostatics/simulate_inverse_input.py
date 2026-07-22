import os
import torch
import numpy as np

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

nu0_true = 1.
factor   = 40

# --- Analytical solution ---
def make_analytical(grid):
    xx = grid[:, 0]
    yy = grid[:, 1]
    zz = grid[:, 2]
    Ax    = factor * (yy - yy**2) * (zz - 1) * (zz - 2)
    Ay    = torch.zeros_like(xx)
    Az    = torch.zeros_like(xx)
    Jx    = factor * 2 * nu0_true * (zz**3 - 3*zz**2 + 2*zz - yy + yy**2)
    gauge = torch.zeros_like(xx)
    Bx    = torch.zeros_like(xx)
    By    = factor * (yy - yy**2) * (2*zz - 3)
    Bz    = -factor * (1 - 2*yy) * (zz - 1) * (zz - 2)
    return Ax, Ay, Az, Jx, gauge, Bx, By, Bz

# --- Test grid (shared across all runs, same as forward) ---
test_n    = 18  # matches 3*n from simulate_forward_input.py
test_grid = torch.cartesian_prod(
    torch.linspace(0, 1, test_n),
    torch.linspace(0, 1, test_n),
    torch.linspace(1, 2, test_n),
)
Ax_t, Ay_t, Az_t, Jx_t, gauge_t, Bx_t, By_t, Bz_t = make_analytical(test_grid)
test_i = torch.ones_like(test_grid[:, 0])

def make_test_x(n_tasks):
    return torch.cat([
        torch.stack([test_grid[:, 0], test_grid[:, 1], test_grid[:, 2], i * test_i], dim=-1)
        for i in range(n_tasks)
    ], dim=0)

# GT test:   tasks 0–7 → [A1, A2, A3, Jx, gauge, Bx, By, Bz]
test_x_GT   = make_test_x(8)
test_y_GT   = torch.cat([Ax_t, Ay_t, Az_t, Jx_t, gauge_t, Bx_t, By_t, Bz_t], dim=0)

# PIGP test: tasks 0–8 → [A1, A2, A3, Jx, Jy, Jz, Bx, By, Bz]
test_x_PIGP = make_test_x(9)
test_y_PIGP = torch.cat([Ax_t, Ay_t, Az_t, Jx_t, gauge_t, gauge_t, Bx_t, By_t, Bz_t], dim=0)

# --- Parameters ---
sigma_list   = [0.001, 0.01, 0.1, 0.2]
npoints_list = [7]

# Zero-task grid size: set to an int to decouple from the observation grid,
# or leave as None to use the same npoints.
mode = "extra_npoints"
n_zero = None
if mode == "extra_npoints":
    n_zero = 10


input_dir = os.path.join(os.path.dirname(__file__), "input_data")
os.makedirs(input_dir, exist_ok=True)

for sigma in sigma_list:
    for npoints in npoints_list:
        nz = n_zero if n_zero is not None else npoints

        # Observation grid (npoints^3): Jx and H field
        obs_grid = torch.cartesian_prod(
            torch.linspace(0, 1, npoints),
            torch.linspace(0, 1, npoints),
            torch.linspace(1, 2, npoints),
        )
        _, _, _, Jx_obs, _, Bx_obs, By_obs, Bz_obs = make_analytical(obs_grid)
        Jx_noisy = Jx_obs + sigma * torch.randn_like(Jx_obs)
        Bx_noisy = Bx_obs + sigma * torch.randn_like(Bx_obs)
        By_noisy = By_obs + sigma * torch.randn_like(By_obs)
        Bz_noisy = Bz_obs + sigma * torch.randn_like(Bz_obs)

        # Zero-task grid (nz^3; may differ from obs_grid)
        zero_grid = torch.cartesian_prod(
            torch.linspace(0, 1, nz),
            torch.linspace(0, 1, nz),
            torch.linspace(1, 2, nz),
        )
        zeros = torch.zeros_like(zero_grid[:, 0])

        # --- GT ---
        # Tasks: 3=Jx obs, 4=gauge=0, 5=Bx obs, 6=By obs, 7=Bz obs
        train_x_GT = torch.cat([obs_grid, zero_grid, obs_grid, obs_grid, obs_grid], dim=0)
        train_i_GT = torch.cat([
            torch.ones_like(obs_grid[:, 0])  * 3,
            torch.ones_like(zero_grid[:, 0]) * 4,
            torch.ones_like(obs_grid[:, 0])  * 5,
            torch.ones_like(obs_grid[:, 0])  * 6,
            torch.ones_like(obs_grid[:, 0])  * 7,
        ], dim=0)
        full_train_x_GT = torch.stack(
            [train_x_GT[:, 0], train_x_GT[:, 1], train_x_GT[:, 2], train_i_GT], dim=-1
        )
        train_y_GT = torch.cat([Jx_noisy, zeros, Bx_noisy, By_noisy, Bz_noisy], dim=0)

        # --- PIGP ---
        # Tasks: 3=Jx obs, 4=Jy=0, 5=Jz=0, 6=Bx obs, 7=By obs, 8=Bz obs
        train_x_PIGP = torch.cat([obs_grid, zero_grid, zero_grid, obs_grid, obs_grid, obs_grid], dim=0)
        train_i_PIGP = torch.cat([
            torch.ones_like(obs_grid[:, 0])  * 3,
            torch.ones_like(zero_grid[:, 0]) * 4,
            torch.ones_like(zero_grid[:, 0]) * 5,
            torch.ones_like(obs_grid[:, 0])  * 6,
            torch.ones_like(obs_grid[:, 0])  * 7,
            torch.ones_like(obs_grid[:, 0])  * 8,
        ], dim=0)
        full_train_x_PIGP = torch.stack(
            [train_x_PIGP[:, 0], train_x_PIGP[:, 1], train_x_PIGP[:, 2], train_i_PIGP], dim=-1
        )
        train_y_PIGP = torch.cat([Jx_noisy, zeros, zeros, Bx_noisy, By_noisy, Bz_noisy], dim=0)

        np.savez_compressed(
            os.path.join(input_dir, mode, "Ex3_GT_n%i_sigma%.3f.npz" % (npoints, sigma)),
            train_x=full_train_x_GT.numpy(),
            train_y=train_y_GT.numpy(),
            test_x=test_x_GT.numpy(),
            test_y=test_y_GT.numpy(),
        )
        print("GT   saved: npoints=%i, sigma=%.3f" % (npoints, sigma))

        np.savez_compressed(
            os.path.join(input_dir, mode, "Ex3_PIGP_n%i_sigma%.3f.npz" % (npoints, sigma)),
            train_x=full_train_x_PIGP.numpy(),
            train_y=train_y_PIGP.numpy(),
            test_x=test_x_PIGP.numpy(),
            test_y=test_y_PIGP.numpy(),
        )
        print("PIGP saved: npoints=%i, sigma=%.3f" % (npoints, sigma))
