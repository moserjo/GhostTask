import os
import torch
import numpy as np

torch.set_default_dtype(torch.float64)

n = 6
x = torch.linspace(0, 1, n)
y = torch.linspace(0, 1, n)
z = torch.linspace(1, 2, n)
full_grid = torch.cartesian_prod(x, y, z)
xx = full_grid[:, 0]
yy = full_grid[:, 1]

# Boundary points: faces where each A component has Dirichlet BC = 0
train_x_0 = full_grid[((full_grid[:, 1] == 0) | (full_grid[:, 1] == 1) | (full_grid[:, 2] == 1) | (full_grid[:, 2] == 2))]
train_x_1 = full_grid[((full_grid[:, 0] == 0) | (full_grid[:, 0] == 1) | (full_grid[:, 2] == 1) | (full_grid[:, 2] == 2))]
train_x_2 = full_grid[((full_grid[:, 0] == 0) | (full_grid[:, 0] == 1) | (full_grid[:, 1] == 0) | (full_grid[:, 1] == 1))]

Jz = 2 * torch.pi**2 * torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
zeros_grid = torch.zeros_like(full_grid[:, 0])

train_i_0 = torch.zeros_like(train_x_0[:, 0])
train_i_1 = torch.ones_like(train_x_1[:, 0])
train_i_2 = torch.ones_like(train_x_2[:, 0]) * 2

# --- GT ---
# Tasks: 0=A1 BC, 1=A2 BC, 2=A3 BC, 3=Jz source, 4=gauge (=0)
train_x_GT = torch.cat([train_x_0, train_x_1, train_x_2, full_grid, full_grid], dim=0)
train_i_GT = torch.cat([train_i_0, train_i_1, train_i_2,
                         torch.ones_like(full_grid[:, 0]) * 3,
                         torch.ones_like(full_grid[:, 0]) * 4], dim=0)
full_train_x_GT = torch.stack([train_x_GT[:, 0], train_x_GT[:, 1], train_x_GT[:, 2], train_i_GT], dim=-1)
train_y_GT = torch.cat([
    torch.zeros_like(train_x_0[:, 0]),
    torch.zeros_like(train_x_1[:, 0]),
    torch.zeros_like(train_x_2[:, 0]),
    Jz,
    zeros_grid,
], dim=0)

# --- PIGP ---
# Tasks: 0=A1 BC, 1=A2 BC, 2=A3 BC, 3=Jx (=0), 4=Jy (=0), 5=Jz source
train_x_PIGP = torch.cat([train_x_0, train_x_1, train_x_2, full_grid, full_grid, full_grid], dim=0)
train_i_PIGP = torch.cat([train_i_0, train_i_1, train_i_2,
                            torch.ones_like(full_grid[:, 0]) * 3,
                            torch.ones_like(full_grid[:, 0]) * 4,
                            torch.ones_like(full_grid[:, 0]) * 5], dim=0)
full_train_x_PIGP = torch.stack([train_x_PIGP[:, 0], train_x_PIGP[:, 1], train_x_PIGP[:, 2], train_i_PIGP], dim=-1)
train_y_PIGP = torch.cat([
    torch.zeros_like(train_x_0[:, 0]),
    torch.zeros_like(train_x_1[:, 0]),
    torch.zeros_like(train_x_2[:, 0]),
    zeros_grid,   # Jx = 0
    zeros_grid,   # Jy = 0
    Jz,
], dim=0)

# --- Test grid (analytical solution) ---
test_n = 2 * n
test_grid = torch.cartesian_prod(
    torch.linspace(0, 1, test_n),
    torch.linspace(0, 1, test_n),
    torch.linspace(1, 2, test_n),
)
tx = test_grid[:, 0]
ty = test_grid[:, 1]
test_grid_i = torch.ones_like(test_grid[:, 0])

Ax       = torch.zeros_like(tx)
Ay       = torch.zeros_like(tx)
Az       = torch.sin(torch.pi * tx) * torch.sin(torch.pi * ty)
Jz_test  = 2 * torch.pi**2 * torch.sin(torch.pi * tx) * torch.sin(torch.pi * ty)
GT_test  = torch.zeros_like(tx)
Bx       = torch.pi * torch.cos(torch.pi * ty) * torch.sin(torch.pi * tx)
By       = -torch.pi * torch.cos(torch.pi * tx) * torch.sin(torch.pi * ty)
Bz       = torch.zeros_like(tx)

def make_test_x(n_tasks):
    return torch.cat([
        torch.stack([test_grid[:, 0], test_grid[:, 1], test_grid[:, 2], i * test_grid_i], dim=-1)
        for i in range(n_tasks)
    ], dim=0)

# GT test: tasks 0–7 → [A1, A2, A3, Jz, gauge, Bx, By, Bz]
test_x_GT = make_test_x(8)
test_y_GT = torch.cat([Ax, Ay, Az, Jz_test, GT_test, Bx, By, Bz], dim=0)

# PIGP test: tasks 0–8 → [A1, A2, A3, Jx, Jy, Jz, Bx, By, Bz]
test_x_PIGP = make_test_x(9)
test_y_PIGP = torch.cat([Ax, Ay, Az, GT_test, GT_test, Jz_test, Bx, By, Bz], dim=0)

# --- Save ---
input_dir = os.path.join(os.path.dirname(__file__), "input_data")
os.makedirs(input_dir, exist_ok=True)

np.savez_compressed(
    os.path.join(input_dir, "Ex3_GT.npz"),
    train_x=full_train_x_GT.numpy(),
    train_y=train_y_GT.numpy(),
    test_x=test_x_GT.numpy(),
    test_y=test_y_GT.numpy(),
)
print("GT saved")

np.savez_compressed(
    os.path.join(input_dir, "Ex3_PIGP.npz"),
    train_x=full_train_x_PIGP.numpy(),
    train_y=train_y_PIGP.numpy(),
    test_x=test_x_PIGP.numpy(),
    test_y=test_y_PIGP.numpy(),
)
print("PIGP saved")
