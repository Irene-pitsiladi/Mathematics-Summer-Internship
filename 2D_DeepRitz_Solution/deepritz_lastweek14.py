#more points than lastweek13

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(123)

# Parameters
epsilon = 1e-4
r_max = 30000
sigma = 0.065
m = 0.5
S = 0.2 * (m**2)
epochs = 10000
N_r = 160
N_theta = 160
LR = 1e-5
GAMMA = 0.95
theta_min = 1e-5
theta_max = np.pi - 1e-5
bc_weight = 5000

# Network
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.net.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

# PDE terms
def A_func(r, theta, S):
    return 18 * S**2 * torch.sin(theta)**2 / (r**6)

def psi_func(r, m):
    return 1.0 + m / (2.0 * r)

# DeepRitz energy functional
def compute_energy(model, r, theta):
    r.requires_grad_(True)
    theta.requires_grad_(True)

    inputs = torch.cat([r, theta], dim=1)
    u = model(inputs)

    u_r = torch.autograd.grad(u, r, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_theta = torch.autograd.grad(u, theta, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    grad_r_sq = u_r**2
    grad_theta_sq = u_theta**2

    sin_theta = torch.sin(theta)
    r_sq = r**2

    A_val = A_func(r, theta, S)
    psi_val = psi_func(r, m)

    energy_density = (0.5 * (grad_r_sq + grad_theta_sq / (r_sq + 1e-14)) + 
                      (1.0 / 6.0) * A_val * (psi_val + u) ** (-6)) * (r**2 * sin_theta)

    sinh_1_over_sigma = torch.sinh(torch.tensor(1.0 / sigma, dtype=torch.float64, device=device))
    numerator = sigma * sinh_1_over_sigma
    denominator = r_max * torch.sqrt(1 + ((r / r_max) * sinh_1_over_sigma).pow(2))
    p_r = numerator / denominator

    energy_term = energy_density / p_r
    return energy_term

# Total loss
def deepritz_loss(model, r, theta, r_boundary, theta_boundary, r_soft, theta_soft):
    u_boundary = model(torch.cat([r_boundary, theta_boundary], dim=1))
    loss_bc = u_boundary.pow(2).mean()

    u_soft = model(torch.cat([r_soft, theta_soft], dim=1))
    loss_soft = u_soft.pow(2).mean()

    energy_term = compute_energy(model, r, theta)
    energy_loss = energy_term.mean()

    return energy_loss + bc_weight * (loss_bc + 0.5 * loss_soft), energy_loss, loss_bc

# ---- Sampling ----
# Radial domain
x_r = torch.linspace(epsilon, 1, N_r, device=device, dtype=torch.float64)
r_vals = r_max * torch.sinh(x_r / sigma) / torch.sinh(torch.tensor(1.0 / sigma, dtype=torch.float64, device=device))

# Angular domain
theta_vals = torch.linspace(theta_min, theta_max, N_theta, dtype=torch.float64, device=device).unsqueeze(1)

# Full grid
r_grid, theta_grid = torch.meshgrid(r_vals.squeeze(), theta_vals.squeeze(), indexing='ij')
r = r_grid.reshape(-1, 1)
theta = theta_grid.reshape(-1, 1)

# Hard boundary condition: r = r_max
r_boundary = torch.tensor([[r_max]], dtype=torch.float64, device=device).repeat(N_theta, 1)
theta_boundary = theta_vals.reshape(-1, 1)

# Soft boundary: r values near r_max
n_rb_extra = 10
r_soft_vals = torch.linspace(0.95 * r_max, r_max, n_rb_extra, device=device, dtype=torch.float64).unsqueeze(1)
r_soft = r_soft_vals.repeat(1, N_theta).T.reshape(-1, 1)
theta_soft = theta_vals.repeat(n_rb_extra, 1)

# Model and optimizer
model = FCNet().double().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=GAMMA)

# ---- Training ----
loss_history = []
energy_history = []
bc_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    total_loss, energy_loss, bc_loss = deepritz_loss(
        model, r, theta, r_boundary, theta_boundary, r_soft, theta_soft
    )

    total_loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(total_loss.item())
    energy_history.append(energy_loss.item())
    bc_history.append(bc_loss.item())

    if epoch % 1000 == 0 or epoch == epochs - 1:
        with torch.no_grad():
            u_soft_val = model(torch.cat([r_soft, theta_soft], dim=1))
            print(f"Epoch {epoch} | Total Loss: {total_loss:.4e} | Energy: {energy_loss:.4e} | "
                  f"BC Loss: {bc_loss:.4e} | Max |u_soft|: {u_soft_val.abs().max():.2e}")

torch.save(model.state_dict(), "deepritz_lastweek14_model.pth")

# ---- Visualization ----
fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# Plot 1: Loss curves
axs[0].semilogy(loss_history, label='Total Loss')
axs[0].semilogy(energy_history, label='Energy Loss')
axs[0].semilogy(bc_history, label='Hard BC Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Loss History')
axs[0].legend()
axs[0].grid()

# Plot 2: Training points in Cartesian coords
r_cpu = r.detach().cpu().numpy().flatten()
theta_cpu = theta.detach().cpu().numpy().flatten()
x = r_cpu * np.sin(theta_cpu)
y = r_cpu * np.cos(theta_cpu)
axs[1].scatter(x, y, s=1, color='black', alpha=0.5)
axs[1].set_xlabel('x = r sin(θ)')
axs[1].set_ylabel('y = r cos(θ)')
axs[1].set_title('Training Points in Cartesian Coordinates')
axs[1].set_aspect('equal')
axs[1].grid(True)

# Plot 3: Exact vs DeepRitz on z-axis
filename = "cart_coords_z_axis_interpolated_uu_data.txt"
data = np.loadtxt(filename)
z = data[:, 2]
uu = data[:, 3]
z_shifted = z - 2.5
mask_positive = z_shifted > 0
z_pos = z_shifted[mask_positive]
uu_pos = uu[mask_positive]

r_tensor = torch.tensor(z_pos.reshape(-1, 1), dtype=torch.float64, device=device)
theta_tensor = torch.zeros_like(r_tensor)

with torch.no_grad():
    uu_deepritz = model(torch.cat([r_tensor, theta_tensor], dim=1)).cpu().numpy().flatten()

axs[2].plot(z_pos, uu_pos, label="Exact solution (shifted)", color="black", linewidth=2)
axs[2].plot(z_pos, uu_deepritz, label="DeepRitz prediction", color="red", linestyle="--", linewidth=2)
axs[2].set_xlabel("r = z - 2.5")
axs[2].set_ylabel("u(r, θ=0)")
axs[2].set_title("Exact vs DeepRitz Solution Along the z-axis (Shifted)")
axs[2].grid(True)
axs[2].legend()

plt.savefig("deepritz_lastweek14_plots.pdf")
plt.show()
