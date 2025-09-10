import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(123)

# Parameters
epsilon = 8e-5
r_max = 30000
sigma = 0.065
m = 0.5
S = 0.2 * (m**2)
epochs = 20000
N_r = 128
N_theta = 128
LR = 5e-5
GAMMA = 1.0
theta_min = 1e-5
theta_max = np.pi - 1e-5
bc_weight= 0.55

# Neural Network
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Sigmoid(),
            nn.Linear(64, 64), nn.Sigmoid(),
            nn.Linear(64, 64), nn.Sigmoid(),
            nn.Linear(64, 64), nn.Sigmoid(),
            nn.Linear(64, 64), nn.Sigmoid(),
            nn.Linear(64, 1)  # No sigmoid on last layer
        )
        self.net.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)

# Functions
def A_func(r, theta, S):
    return 18 * S**2 * torch.sin(theta)**2 / (r**6)

def psi_func(r, m):
    return 1.0 + m / (2.0 * r)

def compute_residual(model, r, theta):
    r.requires_grad_(True)
    theta.requires_grad_(True)
    inputs = torch.cat([r, theta], dim=1)
    u = model(inputs)

    u_r = torch.autograd.grad(u, r, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_rr = torch.autograd.grad(u_r, r, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_theta = torch.autograd.grad(u, theta, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_thetatheta = torch.autograd.grad(u_theta, theta, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    sin_theta = torch.sin(theta)
    sin2_theta = torch.sin(2 * theta)

    laplacian = (sin_theta**2 * r**2 * u_rr +
                 2 * sin_theta**2 * r * u_r +
                 (sin2_theta / 2) * u_theta +
                 sin_theta**2 * u_thetatheta)

    A_val = A_func(r, theta, S)
    psi_val = psi_func(r, m)
    residual = laplacian + r**2 * sin_theta**2 * A_val / (8 * (psi_val + u)**7)
    return residual

# Loss
def pinn_loss(model, r, theta, r_boundary, theta_boundary):
    residual = compute_residual(model, r, theta)
    loss_pde = residual.pow(2).mean()

    u_boundary = model(torch.cat([r_boundary, theta_boundary], dim=1))
    loss_bc = u_boundary.pow(2).mean()

    return loss_pde + bc_weight * loss_bc, loss_pde, loss_bc

# Sampling
x_r = torch.linspace(epsilon, 1.0, N_r, dtype=torch.float64, device=device).unsqueeze(1)
r_vals = r_max * torch.sinh(x_r / sigma) / torch.sinh(torch.tensor(1.0 / sigma, dtype=torch.float64, device=device))
theta_vals = torch.linspace(theta_min, theta_max, N_theta, dtype=torch.float64, device=device).unsqueeze(1)

r_grid, theta_grid = torch.meshgrid(r_vals.squeeze(), theta_vals.squeeze(), indexing='ij')
r = r_grid.reshape(-1,1)
theta = theta_grid.reshape(-1,1)

r_boundary = torch.tensor([[r_max]], dtype=torch.float64, device=device).repeat(N_theta,1)
theta_boundary = theta_vals.repeat(1,1)

# Model and optimizer
model = FCNet().double().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

# Training loop
loss_history = []
pde_loss_history = []
bc_loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    total_loss, pde_loss, bc_loss = pinn_loss(model, r, theta, r_boundary, theta_boundary)
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(total_loss.item())
    pde_loss_history.append(pde_loss.item())
    bc_loss_history.append(bc_loss.item())

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Total: {total_loss:.2e} | PDE: {pde_loss:.2e} | BC: {bc_loss:.2e}")

# Save model
torch.save(model.state_dict(), "pinns_testbc8_model.pth")

# ----------- PLOTTING -----------

fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# --- Plot 1: Loss History ---
axs[0].semilogy(loss_history, label='Total Loss')
axs[0].semilogy(pde_loss_history, label='PDE Loss')
axs[0].semilogy(bc_loss_history, label='BC Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Loss History')
axs[0].legend()
axs[0].grid()

# --- Plot 2: Cartesian Coordinates of Training Points ---
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

# --- Plot 3: Overlay of Exact vs PINN on z-axis ---
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
    uu_pinn = model(torch.cat([r_tensor, theta_tensor], dim=1)).cpu().numpy().flatten()

axs[2].plot(z_pos, uu_pos, label="Exact solution (shifted)", color="black", linewidth=2)
axs[2].plot(z_pos, uu_pinn, label="PINN prediction", color="red", linestyle="--", linewidth=2)
axs[2].set_xlabel("r = z - 2.5")
axs[2].set_ylabel("u(r, θ=0)")
axs[2].set_title("Exact vs PINN Solution Along the z-axis (Shifted)")
axs[2].grid(True)
axs[2].legend()

# plt.tight_layout()
plt.savefig("pinns_testbc7_plots.pdf")
plt.show()
