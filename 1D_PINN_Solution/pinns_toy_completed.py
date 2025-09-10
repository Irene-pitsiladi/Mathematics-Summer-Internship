#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device configuration: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Seed for reproducibility
torch.manual_seed(123)


epsilon = 1e-4
r_max =1000000.0
m = 1.0
sigma = 0.065 
epochs = 100000
N = 128
LR     = 1.e-5
GAMMA = 1.0 #0.9998 #1.0


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.Sigmoid(),
            nn.Linear(128, 128), nn.Sigmoid(),

            nn.Linear(128, 1),nn.Sigmoid()
        )
        self.net.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)  # xavier works best with tanh
            nn.init.zeros_(layer.bias)

    def forward(self, r):
        return self.net(r)


def A(r, m):
    r = r.to(dtype=torch.float64)
    A_val=m*(6 + 6*r + r**2)*(2*r*(2 + 2*r + r**2) + m*(2 + 6*r + 5*r**2))**7 / (32*(r**7) * (2 + 2*r + r**2)**(10))
    return A_val




def compute_pde_residual(model, r, m):
    r = r.requires_grad_(True)
    u = model(r)
    u_r = torch.autograd.grad(u, r, grad_outputs=torch.ones_like(u), create_graph=True)[0] #create graph to compute 2nd deriv
    u_rr = torch.autograd.grad(u_r, r, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]

    
    A_val = A(r, m)

    base = 1 + m / (2 * r) + u
    

    residual = r* u_rr + 2  * u_r + r * A_val * base.pow(-7)

    return residual


def pinn_loss(model, r_interior, r_boundary, m):
    residual = compute_pde_residual(model, r_interior, m)
    loss_pde = residual.pow(2).mean()

    # Boundary losses
    loss_bc = model(r_boundary).pow(2).mean() 

    return loss_pde + 1e-2*loss_bc, loss_pde, loss_bc


x = torch.linspace(epsilon, 1.0, N, dtype=torch.float64, device=device).unsqueeze(1)
r = r_max * torch.sinh(x / sigma) / torch.sinh(torch.tensor(1.0 / sigma, dtype=torch.float64, device=device))



r_boundary = torch.tensor([[r_max]], dtype=torch.float64, device=device)

model = FCNet().double().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=GAMMA)

def u_exact(r, m):
    return 2 * m * (1 + r) / (1 + (1 + r) ** 2)

def loss_LSQ(r,m):
    u = model(r)
    #
    residual = u - u_exact(r,m)
    return residual

# Print sampling stats before training loop

print(f"Sampling range: epsilon={epsilon}, r_max={r_max}")
print(f"r_interior min: {r.min().item()}, max: {r.max().item()}")

loss_history = []
pde_loss_history = []
bc_loss_history = []


for epoch in range(epochs):
    model.train()
    

    optimizer.zero_grad()
    total_loss, pde_loss, bc_loss = pinn_loss(model, r, r_boundary, m)
    total_loss.backward(retain_graph = True)

    optimizer.step()
    scheduler.step()

    loss_history.append(total_loss.item())
    pde_loss_history.append(pde_loss.item())
    bc_loss_history.append(bc_loss.item())

    if epoch % 1000 == 0:
        lr = scheduler.get_last_lr()[0]
        loss_lsq = torch.mean(loss_LSQ(r,m)**2)
        print(f"Epoch {epoch} | LR: {lr:.1e} | LSQ:  {loss_lsq.item():>1.3e} |Total: {total_loss:.1e} | PDE: {pde_loss:.1e} | BC: {bc_loss:.1e}")
torch.save(model.state_dict(), 'pinns_test4_weights.pth')


# Plot losses
plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plt.semilogy(loss_history, label='Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.legend()
plt.grid()





r_test = torch.linspace(epsilon, r_max, 500, dtype=torch.float64, device=device).unsqueeze(1)
u_exact_vals = u_exact(r_test.cpu().numpy(), m)

model.eval()
with torch.no_grad():
    u_pred = model(r_test).cpu().numpy()
avg_pointwise_error = np.mean(np.abs(u_pred - u_exact_vals))
print(f"\nAverage point-wise error: {avg_pointwise_error:.6e}")

plt.subplot(1, 3, 2)
plt.plot(r_test.cpu(), u_exact_vals, label='Exact', linewidth=2)
plt.plot(r_test.cpu(), u_pred, '--', label='PINN', linewidth=2)
plt.plot(r.cpu().detach(),0*r.cpu().detach(),'o')
plt.xlabel('r')
plt.ylabel('u(r)')
plt.title('Exact vs PINN Solution')
plt.legend()
plt.grid()
plt.xlim(0, r_max)


r_test2 = torch.linspace(epsilon, 10, 500, dtype=torch.float64, device=device).unsqueeze(1)
u_exact_vals2 = u_exact(r_test2.cpu().numpy(), m)
with torch.no_grad():
    u_pred2 = model(r_test2).cpu().numpy()
plt.subplot(1,3,3)
plt.plot(r_test2.cpu(), u_exact_vals2, label='Exact', linewidth=2)
plt.plot(r_test2.cpu(), u_pred2, '--', label='PINN', linewidth=2)
plt.plot(r.cpu().detach(),0*r.cpu().detach(),'o')
plt.xlabel('r')
plt.ylabel('u(r)')
plt.title('Exact vs PINN Solution')
plt.legend()
plt.grid()
plt.xlim(0, 10)



plt.tight_layout()
plt.savefig("pinns_test4_plot.pdf")
plt.show()





