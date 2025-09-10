import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
epsilon = 1e-4
r_max = 1_000_000.0
m = 1.0
sigma = 0.065

# Model definition (must match the original)
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.Sigmoid(),
            nn.Linear(128, 128), nn.Sigmoid(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.net.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, r):
        return self.net(r)

# Exact solution
def u_exact(r, m):
    return 2 * m * (1 + r) / (1 + (1 + r)**2)

# Recreate and load model
model = FCNet().double().to(device)
model.load_state_dict(torch.load('pinns_test4_weights.pth', map_location=device))
model.eval()

# Reconstruct domain
x = torch.linspace(epsilon, 1.0, 128, dtype=torch.float64, device=device).unsqueeze(1)
r = r_max * torch.sinh(x / sigma) / torch.sinh(torch.tensor(1.0 / sigma, dtype=torch.float64, device=device))

# Test on wide domain
r_test = torch.linspace(epsilon, r_max, 500, dtype=torch.float64, device=device).unsqueeze(1)
u_exact_vals = u_exact(r_test.cpu().numpy(), m)

with torch.no_grad():
    u_pred = model(r_test).cpu().numpy()

avg_pointwise_error = np.mean(np.abs(u_pred - u_exact_vals))
print(f"\nAverage point-wise error: {avg_pointwise_error:.6e}")

# Plot
plt.figure(figsize=(18, 4))

# Plot placeholder for losses (not available here)
plt.subplot(1, 3, 1)
plt.title('Training Loss History (Not Available)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()

# Plot full domain
plt.subplot(1, 3, 2)
plt.plot(r_test.cpu(), u_exact_vals, label='Exact', linewidth=2)
plt.plot(r_test.cpu(), u_pred, '--', label='PINN', linewidth=2)
plt.plot(r.cpu(), 0 * r.cpu(), 'o', label='Train Points')
plt.xlabel('r')
plt.ylabel('u(r)')
plt.title('Exact vs PINN Solution')
plt.legend()
plt.grid()
plt.xlim(0, r_max)

# Plot zoomed in region
r_test2 = torch.linspace(epsilon, 10, 500, dtype=torch.float64, device=device).unsqueeze(1)
u_exact_vals2 = u_exact(r_test2.cpu().numpy(), m)
with torch.no_grad():
    u_pred2 = model(r_test2).cpu().numpy()

plt.subplot(1, 3, 3)
plt.plot(r_test2.cpu(), u_exact_vals2, label='Exact', linewidth=2)
plt.plot(r_test2.cpu(), u_pred2, '--', label='PINN', linewidth=2)
plt.plot(r.cpu(), 0 * r.cpu(), 'o', label='Train Points')
plt.xlabel('r')
plt.ylabel('u(r)')
plt.title('Zoomed-In: Exact vs PINN')
plt.legend()
plt.grid()
plt.xlim(0, 10)

plt.tight_layout()
plt.savefig("pinns_toy_final_plot.pdf")
plt.show()
