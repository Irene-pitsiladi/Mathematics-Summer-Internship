import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(123)

# Parameters
epsilon = 1e-4
r_max = 1000000.0
m = 1.0
sigma = 0.065

# Model definition
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
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
    return 2 * m * (1 + r) / (1 + (1 + r) ** 2)

# Instantiate and load model
model = FCNet().double().to(device)
model.load_state_dict(torch.load('deepritz_linspace9_weights.pth'))
model.eval()

# Evaluation
r_test = torch.linspace(epsilon, r_max, 500, dtype=torch.float64, device=device).unsqueeze(1)
u_exact_vals = u_exact(r_test.cpu().numpy(), m)

with torch.no_grad():
    u_pred = model(r_test).cpu().numpy()

# Compute pointwise error
pointwise_error = np.abs(u_pred - u_exact_vals)

# Log-scale plot of the pointwise error
plt.figure(figsize=(8, 6))
plt.loglog(r_test.cpu(), pointwise_error, color='darkred', linewidth=2)
plt.xlabel('r')
plt.ylabel('Absolute Error (log scale)')
plt.title('Log Pointwise Error: |u_pred - u_exact|')
plt.grid(True)
plt.xlim(0, r_max)
plt.tight_layout()
plt.savefig("deepritz_loglog_error.pdf")
plt.show()
