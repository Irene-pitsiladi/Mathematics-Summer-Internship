{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5ad72de2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Using device: cpu\n"
     ]
    },
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'pinns_test4_weights.pth'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 39\u001b[0m\n\u001b[0;32m     37\u001b[0m \u001b[38;5;66;03m# Load model and weights\u001b[39;00m\n\u001b[0;32m     38\u001b[0m model \u001b[38;5;241m=\u001b[39m FCNet()\u001b[38;5;241m.\u001b[39mdouble()\u001b[38;5;241m.\u001b[39mto(device)\n\u001b[1;32m---> 39\u001b[0m model\u001b[38;5;241m.\u001b[39mload_state_dict(\u001b[43mtorch\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mload\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mpinns_test4_weights.pth\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmap_location\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdevice\u001b[49m\u001b[43m)\u001b[49m)\n\u001b[0;32m     40\u001b[0m model\u001b[38;5;241m.\u001b[39meval()\n\u001b[0;32m     42\u001b[0m \u001b[38;5;66;03m# Generate test points on the full domain\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\44783\\anaconda3\\envs\\Summerinternship\\lib\\site-packages\\torch\\serialization.py:1479\u001b[0m, in \u001b[0;36mload\u001b[1;34m(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)\u001b[0m\n\u001b[0;32m   1476\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mencoding\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m pickle_load_args\u001b[38;5;241m.\u001b[39mkeys():\n\u001b[0;32m   1477\u001b[0m     pickle_load_args[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mencoding\u001b[39m\u001b[38;5;124m\"\u001b[39m] \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mutf-8\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m-> 1479\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m \u001b[43m_open_file_like\u001b[49m\u001b[43m(\u001b[49m\u001b[43mf\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mrb\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mas\u001b[39;00m opened_file:\n\u001b[0;32m   1480\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m _is_zipfile(opened_file):\n\u001b[0;32m   1481\u001b[0m         \u001b[38;5;66;03m# The zipfile reader is going to advance the current file position.\u001b[39;00m\n\u001b[0;32m   1482\u001b[0m         \u001b[38;5;66;03m# If we want to actually tail call to torch.jit.load, we need to\u001b[39;00m\n\u001b[0;32m   1483\u001b[0m         \u001b[38;5;66;03m# reset back to the original position.\u001b[39;00m\n\u001b[0;32m   1484\u001b[0m         orig_position \u001b[38;5;241m=\u001b[39m opened_file\u001b[38;5;241m.\u001b[39mtell()\n",
      "File \u001b[1;32mc:\\Users\\44783\\anaconda3\\envs\\Summerinternship\\lib\\site-packages\\torch\\serialization.py:759\u001b[0m, in \u001b[0;36m_open_file_like\u001b[1;34m(name_or_buffer, mode)\u001b[0m\n\u001b[0;32m    757\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21m_open_file_like\u001b[39m(name_or_buffer: FileLike, mode: \u001b[38;5;28mstr\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m _opener[IO[\u001b[38;5;28mbytes\u001b[39m]]:\n\u001b[0;32m    758\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m _is_path(name_or_buffer):\n\u001b[1;32m--> 759\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43m_open_file\u001b[49m\u001b[43m(\u001b[49m\u001b[43mname_or_buffer\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmode\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    760\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    761\u001b[0m         \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mw\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01min\u001b[39;00m mode:\n",
      "File \u001b[1;32mc:\\Users\\44783\\anaconda3\\envs\\Summerinternship\\lib\\site-packages\\torch\\serialization.py:740\u001b[0m, in \u001b[0;36m_open_file.__init__\u001b[1;34m(self, name, mode)\u001b[0m\n\u001b[0;32m    739\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28mself\u001b[39m, name: Union[\u001b[38;5;28mstr\u001b[39m, os\u001b[38;5;241m.\u001b[39mPathLike[\u001b[38;5;28mstr\u001b[39m]], mode: \u001b[38;5;28mstr\u001b[39m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 740\u001b[0m     \u001b[38;5;28msuper\u001b[39m()\u001b[38;5;241m.\u001b[39m\u001b[38;5;21m__init__\u001b[39m(\u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mname\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mmode\u001b[49m\u001b[43m)\u001b[49m)\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'pinns_test4_weights.pth'"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import numpy as np\n",
    "\n",
    "# Device config\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Using device: {device}\")\n",
    "\n",
    "# Constants\n",
    "epsilon = 1e-4\n",
    "r_max = 1_000_000.0\n",
    "m = 1.0\n",
    "\n",
    "# Define the model architecture (must match your saved model!)\n",
    "class FCNet(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(1, 128), nn.Sigmoid(),\n",
    "            nn.Linear(128, 128), nn.Sigmoid(),\n",
    "            nn.Linear(128, 1), nn.Sigmoid()\n",
    "        )\n",
    "        self.net.apply(self._init_weights)\n",
    "\n",
    "    def _init_weights(self, layer):\n",
    "        if isinstance(layer, nn.Linear):\n",
    "            nn.init.xavier_uniform_(layer.weight)\n",
    "            nn.init.zeros_(layer.bias)\n",
    "\n",
    "    def forward(self, r):\n",
    "        return self.net(r)\n",
    "\n",
    "# Exact solution function\n",
    "def u_exact(r, m):\n",
    "    return 2 * m * (1 + r) / (1 + (1 + r)**2)\n",
    "\n",
    "# Load model and weights\n",
    "model = FCNet().double().to(device)\n",
    "model.load_state_dict(torch.load('pinns_test4_weights.pth', map_location=device))\n",
    "model.eval()\n",
    "\n",
    "# Generate test points on the full domain\n",
    "r_test = torch.linspace(epsilon, r_max, 500, dtype=torch.float64, device=device).unsqueeze(1)\n",
    "u_exact_vals = u_exact(r_test.cpu().numpy(), m)\n",
    "\n",
    "# Predict with model\n",
    "with torch.no_grad():\n",
    "    u_pred = model(r_test).cpu().numpy()\n",
    "\n",
    "# Calculate errors\n",
    "pointwise_error = np.abs(u_pred - u_exact_vals)\n",
    "avg_error = np.mean(pointwise_error)\n",
    "max_error = np.max(pointwise_error)\n",
    "\n",
    "print(f\"Average pointwise error: {avg_error:.6e}\")\n",
    "print(f\"Maximum pointwise error: {max_error:.6e}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Summerinternship",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
