"""
PINN for the Damped Harmonic Oscillator
========================================
Equation:  d²x/dz² + 2ξ·dx/dz + x = 0
Domain:    z ∈ [0, 20],  ξ ∈ [0.1, 0.4]
ICs:       x(0) = 0.7,   dx/dz(0) = 1.2

File structure:
    pinn_oscillator.py   ← this file (model + training + evaluation)
    outputs/             ← plots saved here (auto-created)

Author: Binoy Saha
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# reproducibility
torch.manual_seed(42)
np.random.seed(42)

# auto-create output folder for plots
os.makedirs("outputs", exist_ok=True)

# use GPU if available, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
X0   = 0.7   # initial position  x(0)
V0   = 1.2   # initial velocity  dx/dz(0)
Z_MAX = 20.0  # end of time domain
XI_MIN = 0.1  # minimum damping ratio
XI_MAX = 0.4  # maximum damping ratio


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICAL SOLUTION (for validation only — PINN never sees this)
# ─────────────────────────────────────────────────────────────────────────────
def analytical_solution(z, xi):
    """
    Exact closed-form solution for the underdamped case (xi < 1).
    
    For xi < 1, the system oscillates with exponentially decaying amplitude.
    Solution:  x(z) = e^(-ξz) * [A*cos(ωz) + B*sin(ωz)]
    where ω = sqrt(1 - ξ²)  (damped natural frequency)
    
    A and B are solved from initial conditions x(0)=X0, x'(0)=V0.
    """
    xi  = np.array(xi)
    omega = np.sqrt(1 - xi**2)               # damped natural frequency
    A = X0                                   # from x(0) = X0
    B = (V0 + xi * X0) / omega              # from x'(0) = V0
    return np.exp(-xi * z) * (A * np.cos(omega * z) + B * np.sin(omega * z))


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    """
    Physics-Informed Neural Network for the damped harmonic oscillator.
    
    Input:  (z, ξ) — time and damping ratio   [batch_size, 2]
    Output: x(z, ξ) — predicted position       [batch_size, 1]
    
    Architecture: 4 hidden layers with tanh activations.
    tanh works better than ReLU for PINNs because:
      - it's smooth and infinitely differentiable (we need 2nd derivatives)
      - ReLU has zero 2nd derivative everywhere → physics loss can't train
    """
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        
        # build the layer list dynamically
        layers = []
        
        # input layer: 2 inputs (z, ξ) → hidden_dim
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())
        
        # hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # output layer: hidden_dim → 1 (the position x)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Xavier initialization — helps with gradient flow in deep nets
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, z, xi):
        """
        Forward pass.
        z  : [batch, 1] time points
        xi : [batch, 1] damping ratios
        returns x : [batch, 1] predicted positions
        """
        inp = torch.cat([z, xi], dim=1)  # concatenate inputs → [batch, 2]
        return self.net(inp)


# ─────────────────────────────────────────────────────────────────────────────
# DERIVATIVE HELPER (using PyTorch autograd)
# ─────────────────────────────────────────────────────────────────────────────
def gradient(output, input_var):
    """
    Compute d(output)/d(input_var) using automatic differentiation.
    
    This is the KEY trick in PINNs — PyTorch tracks every operation on
    tensors with requires_grad=True. grad() walks that computation graph
    backwards to get the exact derivative.
    
    create_graph=True means we can differentiate AGAIN (needed for x'')
    """
    return torch.autograd.grad(
        outputs=output,
        inputs=input_var,
        grad_outputs=torch.ones_like(output),  # chain rule seed
        create_graph=True,                      # allow higher-order derivatives
        retain_graph=True                       # keep graph for multiple calls
    )[0]


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(model, z_phys, xi_phys, z_ic, xi_ic):
    """
    Total PINN loss = physics_loss + initial_condition_loss
    
    physics_loss:
        Residual of the ODE at random collocation points.
        residual = x'' + 2ξx' + x  (should be 0 everywhere)
        
    ic_loss:
        x(0, ξ) should equal X0 = 0.7      (position IC)
        x'(0, ξ) should equal V0 = 1.2     (velocity IC)
    """
    
    # ── 1. PHYSICS LOSS ──────────────────────────────────────────────────────
    # we need autograd to work on z_phys, so mark it
    z_phys.requires_grad_(True)
    
    x = model(z_phys, xi_phys)           # forward pass → x(z, ξ)
    
    x_z  = gradient(x, z_phys)           # dx/dz   — first derivative
    x_zz = gradient(x_z, z_phys)         # d²x/dz² — second derivative
    
    # ODE residual: should be zero for a perfect solution
    residual = x_zz + 2 * xi_phys * x_z + x
    
    physics_loss = torch.mean(residual ** 2)   # MSE of residual
    
    # ── 2. INITIAL CONDITION LOSS ────────────────────────────────────────────
    z_ic.requires_grad_(True)
    
    x_ic   = model(z_ic, xi_ic)          # x(0, ξ) — position at t=0
    x_ic_z = gradient(x_ic, z_ic)        # dx/dz at t=0 — velocity at t=0
    
    ic_pos_loss = torch.mean((x_ic   - X0) ** 2)   # x(0) should = 0.7
    ic_vel_loss = torch.mean((x_ic_z - V0) ** 2)   # x'(0) should = 1.2
    
    ic_loss = ic_pos_loss + ic_vel_loss
    
    # ── 3. TOTAL LOSS ────────────────────────────────────────────────────────
    # weight IC loss higher — ICs are hard constraints, physics is softer
    total_loss = physics_loss + 10.0 * ic_loss
    
    return total_loss, physics_loss.item(), ic_loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(model, n_epochs=10000, n_collocation=2000, n_ic=200, lr=1e-3):
    """
    Train the PINN.
    
    Collocation points: random (z, ξ) samples from the domain.
        The network learns to satisfy the ODE at these points.
        More points = better coverage but slower per step.
    
    IC points: random ξ samples at z=0.
        The network learns to match initial conditions here.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # learning rate scheduler — reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-5
    )
    
    loss_history = []
    
    print(f"\nTraining PINN for {n_epochs} epochs...")
    print(f"Collocation points: {n_collocation} | IC points: {n_ic}")
    print("-" * 50)
    
    for epoch in range(n_epochs):
        
        # ── Sample new random collocation points each epoch ──────────────────
        # z uniformly from [0, Z_MAX], ξ uniformly from [XI_MIN, XI_MAX]
        z_phys  = torch.rand(n_collocation, 1, device=device) * Z_MAX
        xi_phys = torch.rand(n_collocation, 1, device=device) * (XI_MAX - XI_MIN) + XI_MIN
        
        # ── Sample IC points: z=0, random ξ ──────────────────────────────────
        z_ic  = torch.zeros(n_ic, 1, device=device)
        xi_ic = torch.rand(n_ic, 1, device=device) * (XI_MAX - XI_MIN) + XI_MIN
        
        # ── Forward + loss + backward ─────────────────────────────────────────
        optimizer.zero_grad()
        
        total_loss, phys_loss, ic_loss = compute_loss(
            model, z_phys, xi_phys, z_ic, xi_ic
        )
        
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        
        loss_history.append(total_loss.item())
        
        # print progress every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1:5d} | Total: {total_loss.item():.6f} "
                  f"| Physics: {phys_loss:.6f} | IC: {ic_loss:.6f} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    print("-" * 50)
    print("Training complete.\n")
    
    return loss_history


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION + PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_and_plot(model, loss_history):
    """
    Compare PINN predictions against analytical solutions for several ξ values.
    Plot:
        1. Loss curve over training
        2. PINN vs analytical for 4 different damping ratios
    """
    model.eval()  # disable dropout/batchnorm (not used here but good practice)
    
    z_test = np.linspace(0, Z_MAX, 500)                    # 500 test points
    xi_vals = [0.1, 0.2, 0.3, 0.4]                        # test damping ratios
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("PINN — Damped Harmonic Oscillator", fontsize=14)
    
    # ── Plot 1: Loss curve ────────────────────────────────────────────────────
    axes[0, 0].semilogy(loss_history)        # log scale so we can see progress
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss (log scale)")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].grid(True, alpha=0.3)
    
    # ── Plots 2-5: PINN vs analytical for each ξ ─────────────────────────────
    plot_positions = [(0,1), (0,2), (1,0), (1,1)]
    
    for idx, xi_val in enumerate(xi_vals):
        row, col = plot_positions[idx]
        ax = axes[row][col]
        
        # analytical solution
        x_exact = analytical_solution(z_test, xi_val)
        
        # PINN prediction
        with torch.no_grad():
            z_t  = torch.tensor(z_test, dtype=torch.float32).unsqueeze(1).to(device)
            xi_t = torch.full_like(z_t, xi_val)
            x_pred = model(z_t, xi_t).cpu().numpy().flatten()
        
        ax.plot(z_test, x_exact, 'b-',  linewidth=2,   label="Analytical")
        ax.plot(z_test, x_pred,  'r--', linewidth=1.5, label="PINN", alpha=0.8)
        ax.set_xlabel("z (time)")
        ax.set_ylabel("x (position)")
        ax.set_title(f"ξ = {xi_val}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # compute L2 relative error
        l2_err = np.sqrt(np.mean((x_pred - x_exact)**2)) / np.sqrt(np.mean(x_exact**2))
        ax.text(0.02, 0.95, f"L2 err: {l2_err:.4f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', color='green')
    
    # hide unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("outputs/pinn_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to outputs/pinn_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # 1. Build model
    model = PINN(hidden_dim=64, num_layers=6).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. Train
    loss_history = train(
        model,
        n_epochs=30000,
        n_collocation=3000,
        n_ic=300,
        lr=5e-4
    )
    
    # 3. Evaluate and plot
    evaluate_and_plot(model, loss_history)
    
    # 4. Save model weights
    torch.save(model.state_dict(), "outputs/pinn_oscillator.pth")
    print("Model saved to outputs/pinn_oscillator.pth")