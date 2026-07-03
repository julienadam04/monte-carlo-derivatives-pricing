
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ==========================================
# 1. INITIALISATION DU RÉSEAU
# ==========================================

def init_network_deep(layer_sizes=[2, 20, 20, 1]):
    
    np.random.seed(42) 
    params = {}
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i+1]
        
        limit = np.sqrt(6 / (fan_in + fan_out))
        params[f"W{i+1}"] = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
        params[f"b{i+1}"] = np.zeros((1, fan_out))
        
    return params

# ==========================================
# 2. FORWARD PASS
# ==========================================

def forward_deep(params, S, t, K, T):
    
    S_norm = np.array(S).reshape(-1, 1) / K
    t_norm = np.array(t).reshape(-1, 1) / (T + 1e-8)
    
    A = np.concatenate([S_norm, t_norm], axis=1)
    num_layers = len(params) // 2
    
    for i in range(1, num_layers):
        Z = A @ params[f"W{i}"] + params[f"b{i}"]
        A = np.tanh(Z) 
        
    C = A @ params[f"W{num_layers}"] + params[f"b{num_layers}"]
    return C.flatten()

# ==========================================
# 3. DÉRIVÉES / RÉSIDU DE L'EDP
# ==========================================

def derivatives(params, S, t, K, T, eps=1e-3):
   
    C = forward_deep(params, S, t, K, T)

    Cp = forward_deep(params, S + eps, t, K, T)
    Cm = forward_deep(params, S - eps, t, K, T)
    dC_dS = (Cp - Cm) / (2 * eps)
    d2C_dS2 = (Cp - 2*C + Cm) / (eps**2)

    Ct_p = forward_deep(params, S, t + eps, K, T)
    Ct_m = forward_deep(params, S, t - eps, K, T)
    dC_dt = (Ct_p - Ct_m) / (2 * eps)

    return C, dC_dS, d2C_dS2, dC_dt

def bs_residual(params, S, t, K, T, sigma, r):

    C, dC_dS, d2C_dS2, dC_dt = derivatives(params, S, t, K, T)
    return dC_dt + 0.5 * sigma**2 * S**2 * d2C_dS2 + r * S * dC_dS - r * C


# ==========================================
# 4. FONCTION DE PERTE
# ==========================================

def loss_total(params, S_pde, t_pde, S_term, K, T, sigma, r, option_type='call', lambda_term=10.0):

    # 1. Perte Physique (PDE)
    R = bs_residual(params, S_pde, t_pde, K, T, sigma, r)
    L_phys = np.mean(R**2)

    # 2. Perte Terminale (Payoff)
    tT = np.full_like(S_term, T)
    C_term = forward_deep(params, S_term, tT, K, T)
    
    if option_type == 'call':
        payoff = np.maximum(S_term - K, 0)
    else: # put
        payoff = np.maximum(K - S_term, 0)
        
    L_term = np.mean((C_term - payoff)**2)

    return L_phys + lambda_term * L_term

# ==========================================
# 5. ENTRAÎNEMENT (ADAM)
# ==========================================

def train_adam(params, S_pde, t_pde, S_term, K, T, sigma, r, option_type='call', 
               lr=0.01, epochs=50, lambda_term=10.0):
    """
    Entraîne le PINN avec l'algorithme Adam from scratch.
    """
    # Paramètres d'Adam
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m = {key: np.zeros_like(val) for key, val in params.items()}
    v = {key: np.zeros_like(val) for key, val in params.items()}
    t_step = 0

    eps_grad = 1e-4 # Epsilon pour le gradient des POIDS (différent du eps de la PDE)

    for epoch in range(epochs):
        t_step += 1
        L = loss_total(params, S_pde, t_pde, S_term, K, T, sigma, r, option_type, lambda_term)
        grads = {}

        # Calcul du gradient par différences finies sur les paramètres
        for key in params:
            grads[key] = np.zeros_like(params[key])
            it = np.nditer(params[key], flags=['multi_index'], op_flags=['readwrite'])
            
            for _ in it:
                idx = it.multi_index
                old_val = params[key][idx]

                # Perturbation +
                params[key][idx] = old_val + eps_grad
                L_plus = loss_total(params, S_pde, t_pde, S_term, K, T, sigma, r, option_type, lambda_term)

                # Perturbation -
                params[key][idx] = old_val - eps_grad
                L_minus = loss_total(params, S_pde, t_pde, S_term, K, T, sigma, r, option_type, lambda_term)

                # Restauration
                params[key][idx] = old_val
                
                # Gradient central
                grads[key][idx] = (L_plus - L_minus) / (2 * eps_grad)

        # Mise à jour des poids avec Adam
        for key in params:
            m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
            v[key] = beta2 * v[key] + (1 - beta2) * (grads[key]**2)
            
            # Correction du biais
            m_hat = m[key] / (1 - beta1**t_step)
            v_hat = v[key] / (1 - beta2**t_step)
            
            # Update
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss = {L:.4f}")

    return params

if __name__ == "__main__":
    K, T, sigma, r = 100.0, 1.0, 0.2, 0.05
    N_pde = 1000 # Réduit à 1000 pour accélérer les différences finies
    
    # Points de colocation
    S_pde = np.random.uniform(0.0, 200.0, N_pde)
    t_pde = np.random.uniform(0.0, T, N_pde)
    S_term = np.random.uniform(0.0, 200.0, N_pde)

    # -- ENTRAÎNEMENT DU CALL --
    print("--- Entraînement du Call ---")
    params_call = init_network_deep([2, 10, 10, 1]) # Réseau modéré pour le temps de calcul
    params_call_trained = train_adam(params_call, S_pde, t_pde, S_term, K, T, sigma, r, 
                                     option_type='call', lr=0.05, epochs=30, lambda_term=20.0)

    # -- ENTRAÎNEMENT DU PUT --
    print("\n--- Entraînement du Put ---")
    params_put = init_network_deep([2, 10, 10, 1])
    params_put_trained = train_adam(params_put, S_pde, t_pde, S_term, K, T, sigma, r, 
                                    option_type='put', lr=0.05, epochs=30, lambda_term=20.0)

# ==========================================
# 6. VISUALISATION
# ==========================================

def plot_surface(params, K, T, title):
    S_vals = np.linspace(0.01, 200, 40)
    t_vals = np.linspace(0, T, 40)
    S_grid, t_grid = np.meshgrid(S_vals, t_vals)
    
    # Flatten pour passer dans le réseau
    S_flat = S_grid.flatten()
    t_flat = t_grid.flatten()
    
    C_flat = forward_deep(params, S_flat, t_flat, K, T)
    C_grid = C_flat.reshape(S_grid.shape)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, t_grid, C_grid, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("S (Prix du sous-jacent)")
    ax.set_ylabel("t (Temps)")
    ax.set_zlabel("Prix de l'Option")
    plt.show()

# ==========================================
# 7. EXÉCUTION
# ==========================================
if __name__ == "__main__":
    pass
