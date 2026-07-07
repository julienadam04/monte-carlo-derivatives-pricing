import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# ============================================================
# PARTIE 2 — Paramètres et simulation GBM
# ============================================================

class MarketParams:
    """
    Conteneur des paramètres de marché.
    Centralise tous les paramètres pour éviter les erreurs de passage.
    """
    def __init__(
        self,
        S0: float,      # Prix initial de l'action
        K: float,       # Strike
        T: float,       # Maturité en années
        r: float,       # Taux sans risque
        sigma: float,   # Volatilité
        N: int,         # Nombre de simulations Monte Carlo
        M: int          # Nombre de pas de temps
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.M = M
        self.dt = T / M  # Pas de temps

    def __repr__(self):
        return (
            f"MarketParams(S0={self.S0}, K={self.K}, T={self.T}, "
            f"r={self.r}, sigma={self.sigma}, N={self.N}, M={self.M})"
        )


def simulate_gbm(params: MarketParams, seed: Optional[int] = None) -> np.ndarray:
    """
    Simule N trajectoires du prix S_t sous la mesure risque-neutre Q.

    Schéma exact :
        S_{i+1} = S_i * exp((r - sigma²/2)*dt + sigma*sqrt(dt)*Z_i)

    Returns:
        paths : np.ndarray de shape (N, M+1)
                paths[i, j] = prix de la simulation i au temps j*dt
    """
    if seed is not None:
        np.random.seed(seed)

    dt = params.dt
    N, M = params.N, params.M

    # Matrice des chocs gaussiens : shape (N, M)
    # Chaque ligne = une trajectoire, chaque colonne = un pas de temps
    Z = np.random.standard_normal((N, M))

    # Initialisation : toutes les trajectoires commencent à S0
    paths = np.zeros((N, M + 1))
    paths[:, 0] = params.S0

    # Facteur déterministe commun à tous les pas
    drift = (params.r - 0.5 * params.sigma ** 2) * dt

    # Facteur stochastique
    diffusion = params.sigma * np.sqrt(dt)

    # Simulation vectorisée (pas de boucle sur les simulations)
    for j in range(M):
        paths[:, j + 1] = paths[:, j] * np.exp(drift + diffusion * Z[:, j])

    return paths
# ============================================================
# PARTIE 3 — Estimateur Monte Carlo de base
# ============================================================

def mc_price_european(
    paths: np.ndarray,
    params: MarketParams,
    option_type: str = "call"
) -> Tuple[float, float, float]:
    """
    Prix d'une option européenne par Monte Carlo brut.

    Arguments:
        paths       : trajectoires simulées, shape (N, M+1)
        params      : paramètres de marché
        option_type : "call" ou "put"

    Returns:
        price   : prix estimé
        std_err : erreur standard de l'estimateur
        ci_95   : demi-largeur de l'IC à 95%
    """
    # Prix à maturité : dernière colonne de chaque trajectoire
    S_T = paths[:, -1]

    # Calcul des payoffs
    if option_type == "call":
        payoffs = np.maximum(S_T - params.K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(params.K - S_T, 0.0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    # Actualisation
    discount = np.exp(-params.r * params.T)

    # Estimateur de la moyenne
    price = discount * np.mean(payoffs)

    # Erreur standard : std(payoffs) / sqrt(N), puis actualisé
    std_err = discount * np.std(payoffs, ddof=1) / np.sqrt(params.N)

    # IC à 95% : ± 1.96 * std_err
    ci_95 = 1.96 * std_err

    return price, std_err, ci_95


def black_scholes_price(params: MarketParams, option_type: str = "call") -> float:
    """
    Prix analytique Black-Scholes pour une option européenne.
    Sert de benchmark pour valider la simulation Monte Carlo.
    """
    S0, K, T, r, sigma = params.S0, params.K, params.T, params.r, params.sigma

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        # Parité call-put : P = C - S0 + K*e^(-rT)
        call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        price = call - S0 + K * np.exp(-r * T)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    return price
# ============================================================
# PARTIE 4 — Options path-dependent
# ============================================================

def mc_price_asian(
    paths: np.ndarray,
    params: MarketParams,
    option_type: str = "call",
    average_type: str = "arithmetic"
) -> Tuple[float, float, float]:
    """
    Prix d'une option asiatique.
    Payoff basé sur la moyenne du prix sur toute la durée de vie.

    Types de moyenne :
    - "arithmetic" : (1/M) * Σ S_ti  (pas de formule fermée)
    - "geometric"  : exp((1/M) * Σ ln(S_ti))  (formule fermée existe)
    """
    if average_type == "arithmetic":
        # Moyenne arithmétique des prix (exclut S_0)
        avg_price = np.mean(paths[:, 1:], axis=1)
    elif average_type == "geometric":
        avg_price = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    else:
        raise ValueError("average_type: 'arithmetic' ou 'geometric'")

    if option_type == "call":
        payoffs = np.maximum(avg_price - params.K, 0.0)
    else:
        payoffs = np.maximum(params.K - avg_price, 0.0)

    discount = np.exp(-params.r * params.T)
    price = discount * np.mean(payoffs)
    std_err = discount * np.std(payoffs, ddof=1) / np.sqrt(params.N)
    ci_95 = 1.96 * std_err

    return price, std_err, ci_95


def mc_price_barrier(
    paths: np.ndarray,
    params: MarketParams,
    barrier: float,
    barrier_type: str = "up-and-out",
    option_type: str = "call"
) -> Tuple[float, float, float]:
    """
    Prix d'une option à barrière.

    Types :
    - "up-and-out"  : s'annule si S_t dépasse la barrière (knock-out haussier)
    - "down-and-out": s'annule si S_t passe sous la barrière (knock-out baissier)
    - "up-and-in"   : ne s'active que si S_t dépasse la barrière
    - "down-and-in" : ne s'active que si S_t passe sous la barrière
    """
    S_T = paths[:, -1]

    # Payoff de base (call ou put européen)
    if option_type == "call":
        base_payoff = np.maximum(S_T - params.K, 0.0)
    else:
        base_payoff = np.maximum(params.K - S_T, 0.0)

    # Maximum et minimum sur toute la trajectoire
    max_price = np.max(paths, axis=1)
    min_price = np.min(paths, axis=1)

    # Application de la condition barrière
    if barrier_type == "up-and-out":
        # Survit si S_t n'a jamais dépassé la barrière
        alive = (max_price < barrier).astype(float)
        payoffs = base_payoff * alive
    elif barrier_type == "down-and-out":
        alive = (min_price > barrier).astype(float)
        payoffs = base_payoff * alive
    elif barrier_type == "up-and-in":
        triggered = (max_price >= barrier).astype(float)
        payoffs = base_payoff * triggered
    elif barrier_type == "down-and-in":
        triggered = (min_price <= barrier).astype(float)
        payoffs = base_payoff * triggered
    else:
        raise ValueError("barrier_type non reconnu")

    discount = np.exp(-params.r * params.T)
    price = discount * np.mean(payoffs)
    std_err = discount * np.std(payoffs, ddof=1) / np.sqrt(params.N)
    ci_95 = 1.96 * std_err

    return price, std_err, ci_95
# ============================================================
# PARTIE 5.2 — Variables antithétiques
# ============================================================

def simulate_gbm_antithetic(
    params: MarketParams,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simule des paires de trajectoires antithétiques.
    Pour chaque Z, simule aussi -Z.

    Returns:
        paths_pos : trajectoires avec Z, shape (N//2, M+1)
        paths_neg : trajectoires avec -Z, shape (N//2, M+1)
    """
    if seed is not None:
        np.random.seed(seed)

    N_half = params.N // 2
    M = params.M
    dt = params.dt

    # On génère seulement N/2 chocs : l'autre moitié est -Z
    Z = np.random.standard_normal((N_half, M))

    drift = (params.r - 0.5 * params.sigma ** 2) * dt
    diffusion = params.sigma * np.sqrt(dt)

    # Trajectoires avec +Z
    paths_pos = np.zeros((N_half, M + 1))
    paths_pos[:, 0] = params.S0
    for j in range(M):
        paths_pos[:, j + 1] = paths_pos[:, j] * np.exp(drift + diffusion * Z[:, j])

    # Trajectoires avec -Z (antithétiques)
    paths_neg = np.zeros((N_half, M + 1))
    paths_neg[:, 0] = params.S0
    for j in range(M):
        paths_neg[:, j + 1] = paths_neg[:, j] * np.exp(drift + diffusion * (-Z[:, j]))

    return paths_pos, paths_neg


def mc_price_antithetic(
    params: MarketParams,
    option_type: str = "call",
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Prix Monte Carlo avec variables antithétiques.
    """
    paths_pos, paths_neg = simulate_gbm_antithetic(params, seed)

    S_T_pos = paths_pos[:, -1]
    S_T_neg = paths_neg[:, -1]

    if option_type == "call":
        payoffs_pos = np.maximum(S_T_pos - params.K, 0.0)
        payoffs_neg = np.maximum(S_T_neg - params.K, 0.0)
    else:
        payoffs_pos = np.maximum(params.K - S_T_pos, 0.0)
        payoffs_neg = np.maximum(params.K - S_T_neg, 0.0)

    # Moyenne des paires antithétiques : estimateur à variance réduite
    payoffs_anti = (payoffs_pos + payoffs_neg) / 2.0

    discount = np.exp(-params.r * params.T)
    price = discount * np.mean(payoffs_anti)
    std_err = discount * np.std(payoffs_anti, ddof=1) / np.sqrt(params.N // 2)
    ci_95 = 1.96 * std_err

    return price, std_err, ci_95
# ============================================================
# PARTIE 5.3 — Variable de contrôle
# ============================================================

def mc_price_control_variate(
    paths: np.ndarray,
    params: MarketParams,
    option_type: str = "call"
) -> Tuple[float, float, float]:
    """
    Prix Monte Carlo avec variable de contrôle.

    Variable de contrôle : payoff call/put européen
    Valeur connue analytiquement : prix Black-Scholes
    """
    S_T = paths[:, -1]

    # --- Payoffs à estimer (ici call/put européen, peut être remplacé par asiatique) ---
    if option_type == "call":
        payoffs_target = np.maximum(S_T - params.K, 0.0)
    else:
        payoffs_target = np.maximum(params.K - S_T, 0.0)

    # --- Variable de contrôle : call européen (même payoff ici, illustratif) ---
    # En pratique, X serait un produit différent (ex: option géométrique pour asiatique)
    payoffs_control = np.maximum(S_T - params.K, 0.0)

    # Valeur analytique connue de la variable de contrôle
    mu_control = black_scholes_price(params, "call") * np.exp(params.r * params.T)
    # Note : mu_control est l'espérance non actualisée, E^Q[max(S_T-K, 0)]

    # --- Estimation de β par régression OLS ---
    # β = Cov(Φ, X) / Var(X)
    cov_matrix = np.cov(payoffs_target, payoffs_control)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # --- Estimateur contrôlé ---
    payoffs_cv = payoffs_target - beta * (payoffs_control - mu_control)

    discount = np.exp(-params.r * params.T)
    price = discount * np.mean(payoffs_cv)
    std_err = discount * np.std(payoffs_cv, ddof=1) / np.sqrt(params.N)
    ci_95 = 1.96 * std_err

    # Corrélation empirique (mesure l'efficacité)
    rho = np.corrcoef(payoffs_target, payoffs_control)[0, 1]
    variance_reduction = 1 - rho ** 2  # proportion de variance restante

    return price, std_err, ci_95


# ============================================================
# PARTIE 5.4 — Méthode combinée : antithétique + contrôle
# ============================================================

def mc_price_combined(
    params: MarketParams,
    option_type: str = "call",
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Combine variables antithétiques et variable de contrôle.
    Donne généralement la meilleure précision.
    """
    paths_pos, paths_neg = simulate_gbm_antithetic(params, seed)

    S_T_pos = paths_pos[:, -1]
    S_T_neg = paths_neg[:, -1]

    if option_type == "call":
        pay_pos = np.maximum(S_T_pos - params.K, 0.0)
        pay_neg = np.maximum(S_T_neg - params.K, 0.0)
        control_pos = np.maximum(S_T_pos - params.K, 0.0)
        control_neg = np.maximum(S_T_neg - params.K, 0.0)
    else:
        pay_pos = np.maximum(params.K - S_T_pos, 0.0)
        pay_neg = np.maximum(params.K - S_T_neg, 0.0)
        control_pos = np.maximum(S_T_pos - params.K, 0.0)
        control_neg = np.maximum(S_T_neg - params.K, 0.0)

    # Moyennes antithétiques
    payoffs = (pay_pos + pay_neg) / 2.0
    controls = (control_pos + control_neg) / 2.0

    mu_control = black_scholes_price(params, "call") * np.exp(params.r * params.T)

    cov_matrix = np.cov(payoffs, controls)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    payoffs_cv = payoffs - beta * (controls - mu_control)

    discount = np.exp(-params.r * params.T)
    price = discount * np.mean(payoffs_cv)
    std_err = discount * np.std(payoffs_cv, ddof=1) / np.sqrt(params.N // 2)
    ci_95 = 1.96 * std_err

    return price, std_err, ci_95
# ============================================================
# PARTIE 6 — Algorithme de Longstaff-Schwartz
# ============================================================

def laguerre_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Construit la matrice de base avec les polynômes de Laguerre généralisés.
    Utilisés dans le papier original de Longstaff-Schwartz (2001).

    Arguments:
        x      : valeurs des prix S_t, shape (n,)
        degree : nombre de fonctions de base (excluant la constante)

    Returns:
        basis : matrice de shape (n, degree+1)
                colonnes : 1, L_1(x), L_2(x), ..., L_degree(x)
    """
    # Normalisation pour la stabilité numérique
    x_norm = x / np.mean(x)

    L = np.ones((len(x), degree + 1))
    if degree >= 1:
        L[:, 1] = 1 - x_norm
    if degree >= 2:
        L[:, 2] = 1 - 2 * x_norm + 0.5 * x_norm ** 2
    if degree >= 3:
        L[:, 3] = 1 - 3 * x_norm + 1.5 * x_norm ** 2 - x_norm ** 3 / 6

    return L


def polynomial_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Base polynomiale simple : [1, x, x², ..., x^degree]
    Alternative aux polynômes de Laguerre.
    """
    n = len(x)
    basis = np.ones((n, degree + 1))
    for k in range(1, degree + 1):
        basis[:, k] = x ** k
    return basis


def longstaff_schwartz(
    paths: np.ndarray,
    params: MarketParams,
    option_type: str = "put",
    basis_type: str = "laguerre",
    basis_degree: int = 3
) -> Tuple[float, float]:
    """
    Prix d'une option américaine par l'algorithme de Longstaff-Schwartz.

    Notes :
    - Fonctionne pour call et put américains
    - Les puts américains ont une valeur > puts européens (exercice anticipé optimal)
    - Les calls américains sur action sans dividende valent autant que les calls européens
      (exercice anticipé jamais optimal → Black-Scholes s'applique)

    Arguments:
        paths        : trajectoires GBM, shape (N, M+1)
        params       : paramètres de marché
        option_type  : "put" ou "call"
        basis_type   : "laguerre" ou "polynomial"
        basis_degree : ordre des fonctions de base

    Returns:
        price    : prix de l'option américaine
        std_err  : erreur standard de l'estimateur
    """
    N, M_plus_1 = paths.shape
    M = M_plus_1 - 1
    dt = params.dt
    K = params.K
    r = params.r

    # --- Fonction de payoff ---
    if option_type == "put":
        payoff = lambda S: np.maximum(K - S, 0.0)
    elif option_type == "call":
        payoff = lambda S: np.maximum(S - K, 0.0)
    else:
        raise ValueError("option_type: 'put' ou 'call'")

    # --- Choix de la base ---
    if basis_type == "laguerre":
        basis_fn = lambda x: laguerre_basis(x, basis_degree)
    else:
        basis_fn = lambda x: polynomial_basis(x, basis_degree)

    # --- Facteur d'actualisation pour un pas de temps ---
    discount_factor = np.exp(-r * dt)

    # ----------------------------------------------------------------
    # BACKWARD INDUCTION
    # ----------------------------------------------------------------

    # Étape 1 : Initialisation à maturité avec le payoff terminal
    # cash_flows[i] = valeur actualisée future de la trajectoire i
    cash_flows = payoff(paths[:, M]).copy()

    # Étape 2 : Remonter de M-1 à 1 (on exclut t=0 car pas d'exercice à t=0)
    for j in range(M - 1, 0, -1):

        # Prix courant à l'étape j pour toutes les trajectoires
        S_j = paths[:, j]

        # Valeur d'exercice immédiat
        exercise_value = payoff(S_j)

        # Identifier les trajectoires In-The-Money (exercice positif)
        itm_mask = exercise_value > 0
        n_itm = np.sum(itm_mask)

        # Si aucune trajectoire ITM, on continue partout
        if n_itm == 0:
            cash_flows *= discount_factor
            continue

        # --- Régression OLS sur les trajectoires ITM ---
        S_itm = S_j[itm_mask]

        # Variable dépendante : valeur de continuation actualisée
        Y = discount_factor * cash_flows[itm_mask]

        # Matrice des régresseurs : fonctions de base de S_itm
        X_basis = basis_fn(S_itm)

        # Résolution des moindres carrés : a* = (X'X)^{-1} X'Y
        # np.linalg.lstsq est plus stable que l'inversion directe
        coefficients, _, _, _ = np.linalg.lstsq(X_basis, Y, rcond=None)

        # Valeur de continuation estimée par la régression
        continuation_value = X_basis @ coefficients

        # --- Règle d'exercice optimal ---
        # Exercer si la valeur immédiate dépasse la continuation estimée
        exercise_now = exercise_value[itm_mask] > continuation_value

        # Mettre à jour les cash_flows pour les trajectoires ITM
        # Pour celles qu'on exerce : cash_flow = valeur immédiate (pas d'actualisation sup.)
        # Pour celles qu'on continue : cash_flow = valeur actualisée future
        updated_cf = np.where(exercise_now, exercise_value[itm_mask], Y)
        cash_flows[itm_mask] = updated_cf

        # Pour les trajectoires OTM : juste actualiser
        cash_flows[~itm_mask] *= discount_factor

    # --- Prix final ---
    # Actualiser d'un pas de plus pour aller de t=1*dt à t=0
    price = discount_factor * np.mean(cash_flows)
    std_err = discount_factor * np.std(cash_flows, ddof=1) / np.sqrt(N)

    return price, std_err


def black_scholes_put(params: MarketParams) -> float:
    """Prix analytique BS du put européen (borne inférieure du put américain)."""
    return black_scholes_price(params, "put")
# ============================================================
# PARTIE 7 — Visualisations
# ============================================================

def plot_trajectories(
    paths: np.ndarray,
    params: MarketParams,
    n_display: int = 50,
    title: str = "Trajectoires GBM simulées"
) -> None:
    """
    Affiche un sous-ensemble de trajectoires simulées avec la trajectoire moyenne.
    """
    M = params.M
    T = params.T
    time_grid = np.linspace(0, T, M + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Affichage des trajectoires individuelles (transparentes)
    for i in range(min(n_display, paths.shape[0])):
        ax.plot(time_grid, paths[i], alpha=0.15, linewidth=0.8, color="steelblue")

    # Trajectoire moyenne
    ax.plot(time_grid, np.mean(paths, axis=0),
            color="darkblue", linewidth=2.5, label="Moyenne des trajectoires", zorder=5)

    # Intervalles de confiance empiriques (10e et 90e percentiles)
    p10 = np.percentile(paths, 10, axis=0)
    p90 = np.percentile(paths, 90, axis=0)
    ax.fill_between(time_grid, p10, p90, alpha=0.15, color="steelblue",
                    label="Intervalle [10%, 90%]")

    # Prix actuel et strike
    ax.axhline(params.S0, color="green", linestyle="--", linewidth=1.2, label=f"S0 = {params.S0}")
    ax.axhline(params.K, color="red", linestyle="--", linewidth=1.2, label=f"K = {params.K}")

    ax.set_xlabel("Temps (années)", fontsize=12)
    ax.set_ylabel("Prix de l'action (€)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_convergence(
    params: MarketParams,
    option_type: str = "call",
    N_values: list = None
) -> None:
    """
    Montre la convergence du prix MC vers le prix BS en fonction de N.
    Illustre la loi en 1/√N de l'erreur.
    """
    if N_values is None:
        N_values = [100, 500, 1000, 5000, 10000, 50000, 100000]

    bs_price = black_scholes_price(params, option_type)
    mc_prices = []
    mc_errors = []

    # Utiliser des paramètres modifiés pour chaque N
    for N in N_values:
        p = MarketParams(params.S0, params.K, params.T, params.r, params.sigma, N, params.M)
        paths = simulate_gbm(p, seed=42)
        price, std_err, ci = mc_price_european(paths, p, option_type)
        mc_prices.append(price)
        mc_errors.append(ci)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graphe 1 : Prix MC vs Prix BS
    axes[0].semilogx(N_values, mc_prices, "o-", color="steelblue",
                     label="Prix Monte Carlo", linewidth=2)
    axes[0].errorbar(N_values, mc_prices, yerr=mc_errors,
                     fmt="none", color="steelblue", alpha=0.5, capsize=4)
    axes[0].axhline(bs_price, color="red", linestyle="--",
                    linewidth=2, label=f"Prix Black-Scholes = {bs_price:.4f}")
    axes[0].set_xlabel("Nombre de simulations N (échelle log)", fontsize=11)
    axes[0].set_ylabel("Prix de l'option (€)", fontsize=11)
    axes[0].set_title("Convergence du prix MC", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Graphe 2 : Erreur absolue vs N (doit suivre 1/√N)
    abs_errors = np.abs(np.array(mc_prices) - bs_price)
    axes[1].loglog(N_values, abs_errors, "o-", color="darkorange",
                   label="|Prix MC - Prix BS|", linewidth=2)
    # Courbe de référence 1/√N
    ref = abs_errors[0] * np.sqrt(N_values[0]) / np.sqrt(np.array(N_values))
    axes[1].loglog(N_values, ref, "--", color="gray",
                   label="Référence 1/√N", linewidth=1.5)
    axes[1].set_xlabel("Nombre de simulations N (log)", fontsize=11)
    axes[1].set_ylabel("Erreur absolue (log)", fontsize=11)
    axes[1].set_title("Décroissance de l'erreur : loi 1/√N", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_payoff_distribution(
    paths: np.ndarray,
    params: MarketParams,
    option_type: str = "call"
) -> None:
    """
    Distribution des payoffs simulés, avec le prix estimé et l'IC.
    """
    S_T = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(S_T - params.K, 0.0)
    else:
        payoffs = np.maximum(params.K - S_T, 0.0)

    discount = np.exp(-params.r * params.T)
    price = discount * np.mean(payoffs)
    bs_price = black_scholes_price(params, option_type)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogramme (incluant les payoffs nuls)
    ax.hist(payoffs[payoffs > 0], bins=80, density=True,
            color="steelblue", alpha=0.7, label="Payoffs > 0")

    # Proportion de simulations OTM
    pct_otm = np.mean(payoffs == 0) * 100

    ax.axvline(np.mean(payoffs), color="darkblue", linestyle="-",
               linewidth=2, label=f"Payoff moyen = {np.mean(payoffs):.2f}")

    ax.set_xlabel("Payoff à maturité (€)", fontsize=11)
    ax.set_ylabel("Densité", fontsize=11)
    ax.set_title(
        f"Distribution des payoffs — {option_type.capitalize()} "
        f"(S0={params.S0}, K={params.K})\n"
        f"Prix MC = {price:.4f} | Prix BS = {bs_price:.4f} | "
        f"{pct_otm:.1f}% OTM",
        fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_variance_reduction(params: MarketParams) -> None:
    """
    Tableau comparatif des méthodes : MC brut, antithétique, variable de contrôle.
    """
    bs_price = black_scholes_price(params, "call")

    results = {}

    # MC brut
    paths = simulate_gbm(params, seed=42)
    p, se, ci = mc_price_european(paths, params, "call")
    results["MC Brut"] = {"price": p, "std_err": se, "ci": ci}

    # Antithétique
    p, se, ci = mc_price_antithetic(params, "call", seed=42)
    results["Antithétique"] = {"price": p, "std_err": se, "ci": ci}

    # Variable de contrôle
    p, se, ci = mc_price_control_variate(paths, params, "call")
    results["Variable de contrôle"] = {"price": p, "std_err": se, "ci": ci}

    # Combiné
    p, se, ci = mc_price_combined(params, "call", seed=42)
    results["Combiné"] = {"price": p, "std_err": se, "ci": ci}

    # Affichage
    print(f"\n{'='*65}")
    print(f"COMPARAISON DES MÉTHODES — Prix BS de référence : {bs_price:.4f}")
    print(f"{'='*65}")
    print(f"{'Méthode':<25} {'Prix':>8} {'Std Err':>10} {'IC 95%':>10} {'Erreur':>10}")
    print(f"{'-'*65}")
    for method, res in results.items():
        error = abs(res['price'] - bs_price)
        print(f"{method:<25} {res['price']:>8.4f} {res['std_err']:>10.5f} "
              f"±{res['ci']:>8.5f} {error:>10.5f}")
    print(f"{'='*65}\n")
# ============================================================
# SCRIPT PRINCIPAL — Exécution complète du projet
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  PRICING DE DÉRIVÉS ACTIONS PAR MONTE CARLO")
    print("  GBM · Variance Reduction · Longstaff-Schwartz")
    print("=" * 65)

    # ---- Paramètres de marché ----
    params = MarketParams(
        S0=100.0,   # Prix initial
        K=105.0,    # Strike légèrement OTM
        T=1.0,      # Maturité 1 an
        r=0.05,     # Taux sans risque 5%
        sigma=0.20, # Volatilité 20%
        N=50000,    # Simulations
        M=252       # Pas journaliers
    )

    print(f"\nParamètres : {params}\n")

    # ---- 1. Simulation des trajectoires ----
    print(">> Simulation des trajectoires GBM...")
    paths = simulate_gbm(params, seed=42)
    print(f"   Forme du tableau : {paths.shape}  ({params.N} trajectoires × {params.M + 1} dates)")
    print(f"   Prix initial : {paths[:, 0].mean():.4f}  (doit être {params.S0})")
    print(f"   Prix final moyen : {paths[:, -1].mean():.4f}  "
          f"(espéré : {params.S0 * np.exp(params.r * params.T):.4f})\n")

    # ---- 2. Visualisation des trajectoires ----
    plot_trajectories(paths, params, n_display=100,
                      title="Trajectoires GBM sous mesure risque-neutre")

    # ---- 3. Options européennes ----
    print(">> Pricing options EUROPÉENNES")
    print("-" * 50)

    bs_call = black_scholes_price(params, "call")
    bs_put = black_scholes_price(params, "put")
    print(f"   Prix Black-Scholes Call = {bs_call:.4f}")
    print(f"   Prix Black-Scholes Put  = {bs_put:.4f}")

    mc_call, se_call, ci_call = mc_price_european(paths, params, "call")
    mc_put, se_put, ci_put = mc_price_european(paths, params, "put")
    print(f"   Prix MC Call = {mc_call:.4f}  (±{ci_call:.4f})  | erreur = {abs(mc_call - bs_call):.4f}")
    print(f"   Prix MC Put  = {mc_put:.4f}  (±{ci_put:.4f})  | erreur = {abs(mc_put - bs_put):.4f}")

    # Vérification parité call-put : C - P = S0 - K*e^(-rT)
    parity_lhs = mc_call - mc_put
    parity_rhs = params.S0 - params.K * np.exp(-params.r * params.T)
    print(f"\n   Parité call-put : C - P = {parity_lhs:.4f}  (théorique : {parity_rhs:.4f})\n")

    # ---- 4. Distribution des payoffs ----
    plot_payoff_distribution(paths, params, "call")

    # ---- 5. Réduction de variance ----
    print(">> Réduction de variance")
    compare_variance_reduction(params)

    # ---- 6. Convergence ----
    print(">> Analyse de convergence...")
    plot_convergence(params, "call")

    # ---- 7. Options path-dependent ----
    print(">> Pricing options PATH-DEPENDENT")
    print("-" * 50)

    # Asiatique
    asian_call, _, asian_ci = mc_price_asian(paths, params, "call", "arithmetic")
    print(f"   Call Asiatique (arith.)  = {asian_call:.4f}  (±{asian_ci:.4f})")

    asian_geo, _, asian_geo_ci = mc_price_asian(paths, params, "call", "geometric")
    print(f"   Call Asiatique (géom.)   = {asian_geo:.4f}  (±{asian_geo_ci:.4f})")
    print(f"   → L'asiatique < européen ({bs_call:.4f}) car la moyenne lisse le sous-jacent\n")

    # Barrière
    barrier = 130.0
    barrier_call, _, b_ci = mc_price_barrier(paths, params, barrier, "up-and-out", "call")
    print(f"   Call up-and-out (B={barrier}) = {barrier_call:.4f}  (±{b_ci:.4f})")
    print(f"   → Prix < call vanilla ({bs_call:.4f}) : risque de knock-out\n")

    # ---- 8. Option américaine — Longstaff-Schwartz ----
    print(">> Pricing option AMÉRICAINE — Longstaff-Schwartz")
    print("-" * 50)

    # Pour LS, on a besoin de plus de pas (précision de la frontière d'exercice)
    params_ls = MarketParams(
        S0=params.S0,
        K=params.K,
        T=params.T,
        r=params.r,
        sigma=params.sigma,
        N=params.N,
        M=50  # Moins de pas = plus rapide, suffisant pour LS
    )
    paths_ls = simulate_gbm(params_ls, seed=42)

    am_put, am_se = longstaff_schwartz(paths_ls, params_ls, "put",
                                        basis_type="laguerre", basis_degree=3)
    eu_put = black_scholes_price(params_ls, "put")

    print(f"   Put AMÉRICAIN  (LS)    = {am_put:.4f}  (SE = {am_se:.5f})")
    print(f"   Put EUROPÉEN   (BS)    = {eu_put:.4f}")
    print(f"   Prime d'exercice anticipé = {am_put - eu_put:.4f}")
    print(f"   → Le put américain vaut plus : exercice anticipé optimal possible\n")

    # Comparaison avec différents degrés de base
    print("   Sensibilité au degré de la base polynomiale :")
    for degree in [1, 2, 3, 4]:
        p, se = longstaff_schwartz(paths_ls, params_ls, "put",
                                    basis_type="laguerre", basis_degree=degree)
        print(f"   Degré {degree} : Put américain = {p:.4f}  (SE = {se:.5f})")

    # ---- 9. Résumé final ----
    print("\n" + "=" * 65)
    print("  RÉSUMÉ DES PRIX")
    print("=" * 65)
    print(f"  Call européen  (BS)          : {bs_call:.4f}")
    print(f"  Call européen  (MC brut)     : {mc_call:.4f}  ±{ci_call:.4f}")
    print(f"  Call asiatique (MC)          : {asian_call:.4f}")
    print(f"  Call up-and-out (MC)         : {barrier_call:.4f}")
    print(f"  Put européen   (BS)          : {eu_put:.4f}")
    print(f"  Put américain  (LS-MC)       : {am_put:.4f}")
    print(f"  Prime exercice anticipé      : {am_put - eu_put:.4f}")
    print("=" * 65)
