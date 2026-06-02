import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config.settings import L
from core.insurance_math import (optimal_params, P_d, P_p, MV_d, MV_p, k_budget, find_crossing)
from core.fat_tails import MV_d_pareto, MV_p_pareto, THETA

def run_simulation_and_plot(theta_d: float, theta_p: float, output_path: str):
    if theta_d <= 0 or theta_p <= 0:
        raise ValueError("Loading factors must be strictly positive.")
    
    d_star, k_star = optimal_params(theta_d, theta_p)
    
    gamma_d_max = 80_000
    gamma_d = np.linspace(0.0, gamma_d_max, 6000)

    p_d_arr = P_d(d_star, gamma_d, theta_d)
    p_p_val = P_p(k_star, theta_p)
    p_p_arr = np.full_like(gamma_d, p_p_val)

    mv_d_arr = MV_d(d_star, gamma_d, theta_d)
    mv_p_val = MV_p(k_star, theta_p)
    mv_p_arr = np.full_like(gamma_d, mv_p_val)

    k_bm_arr = k_budget(gamma_d, d_star, theta_d, theta_p)
    mv_pb_arr = np.array([MV_p(k, theta_p) for k in k_bm_arr])

    # Indifference thresholds
    gamma_indif = find_crossing(gamma_d, mv_d_arr, mv_p_arr)
    gamma_indif_prime = find_crossing(gamma_d, mv_d_arr, mv_pb_arr)

    found = [g for g in [gamma_indif, gamma_indif_prime] if not np.isnan(g)]
    x_max = max(found) * 1.20 if found else gamma_d_max
    x_max = max(x_max, 15_000)

    # Print results to console
    sep = "=" * 62
    print(sep)
    print(f"  Parametric vs Indemnity — theta_d={theta_d:.2f}, theta_p={theta_p:.2f}")
    print(sep)
    print(f"  Indifference thresholds:")
    if not np.isnan(gamma_indif):
        print(f"    gamma_indif  (MV_d = MV_p opt)    = {gamma_indif:>10,.2f} €")
    if not np.isnan(gamma_indif_prime):
        print(f"    gamma_indif' (MV_d = MV_p budget) = {gamma_indif_prime:>10,.2f} €")
    print(sep)

    # Plotting logic
    C_MVD, C_MVP, C_MVPB = "#2ca02c", "#c03288", "#d4a017"
    C_PD, C_PP = "#1f77b4", "#c0392b"

    mask = gamma_d <= x_max
    gd_plt = gamma_d[mask]

    fig, ax1 = plt.subplots(figsize=(11, 6.5))

    ax1.set_xlabel(r"$\gamma_d$ [€]", fontsize=13)
    ax1.set_ylabel("Premium [€]", fontsize=13)
    l_pd, = ax1.plot(gd_plt, p_d_arr[mask], color=C_PD, lw=2.0, label=r"$P_d$")
    l_pp, = ax1.plot(gd_plt, p_p_arr[mask], color=C_PP, lw=2.0, label=r"$P_p$")
    
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, p_d_arr[mask].max() * 1.28)

    ax2 = ax1.twinx()
   
    ax2.set_ylabel("Mean-Variance [€]", fontsize=13)
    l_mvd, = ax2.plot(gd_plt, mv_d_arr[mask], color=C_MVD, lw=2.2, label=r"$MV_d$")
    l_mvp, = ax2.plot(gd_plt, mv_p_arr[mask], color=C_MVP, lw=2.2, label=r"$MV_p$")
    l_mvpb, = ax2.plot(gd_plt, mv_pb_arr[mask], color=C_MVPB, lw=2.2, label=r"$MV_p^{(\mathrm{budget}\ P_d)}$")

    mv_all = np.concatenate([mv_d_arr[mask], mv_p_arr[mask], mv_pb_arr[mask]])
    mv_range = mv_all.max() - mv_all.min()
    ax2.set_ylim(mv_all.min() - mv_range * 0.10, mv_all.max() + mv_range * 0.10)

    y0, y1 = ax1.get_ylim()

    if not np.isnan(gamma_indif) and gamma_indif <= x_max:
        ax1.axvline(gamma_indif, color=C_MVD, linestyle='--', lw=1.6, alpha=0.85)
        ax1.text(gamma_indif, y0 + (y1 - y0) * 0.05, 
                 rf"$\gamma_{{\mathrm{{indif}}}} = {gamma_indif:,.0f}$ €", 
                 color=C_MVD, ha='center', va='bottom', fontsize=11,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2.0))

    if not np.isnan(gamma_indif_prime) and gamma_indif_prime <= x_max:
        ax1.axvline(gamma_indif_prime, color=C_MVPB, linestyle='--', lw=1.6, alpha=0.85)
        ax1.text(gamma_indif_prime, y0 + (y1 - y0) * 0.05, 
                 rf"$\gamma_{{\mathrm{{indif}}}}' = {gamma_indif_prime:,.0f}$ €", 
                 color=C_MVPB, ha='center', va='bottom', fontsize=11,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2.0))

    ax1.set_title(r"Premium and Mean-Variance vs $\gamma_d$", fontsize=13, pad=10)
    
    all_lines = [l_mvd, l_mvp, l_mvpb, l_pd, l_pp]
    all_labels = [r"$MV_d$", r"$MV_p$", r"$MV_p^{(\mathrm{budget}\ P_d)}$", r"$P_d$", r"$P_p$"]
    ax1.legend(all_lines, all_labels, title="Series", loc="upper center", 
               bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=10, title_fontsize=10, framealpha=0.9)

    tick_step = max(round(x_max / 5 / 5000) * 5000, 1000)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(tick_step))
    ax1.grid(axis='y', linestyle=':', alpha=0.35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved → {output_path}\n")

def run_bonus_fat_tail_plot(theta_d: float, theta_p: float, output_path: str):
    d_star, k_star = optimal_params(theta_d, theta_p)
    gamma_d = np.linspace(0.0, 50_000, 3000)

    mv_d_base = MV_d(d_star, gamma_d, theta_d)
    mv_p_base = np.full_like(gamma_d, MV_p(k_star, theta_p))
    
    # (Fat Tails)
    mv_d_fat = MV_d_pareto(d_star, gamma_d, theta_d)
    mv_p_fat = np.full_like(gamma_d, MV_p_pareto(k_star, theta_p))

    # --- Print Terminal for Bonus ---
    sep = "=" * 62
    print(sep)
    print(f"  Bonus: Fat Tails (Pareto) vs Exponential")
    print(sep)
    print(f"  Dynamic Calibration:")
    print(f"    THETA parameter      = {THETA:>10,.2f} €")
    print(sep)
    # --------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting
    ax.plot(gamma_d, mv_d_base, label="MV_d (Exp - Base)", color="green", linestyle="--", alpha=0.6)
    ax.plot(gamma_d, mv_p_base, label="MV_p (Exp - Base)", color="purple", linestyle="--", alpha=0.6)
    ax.plot(gamma_d, mv_d_fat, label="MV_d (Pareto - Fat Tails)", color="green", lw=2)
    ax.plot(gamma_d, mv_p_fat, label="MV_p (Pareto - Fat Tails)", color="purple", lw=2)

    ax.set_title("Bonus Stress Test: Impact of Fat Tails on Utility")
    ax.set_xlabel(r"$\gamma_d$ [€]")
    ax.set_ylabel("Mean-Variance Utility [€]")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"\n  Fat Tails Bonus Figure saved → {output_path}\n")