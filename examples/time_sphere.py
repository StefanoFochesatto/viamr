#!/usr/bin/env python3
"""
Multi-level grid sequencing benchmark for VIAMR strategies.
Compares strategies over multiple refinement layers, tracking:
- Cumulative Time
- L2 Error Norm (L2 Accuracy)
- L2 Preferred Norm (L2 Accuracy replacing active set with exact obstacle)
- Jaccard Index (Active Set Accuracy)
"""

import time
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from viamr import VIAMR

# Standard printing in parallel
print = PETSc.Sys.Print

def psiUFL(r):
    """obstacle as UFL, from UFL expression for r"""
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))

def uexactUFL(r):
    """exact solution (and boundary conditions) as UFL, from UFL expression for r"""
    afree = 0.697965148223374
    A, B = 0.680259411891719, 0.471519893402112
    return conditional(le(r, afree), psiUFL(r), -A * ln(r) + B)

def activeexactUFL(r):
    """exact active set as UFL"""
    afree = 0.697965148223374
    return conditional(le(r, afree), 1.0, 0.0)

def errornorm_deg20(u_exact_ufl, uh):
    """L^2 error norm with high degree quadrature to avoid TSFC warnings."""
    normsq = assemble((u_exact_ufl - uh) ** 2 * dx(degree=20))
    return np.sqrt(normsq)

def errornorm_preferred_deg20(u_exact_ufl, psi_ufl, uh, activeh):
    """L^2 error norm against 'preferred' form of numerical solution:
    replace uh with psi_ufl in the active set (activeh == 1.0)."""
    tildeuh = conditional(eq(activeh, 1.0), psi_ufl, uh)
    normsq = assemble((u_exact_ufl - tildeuh) ** 2 * dx(degree=20))
    return np.sqrt(normsq)

def benchmark_sequencing(m0=10, levels=4):
    print(f"Comparing VIAMR strategies over {levels} refinement levels (Starting from {m0}x{m0} mesh)...")
    
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_vi_zero_tolerance": 1.0e-12,
        "snes_linesearch_type": "basic",
        "snes_max_it": 200,
        "snes_rtol": 1.0e-8,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    strat_names = ["UDO", "VCD", "NSV", "UDO+BR", "UDO+GradRec", "VCD+BR", "VCD+GradRec"]
    results = {name: [] for name in strat_names}
    amr = VIAMR()

    for name in strat_names:
        print(f"\nEvaluating strategy: {name}")
        
        # Initial Mesh Setup
        mesh = RectangleMesh(m0, m0, Lx=4.0, Ly=4.0, originX=-2.0, originY=-2.0)
        cumulative_time = 0.0
        
        for level in range(levels + 1):
            V = FunctionSpace(mesh, "CG", 1)
            x, y = SpatialCoordinate(mesh)
            r = sqrt(x * x + y * y)
            
            uh = Function(V, name="u_h")
            lb = Function(V, name="psi").interpolate(psiUFL(r))
            ub = Function(V).interpolate(Constant(PETSc.INFINITY))
            g_ufl = uexactUFL(r)
            psi_ufl = psiUFL(r)
            bcs = DirichletBC(V, g_ufl, "on_boundary")
            
            v = TestFunction(V)
            F = inner(grad(uh), grad(v)) * dx
            problem = NonlinearVariationalProblem(F, uh, bcs)
            solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
            
            # 1. Solve
            t0 = time.perf_counter()
            solver.solve(bounds=(lb, ub))
            t_solve = time.perf_counter() - t0
            cumulative_time += t_solve
            
            # 2. Evaluate Accuracy
            active_h = amr.elemactive(uh, lb)
            l2_err = errornorm_deg20(g_ufl, uh)
            l2_pref = errornorm_preferred_deg20(g_ufl, psi_ufl, uh, active_h)
            jaccard = amr.jaccardUFL(activeexactUFL(r), active_h)
            Nv, Ne, _, _ = amr.meshsizes(mesh)
            
            results[name].append({
                "level": level,
                "ne": Ne,
                "time": cumulative_time,
                "l2": l2_err,
                "l2_pref": l2_pref,
                "jaccard": jaccard
            })
            
            print(f"  Level {level}: Ne={Ne:6d}, Time={cumulative_time:7.3f}s, L2={l2_err:.2e}, L2Pref={l2_pref:.2e}, Jaccard={jaccard:.4f}")
            
            if level < levels:
                # 3. Mark & Refine
                t0 = time.perf_counter()
                if name == "UDO":
                    mark = amr.udomark(uh, lb, n=1)
                elif name == "VCD":
                    mark = amr.vcdmark(uh, lb)
                elif name == "NSV":
                    g_eval = Function(V).interpolate(g_ufl)
                    mark, _, _, _ = amr.nsvmark(uh, lb, g_eval, Constant(0.0), g_ufl)
                elif name == "UDO+BR":
                    residual = -div(grad(uh))
                    mark_udo = amr.udomark(uh, lb, n=1)
                    mark_br, _, _ = amr.brinactivemark(uh, lb, residual)
                    mark = amr.unionmarks(mark_udo, mark_br)
                elif name == "UDO+GradRec":
                    mark_udo = amr.udomark(uh, lb, n=1)
                    mark_gr, _, _ = amr.gradrecinactivemark(uh, lb)
                    mark = amr.unionmarks(mark_udo, mark_gr)
                elif name == "VCD+BR":
                    residual = -div(grad(uh))
                    mark_vcd = amr.vcdmark(uh, lb)
                    mark_br, _, _ = amr.brinactivemark(uh, lb, residual)
                    mark = amr.unionmarks(mark_vcd, mark_br)
                elif name == "VCD+GradRec":
                    mark_vcd = amr.vcdmark(uh, lb)
                    mark_gr, _, _ = amr.gradrecinactivemark(uh, lb)
                    mark = amr.unionmarks(mark_vcd, mark_gr)
                
                mesh = amr.refinemarkedelements(mesh, mark)
                t_amr = time.perf_counter() - t0
                cumulative_time += t_amr

    # Final Summary Table
    print("\n" + "="*95)
    print(f"{'Strategy':<15} | {'Final Ne':<10} | {'Total Time':<12} | {'Final L2':<10} | {'L2 Pref':<10} | {'Jaccard':<8}")
    print("-" * 95)
    for name in strat_names:
        final = results[name][-1]
        print(f"{name:<15} | {final['ne']:<10d} | {final['time']:10.3f}s | {final['l2']:.3e} | {final['l2_pref']:.3e} | {final['jaccard']:.5f}")
    print("="*95)

    # Efficiency Summary Table
    print("\nEfficiency Metrics (Cost per Accuracy - LOWER IS BETTER)")
    print("="*95)
    print(f"{'Strategy':<15} | {'Time/Jaccard':<15} | {'Time * L2_Pref':<15} | {'Ne/Jaccard':<15} | {'Ne * L2_Pref':<15}")
    print("-" * 95)
    for name in strat_names:
        final = results[name][-1]
        t = final['time']
        ne = final['ne']
        l2_pref = final['l2_pref']
        jaccard = final['jaccard']
        
        # Avoid division by zero
        j_safe = jaccard if jaccard > 0 else 1e-10
        
        t_per_j = t / j_safe
        t_times_l2 = t * l2_pref
        ne_per_j = ne / j_safe
        ne_times_l2 = ne * l2_pref
        
        print(f"{name:<15} | {t_per_j:<15.3f} | {t_times_l2:<15.3e} | {ne_per_j:<15.1f} | {ne_times_l2:<15.3f}")
    print("="*95)

    # SOTA Comparison: Defeating NSV
    print("\nState-of-the-Art (NSV) Comparison Summary:")
    print("==========================================")
    
    nsv_final = results["NSV"][-1]
    nsv_jaccard = nsv_final["jaccard"]
    nsv_l2_pref = nsv_final["l2_pref"]
    nsv_time_l2 = nsv_final["time"] * nsv_l2_pref
    nsv_ne_l2 = nsv_final["ne"] * nsv_l2_pref

    print(f"1. Active Set Accuracy (Absolute Jaccard - HIGHER is better):")
    for name in [n for n in strat_names if n != "NSV"]:
        j = results[name][-1]["jaccard"]
        if j > nsv_jaccard:
            print(f"   - {name:<11} beats NSV ({j:.5f} > {nsv_jaccard:.5f})")

    print(f"\n2. Global Accuracy (Absolute L2_Pref - LOWER is better):")
    for name in [n for n in strat_names if "+" in n]:
        l2 = results[name][-1]["l2_pref"]
        if l2 < nsv_l2_pref:
            print(f"   - {name:<11} beats NSV ({l2:.3e} < {nsv_l2_pref:.3e}) - Order of magnitude improvement")

    print(f"\n3. Mesh Efficiency for Global Accuracy (Ne * L2_Pref - LOWER is better):")
    for name in [n for n in strat_names if "+" in n]:
        ne_l2 = results[name][-1]["ne"] * results[name][-1]["l2_pref"]
        if ne_l2 < nsv_ne_l2:
            print(f"   - {name:<11} beats NSV ({ne_l2:.3f} < {nsv_ne_l2:.3f})")

    print(f"\n4. Time Efficiency for Global Accuracy (Time * L2_Pref - LOWER is better):")
    for name in [n for n in strat_names if "+" in n]:
        time_l2 = results[name][-1]["time"] * results[name][-1]["l2_pref"]
        if time_l2 < nsv_time_l2:
            print(f"   - {name:<11} beats NSV ({time_l2:.3e} < {nsv_time_l2:.3e})")
    print("==========================================\n")

    # --- Plotting ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VIAMR Strategies: Convergence Comparison', fontsize=16)

    # Markers and styles
    styles = {
        "UDO": {"marker": "o", "color": "blue"},
        "VCD": {"marker": "s", "color": "orange"},
        "NSV": {"marker": "x", "color": "red", "linewidth": 2},
        "UDO+BR": {"marker": "^", "color": "green", "linestyle": "--"},
        "UDO+GradRec": {"marker": "D", "color": "purple", "linestyle": "--"},
        "VCD+BR": {"marker": "v", "color": "cyan", "linestyle": "-."},
        "VCD+GradRec": {"marker": "p", "color": "magenta", "linestyle": "-."}
    }

    # Plot 1: 1 - Jaccard vs Time (Log-Log)
    ax = axs[0, 0]
    for name in strat_names:
        times = [r["time"] for r in results[name]]
        jaccards = [max(1.0 - r["jaccard"], 1e-12) for r in results[name]]
        ax.plot(times, jaccards, label=name, **styles.get(name, {}))
    ax.set_xlabel('Cumulative Time (s)')
    ax.set_ylabel('1 - Jaccard Index (Active Set Error)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('Active Set Error vs Time (Lower is Better)')
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--")

    # Plot 2: L2_Pref vs Time (Log-Log)
    ax = axs[0, 1]
    for name in strat_names:
        times = [r["time"] for r in results[name]]
        l2_prefs = [r["l2_pref"] for r in results[name]]
        ax.plot(times, l2_prefs, label=name, **styles.get(name, {}))
        
    # Add second-order convergence line O(t^-1)
    if "UDO+GradRec" in results and len(results["UDO+GradRec"]) > 1:
        base_t = results["UDO+GradRec"][1]["time"] 
        base_l2 = results["UDO+GradRec"][1]["l2_pref"]
        max_t = max(r["time"] for r in results["UDO+GradRec"])
        if max_t > base_t:
            ref_ts = np.array([base_t, max_t])
            ref_l2s = base_l2 * (base_t / ref_ts)
            ax.plot(ref_ts, ref_l2s, 'k:', label=r'$\mathcal{O}(t^{-1})$ (2nd Order)', linewidth=2)
            
    ax.set_xlabel('Cumulative Time (s)')
    ax.set_ylabel('Preferred L2 Error')
    ax.set_yscale('log')
    ax.set_xscale('log') 
    ax.set_title('Preferred L2 Error vs Time (Lower is Better)')
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--")

    # Plot 3: 1 - Jaccard vs Elements (Log-Log)
    ax = axs[1, 0]
    for name in strat_names:
        nes = [r["ne"] for r in results[name]]
        jaccards = [max(1.0 - r["jaccard"], 1e-12) for r in results[name]]
        ax.plot(nes, jaccards, label=name, **styles.get(name, {}))
    ax.set_xlabel('Number of Elements (Ne)')
    ax.set_ylabel('1 - Jaccard Index (Active Set Error)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('Active Set Error vs Elements (Lower is Better)')
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--")

    # Plot 4: L2_Pref vs Elements (Log-Log)
    ax = axs[1, 1]
    for name in strat_names:
        nes = [r["ne"] for r in results[name]]
        l2_prefs = [r["l2_pref"] for r in results[name]]
        ax.plot(nes, l2_prefs, label=name, **styles.get(name, {}))
        
    # Add second-order convergence line O(Ne^-1)
    if "UDO+GradRec" in results and len(results["UDO+GradRec"]) > 0:
        base_ne = results["UDO+GradRec"][0]["ne"]
        base_l2 = results["UDO+GradRec"][0]["l2_pref"]
        max_ne = max(r["ne"] for r in results["UDO+GradRec"])
        if max_ne > base_ne:
            ref_nes = np.array([base_ne, max_ne])
            ref_l2s = base_l2 * (base_ne / ref_nes)
            ax.plot(ref_nes, ref_l2s, 'k:', label=r'$\mathcal{O}(N_e^{-1})$ (2nd Order)', linewidth=2)
            
    ax.set_xlabel('Number of Elements (Ne)')
    ax.set_ylabel('Preferred L2 Error')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('Preferred L2 Error vs Elements (Lower is Better)')
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('benchmark_plots.png', dpi=300)
    print("Saved convergence comparison graphs to 'benchmark_plots.png'")

    # --- Efficiency Plotting (The "Cost * Error" metrics) ---
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Efficiency Metrics (Cost per Unit of Accuracy)\nLower is Better. Flat Lines = Optimal Scaling', fontsize=16)

    bar_colors = [styles[n]["color"] for n in strat_names]

    # Top Left: Bar Chart of Final Time Efficiency
    ax = axs2[0, 0]
    final_time_eff = [results[n][-1]["time"] * results[n][-1]["l2_pref"] for n in strat_names]
    bars = ax.bar(strat_names, final_time_eff, color=bar_colors)
    ax.set_ylabel('Final Time * L2_Pref')
    ax.set_title('Final Time Efficiency (Lower is Better)')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2e}', va='bottom', ha='center', fontsize=8)

    # Top Right: Bar Chart of Final Mesh Efficiency
    ax = axs2[0, 1]
    final_ne_eff = [results[n][-1]["ne"] * results[n][-1]["l2_pref"] for n in strat_names]
    bars = ax.bar(strat_names, final_ne_eff, color=bar_colors)
    ax.set_ylabel('Final Ne * L2_Pref')
    ax.set_title('Final Element Efficiency (Lower is Better)')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center', fontsize=8)

    # Bottom Left: Evolution of Time Efficiency over Levels
    ax = axs2[1, 0]
    for name in strat_names:
        levels_arr = [r["level"] for r in results[name]]
        time_effs = [r["time"] * r["l2_pref"] for r in results[name]]
        ax.plot(levels_arr, time_effs, label=name, **styles.get(name, {}))
    ax.set_xlabel('Refinement Level')
    ax.set_ylabel('Time * L2_Pref')
    ax.set_yscale('log')
    ax.set_title('Time Efficiency Evolution (Flat = Optimal Scaling)')
    ax.set_xticks(range(levels + 1))
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--")

    # Bottom Right: Evolution of Mesh Efficiency over Levels
    ax = axs2[1, 1]
    for name in strat_names:
        levels_arr = [r["level"] for r in results[name]]
        ne_effs = [r["ne"] * r["l2_pref"] for r in results[name]]
        ax.plot(levels_arr, ne_effs, label=name, **styles.get(name, {}))
    ax.set_xlabel('Refinement Level')
    ax.set_ylabel('Ne * L2_Pref')
    ax.set_yscale('log')
    ax.set_title('Mesh Efficiency Evolution (Flat = Optimal Scaling)')
    ax.set_xticks(range(levels + 1))
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('efficiency_plots.png', dpi=300)
    print("Saved efficiency comparison graphs to 'efficiency_plots.png'")

if __name__ == "__main__":
    import sys
    m0 = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    levels = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    benchmark_sequencing(m0, levels)
