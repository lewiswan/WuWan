"""
WuWan_pavement_slo.py
==================================================================
Sensor Location Optimization (SLO) for FWD sensor layouts.

D-optimal design via SAA (Sample-Average Approximation) over a
log-uniform modulus prior, with Differential Evolution searching the
free sensor positions (cumulative-gap reparameterisation, exactly as
in the reference notebook/script WuWan_SLO.py / WuWan_FIM_final.ipynb).

This module is the GUI-callable counterpart of WuWan_pavement_inverse /
WuWan_pavement_montecarlo: it consumes the SAME arr_main / arr_noise
layout produced by InputFormsManager.build_arr_main() / build_arr_noise()
in WuWan.py, so the Sensor Location Optimization window can reuse the
exact same default pavement / load / noise data as Back Calculation.

arr_main  columns: 0 Layer | 1 Modulus | 2 Poisson | 3 Thickness |
                   4 (empty) | 5 EvalPoint | 6 r [mm] | 7 Deflection
arr_noise columns: 0 Layer | 1 Modulus prior lo | 2 Modulus prior hi |
                   3 Thickness noise | 4 EvalPoint | 5 r noise |
                   6 Deflection noise

Unlike WuWan_SLO.py (which loads a synthetic "true" pavement from
data_5m.npy), the "true" moduli here are simply whatever the user has
entered in the Layered Profile panel -- the tool optimises the sensor
layout around the pavement the user is actually studying.
"""

import multiprocessing

import numpy as np
from scipy.stats import qmc
from scipy.optimize import differential_evolution

import WuWan_pavement_forward
import WuWan_pavement_montecarlo

P_LAYERS = 5
N_SENSORS = 10


class SLOResult:
    """Plain container for the optimisation outputs (see optimize_sensor_layout)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def optimize_sensor_layout(arr_main, arr_noise, *,
                            num_fixed=3, r_min=300.0, r_max=3000.0, min_gap=100.0,
                            n_saa=32, np_mult=10, maxiter=200, tol=1e-5, seed=42,
                            eps_num=0.0, callback=None):
    """Run the robust D-optimal sensor-layout search.

    Parameters
    ----------
    arr_main, arr_noise : as built by InputFormsManager (11x8 / 11x7).
    num_fixed : how many of the first `num_fixed` evaluation points (in
        the order entered, by r [mm]) are held fixed; the remaining
        (10 - num_fixed) sensors are free and re-optimised.
    r_min, r_max, min_gap : search range and minimum spacing [mm] for
        the free sensors.
    n_saa : number of Sobol log-uniform modulus samples used to
        approximate the robust objective E[ln det FIM].
    np_mult, maxiter, tol, seed : Differential Evolution settings
        (population = np_mult * num_free, polish disabled).
    eps_num : numerical regularisation added to the FIM diagonal.
    callback(step, value, best, convergence) : optional progress hook,
        called once per DE iteration (value/best in natural-log units).

    Returns
    -------
    SLOResult with initial/final sensor positions, log10 det FIM and
    condition number at the true moduli, the robust SAA objective
    before/after, and the DE convergence history (D-efficiency and
    confidence-ellipsoid volume ratio vs. iteration).
    """
    arr_main = np.array(arr_main, dtype=np.float64, copy=True)
    arr_noise = np.array(arr_noise, dtype=np.float64, copy=True)

    P = P_LAYERS
    k_free = N_SENSORS - num_fixed
    if not (1 <= num_fixed < N_SENSORS):
        raise ValueError(f"num_fixed must be between 1 and {N_SENSORS - 1}.")

    initial_pos = arr_main[1:, 6].copy()
    fixed = initial_pos[:num_fixed].copy()
    entered_moduli = arr_main[2:7, 1].copy()

    E_lower = arr_noise[2:7, 1].copy()
    E_upper = arr_noise[2:7, 2].copy()
    if np.any(E_lower <= 0) or np.any(E_upper <= E_lower):
        raise ValueError("Modulus prior bounds (layer noise lower/upper) must be positive with lower < upper.")
    if np.any(arr_noise[1:, 6] <= 0):
        raise ValueError("Deflection noise level must be > 0 for every evaluation point (needed by the FIM weight).")

    # Modulus is UNKNOWN — it is searched for within [E_lower, E_upper] (the SAA robust
    # objective already samples the whole prior). A blank/0 "Initial Modulus" entry just
    # means "no starting guess" -> fall back to the log-center of the prior range, exactly
    # like Back Calculation's blank-initial-modulus convention. This `eval_moduli` is the
    # single representative point used for the deterministic D-criterion (logdet10/cond)
    # and for generating the synthetic deflections in the before/after Monte Carlo.
    log_center = np.sqrt(E_lower * E_upper)
    eval_moduli = np.where(entered_moduli > 0, entered_moduli, log_center)

    # ---- fixed normalisation constant, evaluated once at the initial layout ----
    arr_main[1:, 6] = initial_pos
    arr_main[2:7, 1] = eval_moduli
    ret0 = WuWan_pavement_forward.Calculation(arr_main, calc_grad=True)
    rel2_0 = (arr_noise[1:, 6] / (ret0.result_displacement * 1000.0)) ** 2
    scale = float(np.max(1.0 / rel2_0))

    def fim_matrix(positions, moduli):
        moduli = np.asarray(moduli, dtype=float)
        arr_main[1:, 6] = positions
        arr_main[2:7, 1] = moduli
        ret = WuWan_pavement_forward.Calculation(arr_main, calc_grad=True)
        n = ret.result_displacement.shape[0]
        d = np.repeat(ret.result_displacement[:, None], P, axis=1).T
        E = np.repeat(moduli[:, None], n, axis=1)
        J = (ret.J_E.T * E / d).T
        rel2 = (arr_noise[1:, 6] / (ret.result_displacement * 1000.0)) ** 2
        W = np.linalg.inv(np.diag(rel2)) / scale
        return J.T @ W @ J + eps_num * np.eye(P)

    def fim_logdet_nat(positions, moduli):
        sign, ld = np.linalg.slogdet(fim_matrix(positions, moduli))
        return ld if sign > 0 else -np.inf

    def logdet10(M):
        sign, ld = np.linalg.slogdet(M)
        return ld / np.log(10.0) if sign > 0 else -np.inf

    sobol = qmc.Sobol(d=P, scramble=True, seed=seed)
    u = sobol.random(n_saa)
    e_samples = 10 ** (np.log10(E_lower) + u * (np.log10(E_upper) - np.log10(E_lower)))

    def saa_neg_meanlogdet_pos(positions):
        lds = [fim_logdet_nat(positions, e_samples[s]) for s in range(n_saa)]
        lds = [v for v in lds if np.isfinite(v)]
        if len(lds) < 0.5 * n_saa:
            return 1e9
        return -float(np.mean(lds))

    g_max = r_max - r_min - (k_free - 1) * min_gap
    if g_max <= 0:
        raise ValueError("Search range [r_min, r_max] is too small for the number of free "
                          "sensors and the minimum gap; widen the range or reduce min gap.")

    def gaps_to_positions(g):
        pos = np.empty(k_free)
        pos[0] = r_min + g[0]
        for k in range(1, k_free):
            pos[k] = pos[k - 1] + min_gap + g[k]
        return pos

    def pos_to_gap(pos_free):
        g = np.empty(k_free)
        g[0] = pos_free[0] - r_min
        for k in range(1, k_free):
            g[k] = pos_free[k] - pos_free[k - 1] - min_gap
        return np.clip(g, 0.0, None)

    def objective_smooth(g, soft=1.0):
        pos_free = gaps_to_positions(g)
        overflow = max(0.0, pos_free[-1] - r_max)
        if overflow > 0:
            return 1e6 + soft * overflow ** 2
        return saa_neg_meanlogdet_pos(np.concatenate((fixed, pos_free)))

    # ---- initial DE population (member 0 = the reference free positions) ----
    rng_init = np.random.default_rng(seed)
    n_pop = max(np_mult * k_free, 4)
    init_pos_list = []
    guard = 0
    while len(init_pos_list) < n_pop and guard < 200000:
        guard += 1
        x = np.sort(rng_init.uniform(r_min, r_max, k_free))
        if np.all(np.diff(np.concatenate((fixed, x))) >= min_gap):
            init_pos_list.append(x)
    if len(init_pos_list) < n_pop:
        raise RuntimeError("Could not build a valid initial population; widen [r_min, r_max] or reduce min gap.")
    init_pos_pop = np.array(init_pos_list)

    ref_free = np.clip(initial_pos[num_fixed:], r_min, r_max)
    ref_free = np.maximum.accumulate(ref_free)
    if np.all(np.diff(np.concatenate((fixed, ref_free))) >= min_gap - 1e-9):
        init_pos_pop[0] = ref_free
    init_pop_g = np.array([pos_to_gap(p) for p in init_pos_pop])

    # ---- initial-layout metrics (at the representative evaluation point) ----
    m_init = fim_matrix(initial_pos, eval_moduli)
    init_logdet10 = logdet10(m_init)
    init_cond = np.linalg.cond(m_init)
    initial_saa = saa_neg_meanlogdet_pos(initial_pos)

    # ---- Differential Evolution ----
    history, best_hist = [], []
    state = {'step': 0, 'best': np.inf}

    def de_callback(xk, convergence=None):
        state['step'] += 1
        val = objective_smooth(xk)
        if val < state['best']:
            state['best'] = val
        history.append(val / np.log(10))
        best_hist.append(state['best'] / np.log(10))
        if callback is not None:
            callback(state['step'], val, state['best'], convergence)

    res_de = differential_evolution(
        func=objective_smooth,
        bounds=[(0.0, g_max)] * k_free,
        init=init_pop_g,
        strategy='best1bin',
        popsize=np_mult,
        mutation=(0.5, 1.0),
        recombination=0.9,
        maxiter=maxiter,
        tol=tol,
        seed=seed,
        polish=False,
        callback=de_callback,
        disp=False,
    )

    final_pos = np.concatenate((fixed, gaps_to_positions(res_de.x)))
    m_final = fim_matrix(final_pos, eval_moduli)
    final_logdet10 = logdet10(m_final)
    final_cond = np.linalg.cond(m_final)
    final_saa = float(res_de.fun)

    ln10 = np.log(10.0)
    if best_hist:
        neg_lndet = np.concatenate(([initial_saa], np.array(best_hist) * ln10))
    else:
        neg_lndet = np.array([initial_saa])
    delta = initial_saa - neg_lndet
    d_eff = np.exp(delta / P)
    vol_ratio = np.exp(-delta / 2.0) * 100.0

    return SLOResult(
        initial_pos=initial_pos, final_pos=final_pos, fixed=fixed, num_fixed=num_fixed,
        eval_moduli=eval_moduli,
        init_logdet10=init_logdet10, init_cond=init_cond, initial_saa=initial_saa,
        final_logdet10=final_logdet10, final_cond=final_cond, final_saa=final_saa,
        history=np.asarray(history), best_hist=np.asarray(best_hist),
        d_eff=d_eff, vol_ratio=vol_ratio,
        d_eff_final=float(d_eff[-1]), vol_final=float(vol_ratio[-1]),
        n_iter=state['step'], de_message=str(res_de.message),
        scale=scale, k_free=k_free, r_min=r_min, r_max=r_max, min_gap=min_gap, n_saa=n_saa,
    )


def run_monte_carlo_at(arr_main, arr_noise, positions, gen_moduli, num_threads=None):
    """Synthetic-data Monte Carlo back-calculation at a given sensor layout.

    Generates the noise-free deflection basin at `positions` from
    `gen_moduli` (forward model) -- typically the SLOResult.eval_moduli,
    i.e. the representative/assumed pavement modulus (entered value, or
    log-center of the prior where left blank). Writes the synthetic basin
    into arr_main as the "measured" deflections, starts the inversion from
    the geometric centre of the modulus prior (arr_noise lower/upper
    bounds), and runs the same triangular-noise Monte Carlo used by Back
    Calculation. Used to compare the back-calculated modulus distribution
    before vs. after optimising the sensor layout (initial_pos vs. final_pos).
    """
    arr_main = np.array(arr_main, dtype=np.float64, copy=True)
    arr_noise = np.array(arr_noise, dtype=np.float64, copy=True)
    gen_moduli = np.asarray(gen_moduli, dtype=float)

    arr_main[1:, 6] = positions
    arr_main[2:7, 1] = gen_moduli
    ret = WuWan_pavement_forward.Calculation(arr_main, calc_grad=False)
    arr_main[1:, 7] = ret.result_displacement * 1000.0

    e_lower = arr_noise[2:7, 1]
    e_upper = arr_noise[2:7, 2]
    arr_main[2:7, 1] = np.sqrt(e_lower * e_upper)   # inversion start = prior geometric centre

    nt = num_threads if num_threads else multiprocessing.cpu_count()
    return WuWan_pavement_montecarlo.ParalleMonteCarlo(arr_main, arr_noise, nt)
