"""
Demo: Elastic Backcalculation - M85 CKt szelvény

Feladat:
  - minden mérésre külön visszaszámítás
  - minden méréshez külön teknőábra mentése
  - a teknő 0-1800 mm között legyen kirajzolva
  - a mért pontok kerüljenek rá az ábrára
  - az alapértelmezett modultartományok ne szorítsák a megoldást a határokra
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import WuWan_pavement_forward
from WuWan_pavement_inverse import ForwardModelLogCached

# ─────────────────────────────────────────────────────────────────────────────
# Konstansok
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_PATH = PROJECT_ROOT / "Elastic_Backcalculation_Input.xlsm"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "Elastic_Backcalculation_Output_demo.xlsx"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "demo_figure" / "backcalc_basins"

R_SENSORS_MM = np.array([0.0, 200.0, 300.0, 450.0, 600.0, 900.0, 1200.0])
N_SENSORS = len(R_SENSORS_MM)
RADIUS_MM = 150.0
N_EVAL = 10
R_PLOT_MM = np.linspace(0.0, 1800.0, 61)

# Structure ful alapjan beallitott 5 reteg:
#   L1: 230 mm osszevont aszfalt
#   L2: 200 mm CKt
#   L3: 1000 mm foldmu / epitett toltes
#   L4: 500 mm also foldmu
#   L5: felter
H_FIXED = [230.0, 200.0, 1000.0, 500.0, 0.0]
NU = [0.35, 0.25, 0.45, 0.45, 0.45]

# Negy fuggetlen parameter: E3 es E4 azonos modulussal fut.
#   E1 Aszfalt [MPa]      : 5 000-15 000
#   E2 CKt [MPa]          : 1 500-17 500
#   E34 Foldmu+also foldmu:   300-600
#   E5 Felter [MPa]       :   100-1 600
X_INIT = np.array([10000.0, 5000.0, 450.0, 300.0])
X_LOWER = np.array([5000.0, 1500.0, 300.0, 100.0])
X_UPPER = np.array([15000.0, 17500.0, 600.0, 1600.0])

# Mr becslés a 900 mm-es szenzorból (Boussinesq féltér, ν=0.45)
# Mr [MPa] = (1-0.45²) * σ [kPa] * 9.8 / (π * 900 [mm] * D900 [μm] / 1000)
MR_NU = 0.45
MR_R_MM = 900.0   # a 900 mm-es szenzor r értéke a képletben
IDX_R900 = 5      # R_SENSORS_MM[5] = 900 mm
DEFAULT_MAX_NFEV = 2000
DEFAULT_FTOL = 1e-12
DEFAULT_XTOL = 1e-12
DEFAULT_GTOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WuWan demo backcalculation")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input Excel path, Deflection munkalappal.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output Excel path.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Kimeneti mappa a külön teknőábrákhoz.",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=DEFAULT_MAX_NFEV,
        help="Maximális függvénykiértékelések száma mérésenként.",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=DEFAULT_FTOL,
        help="least_squares ftol.",
    )
    parser.add_argument(
        "--xtol",
        type=float,
        default=DEFAULT_XTOL,
        help="least_squares xtol.",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=DEFAULT_GTOL,
        help="least_squares gtol.",
    )
    return parser.parse_args()


class MeasuredSensorModel:
    """A WuWan 10 pontot var, de csak a 7 tenyleges szenzorra illesztunk."""

    def __init__(self, base_model: ForwardModelLogCached):
        self.base_model = base_model

    def fun(self, log_x: np.ndarray) -> np.ndarray:
        return self.base_model.fun(log_x)[:N_SENSORS]

    def jac(self, log_x: np.ndarray) -> np.ndarray:
        return self.base_model.jac(log_x)[:N_SENSORS]


class TiedE3E4Model:
    """4 fuggetlen parameterbol 5 reteget kepez: E3 es E4 azonos."""

    def __init__(self, base_model: ForwardModelLogCached):
        self.base_model = base_model

    @staticmethod
    def expand_log_x(log_x_tied: np.ndarray) -> np.ndarray:
        return np.array(
            [log_x_tied[0], log_x_tied[1], log_x_tied[2], log_x_tied[2], log_x_tied[3]]
        )

    @staticmethod
    def expand_x(x_tied: np.ndarray) -> np.ndarray:
        return np.array([x_tied[0], x_tied[1], x_tied[2], x_tied[2], x_tied[3]])

    def fun(self, log_x_tied: np.ndarray) -> np.ndarray:
        return self.base_model.fun(self.expand_log_x(log_x_tied))[:N_SENSORS]

    def jac(self, log_x_tied: np.ndarray) -> np.ndarray:
        jac_full = self.base_model.jac(self.expand_log_x(log_x_tied))[:N_SENSORS]
        return np.column_stack(
            [
                jac_full[:, 0],
                jac_full[:, 1],
                jac_full[:, 2] + jac_full[:, 3],
                jac_full[:, 4],
            ]
        )

    def full_fun(self, log_x_tied: np.ndarray) -> np.ndarray:
        return self.base_model.fun(self.expand_log_x(log_x_tied))


def build_arr(E_all: np.ndarray, stress: float, r_positions: np.ndarray) -> np.ndarray:
    """WuWan (11x8) input tomb. E_all = [E1,E2,E3,E4,E5]."""
    arr = np.zeros((11, 8))
    for i in range(5):
        arr[i + 2, 0] = i + 1
        arr[i + 2, 1] = E_all[i]
        arr[i + 2, 2] = NU[i]
        arr[i + 2, 3] = H_FIXED[i]
    for j in range(N_EVAL):
        arr[j + 1, 5] = j + 1
        arr[j + 1, 6] = r_positions[j]
    arr[10, 1] = stress
    arr[10, 3] = RADIUS_MM
    return arr


def build_eval_positions() -> np.ndarray:
    return np.concatenate([R_SENSORS_MM, np.full(N_EVAL - N_SENSORS, R_SENSORS_MM[-1])])


def forward_curve(E_all: np.ndarray, stress: float, r_positions: np.ndarray) -> np.ndarray:
    """Forward model futtatása tetszőleges r pontokra, 10-es batch-ekben."""
    u_all = []
    for start in range(0, len(r_positions), N_EVAL):
        batch = r_positions[start:start + N_EVAL]
        r_batch = np.concatenate([batch, np.full(N_EVAL - len(batch), batch[-1])])
        arr = build_arr(E_all, stress, r_batch)
        result = WuWan_pavement_forward.Calculation(
            np.ascontiguousarray(arr, dtype=np.float64), calc_grad=False
        )
        u_all.extend(np.asarray(result.result_displacement[: len(batch)]))
    return np.asarray(u_all) * 1e3


def save_basin_plot(
    plot_path: Path,
    measurement_no: int,
    stress: float,
    E_all: np.ndarray,
    measured_defl_um: np.ndarray,
) -> None:
    predicted_defl_um = forward_curve(E_all, stress, R_PLOT_MM)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        R_PLOT_MM,
        predicted_defl_um,
        color="#0b5d8f",
        linewidth=2.0,
        label="Visszaszámított teknő",
    )
    ax.scatter(
        R_SENSORS_MM,
        measured_defl_um,
        color="#c84c09",
        s=36,
        zorder=3,
        label="Mért pontok",
    )

    for r_mm in R_SENSORS_MM:
        ax.axvline(r_mm, color="#d0d0d0", linewidth=0.7, linestyle="--", zorder=0)

    ax.set_xlim(0, 1800)
    ax.invert_yaxis()
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_xlabel("Távolság r [mm]")
    ax.set_ylabel("Behajlás [μm]")
    ax.set_title(f"Teknő #{measurement_no} | terhelés = {stress:.3f} MPa")
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    plot_dir = args.plot_dir.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Nem található a bemeneti Excel fájl: {input_path}\n"
            "Adj meg másik fájlt a --input kapcsolóval."
        )

    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_excel(input_path, sheet_name="Deflection", header=None, engine="openpyxl")
    data = df_raw.iloc[3:].reset_index(drop=True)
    n_meas = len(data)
    stress_mpa = data.iloc[:, 1].values.astype(float) / 1000.0
    defl_um = data.iloc[:, 2:9].values.astype(float)

    print(f"Beolvasva: {n_meas} mérés")
    print(f"Szenzor pozíciók [mm]: {R_SENSORS_MM}")
    print(f"Stressz tartomány: {stress_mpa.min():.3f} - {stress_mpa.max():.3f} MPa")
    print(f"Kimeneti Excel: {output_path}")
    print(f"Külön teknőábrák mappája: {plot_dir}\n")
    print(
        "Optimalizáció: "
        f"max_nfev={args.max_nfev}, ftol={args.ftol:.1e}, "
        f"xtol={args.xtol:.1e}, gtol={args.gtol:.1e}\n"
    )

    r_eval = build_eval_positions()

    # Log-ter korlatok: [E1, E2, E34, E5]
    log_lower = np.log(X_LOWER)
    log_upper = np.log(X_UPPER)

    records = []
    current_x_init = X_INIT.copy()

    for i in range(n_meas):
        stress = stress_mpa[i]
        stress_kpa = stress * 1000.0
        d_um_7 = defl_um[i]
        d_mm_7 = d_um_7 * 1e-3
        d_mm_10 = np.concatenate([d_mm_7, np.full(N_EVAL - N_SENSORS, d_mm_7[-1])])

        # Mr becslés a 900 mm-es szenzorból tajekoztato ertekkent.
        d_r900_um = d_um_7[IDX_R900]
        mr_mpa = ((1.0 - MR_NU**2) * stress_kpa * 9.8) / (
            np.pi * MR_R_MM * d_r900_um / 1000.0
        )
        mr_mpa = float(np.clip(mr_mpa, 30.0, 500.0))
        arr_template = build_arr(TiedE3E4Model.expand_x(current_x_init), stress, r_eval).T.copy()

        base_model = ForwardModelLogCached(
            arr_template,
            d_mm_10,
            (np.log(X_LOWER) + np.log(X_UPPER)) / 2,
            (np.log(X_UPPER) - np.log(X_LOWER)) / 4,
            forward_module=WuWan_pavement_forward,
        )
        tied_model = TiedE3E4Model(base_model)

        t0 = time.time()
        opt = least_squares(
            tied_model.fun,
            np.log(current_x_init),
            jac=tied_model.jac,
            bounds=(log_lower, log_upper),
            method="trf",
            verbose=0,
            x_scale="jac",
            max_nfev=args.max_nfev,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
        )
        elapsed = time.time() - t0

        x_tied_res = np.exp(opt.x)
        x_res = TiedE3E4Model.expand_x(x_tied_res)
        E1, E2, E3, E4, E5 = x_res
        current_x_init = np.clip(x_tied_res, X_LOWER, X_UPPER)
        residual_7 = tied_model.fun(opt.x)
        residual_10 = tied_model.full_fun(opt.x)
        rms_7_pct = np.sqrt(np.mean(residual_7**2)) * 100.0
        rms_10_pct = np.sqrt(np.mean(residual_10**2)) * 100.0
        plot_path = plot_dir / f"tekno_{i + 1:03d}.png"

        save_basin_plot(
            plot_path=plot_path,
            measurement_no=i + 1,
            stress=stress,
            E_all=x_res,
            measured_defl_um=d_um_7,
        )

        records.append(
            {
                "No": i + 1,
                "Stress [MPa]": round(stress, 4),
                "Mr_becsl [MPa]": round(mr_mpa, 1),
                "E1_Aszfalt [MPa]": round(E1, 1),
                "E2_CKt [MPa]": round(E2, 1),
                "E3_Foldmu_1000 [MPa]": round(E3, 1),
                "E4_Also_foldmu [MPa]": round(E4, 1),
                "E5_Felter [MPa]": round(E5, 1),
                "RMS_7_mert_pont [%]": round(rms_7_pct, 3),
                "RMS_10_WuWan_pont [%]": round(rms_10_pct, 3),
                "Ido [s]": round(elapsed, 2),
                "nfev": opt.nfev,
                "status": int(opt.status),
                "message": opt.message,
                "Plot": str(plot_path),
            }
        )

        print(
            f"[{i + 1:2d}/{n_meas}] "
            f"Aszfalt={E1:6.0f}  CKt={E2:6.0f}  Foldmu={E3:5.0f}  "
            f"Also={E4:5.0f}  Felter={E5:5.0f}  Mr={mr_mpa:5.1f} MPa  |  "
            f"RMS7={rms_7_pct:.2f}%  RMS10={rms_10_pct:.2f}%  {elapsed:.1f}s  nfev={opt.nfev}"
        )

    df_out = pd.DataFrame(records)
    df_out.to_excel(output_path, index=False)

    print(f"\nKész. Eredmények mentve: {output_path}")
    print(f"Külön teknőábrák: {plot_dir}")
    print(
        df_out[
            [
                "No",
                "Mr_becsl [MPa]",
                "E1_Aszfalt [MPa]",
                "E2_CKt [MPa]",
                "E3_Foldmu_1000 [MPa]",
                "E4_Also_foldmu [MPa]",
                "E5_Felter [MPa]",
                "RMS_7_mert_pont [%]",
                "RMS_10_WuWan_pont [%]",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
