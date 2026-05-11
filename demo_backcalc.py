"""
Demo: Elastic Backcalculation - M85 CKt szelvény
Rétegrend:
  Layer 1: Aszfalt           h =  200 mm,  nu = 0.35   → szabad: E1
  Layer 2: CKt               h =  200 mm,  nu = 0.30   → szabad: E2
  Layer 3: Földmű (felső)    h =  500 mm,  nu = 0.40   → szabad: E3
  Layer 4: Földmű (alsó)     h =  500 mm,  nu = 0.40   → szabad: E4  (E3 ≈ E4)
  Layer 5: Végtelen féltér   h =  ∞,       nu = 0.45   → RÖGZÍTETT: 1000 MPa (szűk határ)
"""
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, r'D:\Repositories\WuWan')
sys.path.insert(0, r'D:\Repositories\WuWan\src')
import WuWan_pavement_forward
from WuWan_pavement_inverse import ForwardModelLogCached

# ─────────────────────────────────────────────────────────────────────────────
# Konstansok
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_PATH  = r'D:\OneDrive\MKIF\BAKSIMPLEX\m85_ckt\Elastic_Backcalculation_Input.xlsm'
OUTPUT_PATH = r'D:\OneDrive\MKIF\BAKSIMPLEX\m85_ckt\Elastic_Backcalculation_Output_demo.xlsx'

R_SENSORS_MM = np.array([0.0, 200.0, 300.0, 450.0, 600.0, 900.0, 1200.0])
N_SENSORS    = len(R_SENSORS_MM)
RADIUS_MM    = 150.0
N_EVAL       = 10      # WuWan fix 10 kiértékelési pontot vár

# Réteg geometria
H  = [200.0, 200.0, 500.0, 500.0,   0.0]   # vastagság [mm], 0 = féltér
NU = [0.35,  0.30,  0.40,  0.40,   0.45]   # Poisson-szám

# 5 szabad paraméter: [E1, E2, E3, E4, E5]
# E5 szűk határ → közel rögzített 1000 MPa-on
# E2 (CKt): 100 kg/m³ cement ≈ 4.8% → max ~12 000 MPa érett állapotban
X_INIT  = np.array([10000.0,  6000.0, 275.0, 275.0, 1000.0])
X_LOWER = np.array([ 1000.0,   500.0, 100.0, 100.0,  995.0])
X_UPPER = np.array([30000.0, 15000.0, 600.0, 600.0, 1005.0])

# ─────────────────────────────────────────────────────────────────────────────
# Adatok beolvasása
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_excel(EXCEL_PATH, sheet_name='Deflection', header=None, engine='openpyxl')

data       = df_raw.iloc[3:].reset_index(drop=True)
N_MEAS     = len(data)
stress_mpa = data.iloc[:, 1].values.astype(float) / 1000.0   # kPa → MPa
defl_um    = data.iloc[:, 2:9].values.astype(float)           # (N_MEAS × 7) [μm]

print(f"Beolvasva: {N_MEAS} mérés")
print(f"Szenzor pozíciók [mm]: {R_SENSORS_MM}")
print(f"Stressz tartomány: {stress_mpa.min():.3f} – {stress_mpa.max():.3f} MPa\n")

# 10 kiértékelési pont: 7 szenzor + 3 ismételt utolsó pozíció
r_eval = np.concatenate([R_SENSORS_MM, np.full(N_EVAL - N_SENSORS, R_SENSORS_MM[-1])])

# ─────────────────────────────────────────────────────────────────────────────
# WuWan input tömb felépítése (11×8) – C++ formátum
# ─────────────────────────────────────────────────────────────────────────────
def build_arr(E_all, stress):
    """(11×8) C++ input tömb. ForwardModelLogCached .T transzponálva kapja."""
    arr = np.zeros((11, 8))
    for i in range(5):
        arr[i + 2, 0] = i + 1
        arr[i + 2, 1] = E_all[i]
        arr[i + 2, 2] = NU[i]
        arr[i + 2, 3] = H[i]
    for j in range(N_EVAL):
        arr[j + 1, 5] = j + 1
        arr[j + 1, 6] = r_eval[j]
    arr[10, 1] = stress
    arr[10, 3] = RADIUS_MM
    return arr

# ─────────────────────────────────────────────────────────────────────────────
# Visszaszámítás – minden mérési pontra
# ─────────────────────────────────────────────────────────────────────────────
log_lower   = np.log(X_LOWER)
log_upper   = np.log(X_UPPER)
x_prior_log = (log_lower + log_upper) / 2
sigma_prior = (log_upper - log_lower) / 4

records = []

for i in range(N_MEAS):
    stress  = stress_mpa[i]
    d_mm_7  = defl_um[i] * 1e-3   # μm → mm, 7 szenzor

    # 10 célértékre padded (a cost fn. mind a 10 pontot használja)
    d_mm_10 = np.concatenate([d_mm_7, np.full(N_EVAL - N_SENSORS, d_mm_7[-1])])

    # arr_template: (8×11) = C++ (11×8) transzponáltja
    # ForwardModelLogCached belsőleg self.arr[1, 2:7] = E értékek,
    # majd self.arr.T-t adja a C++ modulnak
    arr_template = build_arr(X_INIT, stress).T.copy()

    model = ForwardModelLogCached(
        arr_template, d_mm_10,
        x_prior_log, sigma_prior,
        forward_module=WuWan_pavement_forward
    )

    t0 = time.time()
    opt = least_squares(
        model.fun,
        np.log(X_INIT),
        jac=model.jac,
        bounds=(log_lower, log_upper),
        method='trf',
        verbose=0,
        max_nfev=150,
        ftol=1e-9, xtol=1e-9, gtol=1e-9
    )
    elapsed = time.time() - t0

    x_res   = np.exp(opt.x)
    rms_pct = np.sqrt(np.mean(opt.fun**2)) * 100.0

    records.append({
        'No':               i + 1,
        'Stress [MPa]':     round(stress, 4),
        'E1_Aszfalt [MPa]': round(x_res[0], 1),
        'E2_CKt [MPa]':     round(x_res[1], 1),
        'E3_Folmu_f [MPa]': round(x_res[2], 1),
        'E4_Folmu_a [MPa]': round(x_res[3], 1),
        'E5_Felter [MPa]':  round(x_res[4], 1),
        'RMS_hiba [%]':     round(rms_pct, 3),
        'Ido [s]':          round(elapsed, 2),
        'nfev':             opt.nfev,
    })

    print(
        f"[{i+1:2d}/{N_MEAS}] "
        f"E1={x_res[0]:7.0f}  E2={x_res[1]:7.0f}  "
        f"E3={x_res[2]:6.1f}  E4={x_res[3]:6.1f}  E5={x_res[4]:.0f}  MPa  |  "
        f"RMS={rms_pct:.2f}%  {elapsed:.1f}s"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Eredmények mentése
# ─────────────────────────────────────────────────────────────────────────────
df_out = pd.DataFrame(records)
df_out.to_excel(OUTPUT_PATH, index=False)
print(f"\nKész. Eredmények mentve: {OUTPUT_PATH}")
print(df_out[['No', 'E1_Aszfalt [MPa]', 'E2_CKt [MPa]',
              'E3_Folmu_f [MPa]', 'E4_Folmu_a [MPa]',
              'E5_Felter [MPa]', 'RMS_hiba [%]']].to_string(index=False))
