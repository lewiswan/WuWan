"""
Visszaszámítás eredményének megjelenítése: forward model teknő + mért pontok.
Szükséges: demo_backcalc.py már lefutott (Output Excel létezik).
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, r'D:\Repositories\WuWan')
import WuWan_pavement_forward

# ─────────────────────────────────────────────────────────────────────────────
EXCEL_INPUT  = r'D:\OneDrive\MKIF\BAKSIMPLEX\m85_ckt\Elastic_Backcalculation_Input.xlsm'
EXCEL_OUTPUT = r'D:\OneDrive\MKIF\BAKSIMPLEX\m85_ckt\Elastic_Backcalculation_Output_demo.xlsx'
OUTPUT_PNG   = r'D:\OneDrive\MKIF\BAKSIMPLEX\m85_ckt\Elastic_Backcalculation_Tekno.png'

R_SENSORS_MM = np.array([0.0, 200.0, 300.0, 450.0, 600.0, 900.0, 1200.0])
RADIUS_MM    = 150.0
H  = [200.0, 200.0, 500.0, 500.0,   0.0]
NU = [0.35,  0.30,  0.40,  0.40,   0.45]

# ─────────────────────────────────────────────────────────────────────────────
# Adatok beolvasása
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_excel(EXCEL_INPUT, sheet_name='Deflection', header=None, engine='openpyxl')
data       = df_raw.iloc[3:].reset_index(drop=True)
stress_mpa = data.iloc[:, 1].values.astype(float) / 1000.0
defl_um    = data.iloc[:, 2:9].values.astype(float)   # (N × 7) [μm]

df_res = pd.read_excel(EXCEL_OUTPUT)
N_MEAS = len(df_res)

# ─────────────────────────────────────────────────────────────────────────────
# Forward model segédfüggvény – tetszőleges r pozíciókra, 10-es batchekben
# ─────────────────────────────────────────────────────────────────────────────
def build_arr(E_all, stress, r_batch_10):
    arr = np.zeros((11, 8))
    for i in range(5):
        arr[i + 2, 0] = i + 1
        arr[i + 2, 1] = E_all[i]
        arr[i + 2, 2] = NU[i]
        arr[i + 2, 3] = H[i]
    for j in range(10):
        arr[j + 1, 5] = j + 1
        arr[j + 1, 6] = r_batch_10[j]
    arr[10, 1] = stress
    arr[10, 3] = RADIUS_MM
    return arr


def forward_curve(E_all, stress, r_positions):
    """Forward model futtatása tetszőleges r pontokra (10-es batchekben)."""
    u_all = []
    for start in range(0, len(r_positions), 10):
        batch = r_positions[start:start + 10]
        n = len(batch)
        r_batch = np.concatenate([batch, np.full(10 - n, batch[-1])])
        arr = build_arr(E_all, stress, r_batch)
        res = WuWan_pavement_forward.Calculation(
            np.ascontiguousarray(arr, dtype=np.float64), calc_grad=False
        )
        u_all.extend(np.array(res.result_displacement)[:n])
    return np.array(u_all) * 1e3   # mm → μm

# ─────────────────────────────────────────────────────────────────────────────
# Rajzolás
# ─────────────────────────────────────────────────────────────────────────────
N_PLOT   = 50
r_smooth = np.linspace(0.0, 1800.0, N_PLOT)

colors = cm.plasma(np.linspace(0.1, 0.9, N_MEAS))

fig, ax = plt.subplots(figsize=(12, 7))

for i in range(N_MEAS):
    row    = df_res.iloc[i]
    stress = stress_mpa[i]
    E_all  = np.array([
        row['E1_Aszfalt [MPa]'],
        row['E2_CKt [MPa]'],
        row['E3_Folmu_f [MPa]'],
        row['E4_Folmu_a [MPa]'],
        row['E5_Felter [MPa]'],
    ])

    # Sima görbe (forward model)
    u_smooth = forward_curve(E_all, stress, r_smooth)
    ax.plot(r_smooth, u_smooth, color=colors[i], alpha=0.45, linewidth=1.0)

    # Mért pontok
    ax.scatter(R_SENSORS_MM, defl_um[i], color=colors[i], s=18, zorder=3,
               alpha=0.8, edgecolors='none')

# Szenzor pozíciók jelölése
for r in R_SENSORS_MM:
    ax.axvline(r, color='#aaaaaa', linewidth=0.5, linestyle='--', zorder=0)

sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(vmin=1, vmax=N_MEAS))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Mérés sorszáma', fontsize=10)

ax.set_xlabel('Szenzor távolság r [mm]', fontsize=11)
ax.set_ylabel('Behajlás [μm]', fontsize=11)
ax.set_title('M85 CKt – visszaszámított forward model teknő vs. mért behajlások', fontsize=12)
ax.set_xlim(0, 1800)
ax.invert_yaxis()   # teknő: lefelé nagyobb
ax.grid(True, linewidth=0.4, alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"Ábra mentve: {OUTPUT_PNG}")
plt.show()
