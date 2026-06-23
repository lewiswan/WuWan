import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkintertable import TableCanvas, TableModel
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import datetime
import threading
import multiprocessing
from scipy import stats
from tabulate import tabulate

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

import WuWan_pavement_forward
import WuWan_pavement_inverse
import WuWan_pavement_montecarlo
import WuWan_pavement_slo


# FlatButton
class FlatButton(tk.Label):
    def __init__(self, parent, text='', command=None,
                 bg='#2C5282', fg='white', activebg=None, activefg=None,
                 font=("Segoe UI", 10, "bold"), padx=10, pady=4,
                 anchor='center', width=0, height=0, state='normal',
                 disabledbg='#D6D3C7', disabledfg='#6B7280', **kwargs):
        super().__init__(parent, text=text, bg=bg, fg=fg, font=font,
                         padx=padx, pady=pady, anchor=anchor, **kwargs)
        if width:
            self.config(width=width)
        if height:
            self.config(height=height)
        self._command = command
        self._bg = bg
        self._fg = fg
        self._activebg = activebg or bg
        self._activefg = activefg or fg
        self._state = state
        self._disabledbg = disabledbg
        self._disabledfg = disabledfg
        self.bind('<Button-1>', self._on_click)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self._apply_state()

    def _apply_state(self):
        if self._state == 'disabled':
            self.config(bg=self._disabledbg, fg=self._disabledfg, cursor='')
        else:
            self.config(bg=self._bg, fg=self._fg, cursor='hand2')

    def _on_click(self, e):
        if self._state != 'disabled' and self._command:
            self._command()

    def _on_enter(self, e):
        if self._state != 'disabled':
            self.config(bg=self._activebg, fg=self._activefg)

    def _on_leave(self, e):
        if self._state != 'disabled':
            self.config(bg=self._bg, fg=self._fg)


# ToolTip 
class ToolTip:
    def __init__(self, widget, text, wraplength=460,
                 bg='#2C3E50', fg='#F7F5EE', font=("Segoe UI", 10)):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.bg = bg
        self.fg = fg
        self.font = font
        self.tipwindow = None
        widget.bind('<Enter>', self._show)
        widget.bind('<Leave>', self._hide)
        widget.bind('<ButtonPress>', self._hide)

    def _show(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass
        label = tk.Label(tw, text=self.text, justify='left',
                         bg=self.bg, fg=self.fg,
                         relief='solid', borderwidth=1,
                         font=self.font, wraplength=self.wraplength,
                         padx=12, pady=9)
        label.pack()

    def _hide(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


# Main Menu
class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title('WuWan v0.3 - Main Menu')
        self.root.geometry('600x600')
        self.root.configure(bg='#F5F2E8')

        self.main_frame = tk.Frame(root, bg='#F5F2E8', padx=20, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.main_frame, text="WuWan Analysis System",
                 font=("Georgia", 24, "bold"), bg='#F5F2E8', fg="#2C3E50").pack(pady=30)
        tk.Label(self.main_frame, text="Please Select Analysis Module",
                 font=("Georgia", 16), bg='#F5F2E8', fg="#5D6D7E").pack(pady=10)

        button_frame = tk.Frame(self.main_frame, bg='#F5F2E8')
        button_frame.pack(pady=30)

        FlatButton(button_frame, text="Forward Calculation",
                   command=self.open_forward_calculation,
                   font=("Segoe UI", 12, "bold"), bg="#2C5282", fg="white",
                   activebg="#1A365D", activefg="white",
                   width=25, height=3, relief=tk.FLAT, borderwidth=0).pack(pady=10)

        FlatButton(button_frame, text="Back Calculation",
                   command=self.open_back_calculation,
                   font=("Segoe UI", 12, "bold"), bg="#2D5F3F", fg="white",
                   activebg="#1E4029", activefg="white",
                   width=25, height=3, relief=tk.FLAT, borderwidth=0).pack(pady=10)

        FlatButton(button_frame, text="Sensor Location Optimization",
                   command=self.open_sensor_location_optimization,
                   font=("Segoe UI", 12, "bold"), bg="#8B6F47", fg="white",
                   activebg="#6F5736", activefg="white",
                   width=25, height=3, relief=tk.FLAT, borderwidth=0).pack(pady=10)

        tk.Label(self.main_frame, text="Version 0.3 | © 2026",
                 font=("Georgia", 9, "italic"), bg='#F5F2E8', fg="#A6A08D"
                 ).pack(side=tk.BOTTOM, pady=10)

        tk.Label(self.main_frame, text="WuWan (0.3) is an open-source computational tool for FWD analysis written in C++  and Python. It solves for surface deflections in a five-layered half-space under a single circular stress patch. The tool has three modules: (i) forward-calculation, in which the user inputs the layering composition and elastic properties of the half-space and the code produces deflections at ten offset distances from the center of the stress patch; (ii) back-calculation, in which the user inputs the layering composition and deflections (including noise levels) and the code produces the elastic moduli with uncertainty analysis based on a Monte-Carlo approach; and (iii) sensor optimization (forthcoming), in which the user inputs the layering composition and elastic properties (including noise levels), and the code produces a suggested spacing for the deflection sensors to optimize back-calculation results. ",
                 font=("Georgia", 10), bg='#F5F2E8', fg="#291A61", wraplength=500, justify='left').pack(pady=10)

    # Forward Calculation
    def open_forward_calculation(self):
        self.root.withdraw()
        self.forward_window = tk.Toplevel(self.root)
        self.forward_window.geometry('820x540')
        self.forward_window.title('WuWan v0.3 - Forward Calculation')
        self.forward_window.configure(bg='#F5F2E8')
        self.is_profile_open = False

        top_bar = tk.Frame(self.forward_window, height=42, bg="#2C3E50")
        top_bar.pack(side=tk.TOP, fill=tk.X)
        top_bar.pack_propagate(False)

        FlatButton(top_bar, text="← Return to Main Menu",
                   command=lambda: self.return_to_main(self.forward_window),
                   font=("Segoe UI", 9, "bold"), bg="#E74C3C", fg="white",
                   activebg="#C0392B", activefg="white",
                   relief=tk.FLAT, padx=14, pady=4).pack(side=tk.LEFT, padx=10, pady=8)

        self.profile_btn = FlatButton(top_bar, text="Show Profile Plot >>",
                                      command=self.toggle_profile_view,
                                      font=("Segoe UI", 9, "bold"),
                                      bg="#8B6F47", fg="white",
                                      activebg="#6F5736", activefg="white",
                                      relief=tk.FLAT, padx=14, pady=4)
        self.profile_btn.pack(side=tk.LEFT, padx=5, pady=8)

        content_container = tk.Frame(self.forward_window, bg='#F5F2E8')
        content_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.calc_frame = tk.Frame(content_container, width=820, bg='#FCFCFA',
                                   highlightbackground='#D6D3C7', highlightthickness=1)
        self.calc_frame.pack_propagate(0)
        self.calc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.table_app = TableApp(self.calc_frame)
        self.table_app.parent_menu = self

        self.plot_frame = tk.Frame(content_container, bg="white", width=800,
                                   highlightbackground='#D6D3C7', highlightthickness=1)
        self.plot_frame.pack_propagate(0)

        tk.Label(self.plot_frame,
                 text="Deflection profile will be shown here\nafter computation",
                 fg="#7f8c8d", bg="white", font=("Georgia", 14, "italic")
                 ).place(relx=0.5, rely=0.5, anchor="center")

        self.forward_window.protocol(
            "WM_DELETE_WINDOW",
            lambda: self.return_to_main(self.forward_window))

    # Back Calculation — ACADEMIC LAYOUT
    def open_back_calculation(self):
        self.root.withdraw()

        self.back_window = tk.Toplevel(self.root)
        self.back_window.geometry('960x800')
        self.back_window.title('WuWan v0.3 - Back Calculation')
        self.back_window.configure(bg='#F5F2E8')

        # --- Top toolbar ---
        top_bar = tk.Frame(self.back_window, height=42, bg="#2C3E50")
        top_bar.pack(side=tk.TOP, fill=tk.X)
        top_bar.pack_propagate(False)

        FlatButton(top_bar, text="← Main Menu",
                   command=lambda: self.return_to_main(self.back_window),
                   font=("Segoe UI", 9, "bold"),
                   bg="#E74C3C", fg="white",
                   activebg="#C0392B", activefg="white",
                   relief=tk.FLAT, padx=14, pady=4).pack(side=tk.LEFT, padx=10, pady=8)

        tk.Label(top_bar, text="◈  Back Calculation Module",
                 font=("Georgia", 12, "bold"),
                 bg="#2C3E50", fg="white").pack(side=tk.LEFT, padx=15)

        # --- Main container ---
        main_container = tk.Frame(self.back_window, bg='#F5F2E8')
        main_container.pack(fill=tk.BOTH, expand=True)

        # --- Left sidebar ---
        sidebar = tk.Frame(main_container, width=240, bg="#E8E4D8",
                           highlightbackground="#BDB7A4", highlightthickness=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # --- Right area: whiteboard + output ---
        right_area = tk.Frame(main_container, bg='#F5F2E8')
        right_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bottom output
        output_container = tk.Frame(right_area, bg="#FCFCFA", height=170)
        output_container.pack(side=tk.BOTTOM, fill=tk.X)
        output_container.pack_propagate(False)

        out_header = tk.Frame(output_container, bg="#2C3E50", height=24)
        out_header.pack(side=tk.TOP, fill=tk.X)
        out_header.pack_propagate(False)
        tk.Label(out_header, text="  ▼ Output Log",
                 font=("Segoe UI", 9, "bold"),
                 bg="#2C3E50", fg="white", anchor="w"
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.bc_output_text = tk.Text(output_container, height=10,
                                      bg="#FCFCFA", fg="#1B2631",
                                      font=("Consolas", 9),
                                      relief=tk.FLAT, borderwidth=0)
        self.bc_output_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Top whiteboard
        self.bc_whiteboard = tk.Frame(right_area, bg='#F5F2E8')
        self.bc_whiteboard.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Input forms manager ---
        self.forms = InputFormsManager(self.bc_whiteboard)

        # --- Result views ---
        self.bc_view_single = tk.Frame(self.bc_whiteboard, bg='#F5F2E8')
        self.bc_view_mc = tk.Frame(self.bc_whiteboard, bg='#F5F2E8')
        self._build_placeholder(self.bc_view_single,
                                "Click 'Run Single Calculation' to compute.\nResults will appear here.")
        self._build_placeholder(self.bc_view_mc,
                                "Click 'Run Monte Carlo' to compute.\nResults and violin plot will appear here.")

        self._build_sidebar(sidebar)
        self.bc_show('deflections')

        self.back_window.protocol(
            "WM_DELETE_WINDOW",
            lambda: self.return_to_main(self.back_window))

    def _build_sidebar(self, parent):
        tk.Label(parent, text="WuWan",
                 font=("Georgia", 32, "bold"),
                 bg="#E8E4D8", fg="#2C3E50", pady=8).pack(fill=tk.X)
        tk.Label(parent, text="Back Calculation",
                 font=("Georgia", 16, "italic"),
                 bg="#E8E4D8", fg="#5D6D7E", pady=2).pack(fill=tk.X)

        # --- INPUT DATA ---
        self._sidebar_section(parent, "INPUT DATA", "#2C5282")
        self._sidebar_btn(parent, "Ⅰ  Deflection Bowl & Loading System", lambda: self.bc_show('deflections'), "#2C5282")
        self._sidebar_btn(parent, "Ⅱ  Layered System", lambda: self.bc_show('layered_profile'), "#2C5282")

        # --- COMPUTE (single) ---
        self._sidebar_section(parent, "COMPUTE", "#2D5F3F")
        self._sidebar_btn(parent, "▶  Run Single Calculation", self.bc_run_single, "#2D5F3F")

        # --- UNCERTAINTY ---
        self._sidebar_section(parent, "UNCERTAINTY  (Triangular)", "#8B6F47")
        self._sidebar_btn(parent, "Ⅲ  Layered System Noise & Setting", lambda: self.bc_show('layer_noise'), "#8B6F47")
        #self._sidebar_btn(parent, "Ⅳ  Loading Noise", lambda: self.bc_show('loading_noise'), "#8B6F47")
        #self._sidebar_btn(parent, "Ⅴ  Deflections Noise", lambda: self.bc_show('deflection_noise'), "#8B6F47")
        self._sidebar_btn(parent, "Ⅳ  Deflections & Loading Noise", lambda: self.bc_show('defl_load_noise'), "#8B6F47")

        # --- COMPUTE (Monte Carlo) ---
        self._sidebar_section(parent, "COMPUTE", "#5D4E37")
        self._sidebar_btn(parent, "▶  Run Monte Carlo", self.bc_run_mc, "#5D4E37")

        # --- SAVE ---
        self._sidebar_section(parent, "SAVE", "#34495E")
        self._sidebar_btn(parent, "💾  Save Input/Output Data", self.bc_save_data, "#34495E")

    def _sidebar_section(self, parent, text, color):
        sep = tk.Frame(parent, bg=color, height=2)
        sep.pack(fill=tk.X, padx=12, pady=(12, 0))
        tk.Label(parent, text=text, font=("Segoe UI", 8, "bold"),
                 bg="#E8E4D8", fg=color, anchor="w").pack(fill=tk.X, padx=14, pady=(4, 3))

    def _sidebar_btn(self, parent, text, command, color):
        btn = tk.Label(parent, text=text,
                       font=("Segoe UI", 9, "bold"),
                       bg="#F5F2E8", fg="#1B2631",
                       relief=tk.FLAT, borderwidth=0, highlightthickness=0,
                       cursor="hand2", anchor="w", padx=14, pady=8)
        btn.pack(fill=tk.X, padx=10, pady=2)
        btn.bind('<Button-1>', lambda e: command())
        btn.bind('<Enter>', lambda e: btn.config(bg='#FFFFFF', fg=color))
        btn.bind('<Leave>', lambda e: btn.config(bg="#F5F2E8", fg="#1B2631"))
        return btn

    def _build_placeholder(self, frame, text):
        tk.Label(frame, text=text, fg="#7f8c8d", bg="#F5F2E8",
                 font=("Georgia", 13, "italic"), justify="center").place(relx=0.5, rely=0.5, anchor="center")

    def bc_show(self, name):
        self.forms.hide_all()
        self.bc_view_single.pack_forget()
        self.bc_view_mc.pack_forget()

        if name == 'single_result':
            self.bc_view_single.pack(fill=tk.BOTH, expand=True)
        elif name == 'mc_result':
            self.bc_view_mc.pack(fill=tk.BOTH, expand=True)
        else:
            self.forms.show(name)
            #if name == 'layered_profile':
                #self._draw_layered_profile(self.forms.layer_plot_frame)
        self.back_window.update_idletasks()

    # Physical range validation (shared by Forward / Backcalculation)
    def _validate_physical(self, forms=None, check_noise=False):
        forms = forms if forms is not None else self.forms
        errors = []
        lp = forms.vars['layered_profile']
        ln = forms.vars['layer_noise']

        for i in range(5):
            mvar = lp.get(f'modulus_{i}')
            raw = mvar.get().strip() if mvar else ''
            if raw:  
                mod = forms._f(mvar)
                if not (0 < mod < 1_000_000):
                    errors.append(f"Layer {i+1}: initial modulus {mod:g} MPa out of range (0 < E < 1,000,000).")
            pois = forms._f(lp.get(f'poisson_{i}'))
            if not (-1.0 <= pois <= 0.5):
                errors.append(f"Layer {i+1}: Poisson {pois:g} out of physical range (-1 to 0.5).")
            thk_var = lp.get(f'thickness_{i}')
            if thk_var is not None:
                thk = forms._f(thk_var)
                if i == 0:
                    if not (0 < thk <= 1_000_000):
                        errors.append(f"Layer 1: thickness {thk:g} mm must be > 0 and <= 1,000,000.")
                else:
                    if not (0 <= thk <= 1_000_000):
                        errors.append(f"Layer {i+1}: thickness {thk:g} mm must be 0 to 1,000,000.")

        if check_noise:
            for i in range(5):
                lo = forms._f(ln.get(f'mod_lower_{i}'))
                up = forms._f(ln.get(f'mod_upper_{i}'))
                if not (0 < lo < 1_000_000):
                    errors.append(f"Layer {i+1}: modulus lower bound {lo:g} out of range (0 to 1,000,000).")
                if not (0 < up < 1_000_000):
                    errors.append(f"Layer {i+1}: modulus upper bound {up:g} out of range (0 to 1,000,000).")
                if lo >= up:
                    errors.append(f"Layer {i+1}: modulus lower bound must be < upper bound.")
                thk_var = lp.get(f'thickness_{i}')
                thk_noise_var = ln.get(f'thk_noise_{i}')
                if thk_var is not None and thk_noise_var is not None:
                    thk = forms._f(thk_var)
                    tn = forms._f(thk_noise_var)
                    if tn < 0:
                        errors.append(f"Layer {i+1}: thickness noise must be >= 0.")
                    if i == 0 and thk - tn <= 0:
                        errors.append(f"Layer 1: thickness - noise = {thk-tn:g} <= 0 (must stay > 0).")
                    elif thk - tn < 0:
                        errors.append(f"Layer {i+1}: thickness - noise = {thk-tn:g} < 0 (negative thickness).")

            stress = forms._f(forms.vars['loading'].get('stress'))
            snoise = forms._f(forms.vars['loading_noise'].get('stress_noise'))
            if not (0 <= snoise < 1):
                errors.append(f"Stress noise {snoise:g} must be in [0, 1).")
            if stress * (1 - snoise) <= 0:
                errors.append(f"Stress noise too large: stress*(1-noise) = {stress*(1-snoise):g} <= 0.")

            for i in range(10):
                r = forms._f(forms.vars['deflections'].get(f'r_{i}'))
                rn = forms._f(forms.vars['deflection_noise'].get(f'r_noise_{i}'))
                d = forms._f(forms.vars['deflections'].get(f'defl_{i}'))
                dn = forms._f(forms.vars['deflection_noise'].get(f'd_noise_{i}'))
                if rn < 0:
                    errors.append(f"Point {i+1}: r noise must be >= 0.")
                if r - rn < 0:
                    errors.append(f"Point {i+1}: r - noise = {r-rn:g} < 0.")
                if dn < 0:
                    errors.append(f"Point {i+1}: deflection noise must be >= 0.")
                if d - dn < 0:
                    errors.append(f"Point {i+1}: deflection - noise = {d-dn:g} < 0.")
        return errors

    def _report_errors_dialog(self, errors, target_view=None):
        msg = "Please fix the following before computing:\n\n" + \
              "\n".join(f"• {e}" for e in errors)
        messagebox.showerror("Invalid Input", msg)
        self._write_log("[ERROR] Calculation aborted. Invalid input:\n" +
                        "\n".join("   - " + e for e in errors) + "\n")
        if target_view is not None:
            for w in target_view.winfo_children():
                w.destroy()
            tk.Label(target_view, text="⚠  Invalid input.\nSee dialog / Output Log for details.",
                     fg="#C0392B", bg="#F5F2E8", font=("Georgia", 13, "italic"),
                     justify="center").place(relx=0.5, rely=0.5, anchor="center")
            target_view.update_idletasks()

    def bc_save_data(self):
        folder = filedialog.askdirectory(title="Select folder to save data")
        if not folder:
            return
        try:
            arr_main = self.forms.build_arr_main()
            arr_noise = self.forms.build_arr_noise()
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            pd.DataFrame(arr_main).to_csv(os.path.join(folder, f"backcalc_input_main_{ts}.csv"), index=False)
            pd.DataFrame(arr_noise).to_csv(os.path.join(folder, f"backcalc_input_noise_{ts}.csv"), index=False)

            out_rows = [[f"Layer {i + 1}", self.forms.vars['layered_profile'].get(f'modulus_{i}').get() if self.forms.vars['layered_profile'].get(f'modulus_{i}') else ''] for i in range(5)]
            out_df = pd.DataFrame(out_rows, columns=['Layer', 'Modulus [MPa]'])
            out_df.to_csv(os.path.join(folder, f"backcalc_output_{ts}.csv"), index=False)

            with open(os.path.join(folder, f"backcalc_{ts}.log"), 'w', encoding='utf-8') as f:
                f.write("WuWan Back Calculation Data\nSaved: " + ts + "\n\n=== INPUT arr_main ===\n")
                f.write(pd.DataFrame(arr_main).to_string() + "\n\n=== INPUT arr_noise ===\n")
                f.write(pd.DataFrame(arr_noise).to_string() + "\n\n=== OUTPUT moduli ===\n")
                f.write(out_df.to_string() + "\n\n=== Output Log ===\n")
                f.write(self.bc_output_text.get(1.0, tk.END))

            messagebox.showinfo("Saved", f"Data saved to folder:\n{folder}")
            self._write_log(f"[✓] Input/Output data saved to: {folder}\n")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save data:\n{e}")
            self._write_log(f"[ERROR] Save failed: {e}\n")

    # ------- Single Calculation (Threaded) -------
    def bc_run_single(self):
        errors = self._validate_physical(self.forms, check_noise=False)
        if errors:
            self.bc_show('single_result')
            self._report_errors_dialog(errors, self.bc_view_single)
            return

        arr_main = self.forms.build_arr_main()
        arr_noise = self.forms.build_arr_noise(zero_errors=True)

        self.bc_show('single_result')
        for w in self.bc_view_single.winfo_children():
            w.destroy()

        tk.Label(self.bc_view_single, text="⏳  Running single back-calculation, please wait...",
                 fg="#5D6D7E", bg="#F5F2E8", font=("Georgia", 13, "italic")).place(relx=0.5, rely=0.4, anchor="center")
        pb = ttk.Progressbar(self.bc_view_single, mode='indeterminate')
        pb.place(relx=0.5, rely=0.55, width=300, anchor="center")
        pb.start(15)
        self.back_window.update()

        threading.Thread(target=self._thread_single_task, args=(arr_main, arr_noise), daemon=True).start()

    def _thread_single_task(self, arr_main, arr_noise):
        try:
            try:
                pd.DataFrame(arr_main).to_csv('back_calc_data.csv', index=False)
                pd.DataFrame(arr_noise).to_csv('noise_prior_data.csv', index=False)
            except Exception as e:
                self.back_window.after(0, lambda err=e: self._write_log(f"[WARN] Could not save prior data: {err}\n", clear=False))

            inverse_result = WuWan_pavement_inverse.BackCalculation(arr_main, arr_noise, verbose=2)
            modulus_values = inverse_result.final_moduli

            self.back_window.after(0, self._finish_single_task, modulus_values, arr_main)
        except Exception as e:
            self.back_window.after(0, lambda err=e: self._write_log(f"[ERROR] Single calculation failed: {err}\n"))

    def _finish_single_task(self, modulus_values, arr_main):
        for i, val in enumerate(modulus_values):
            v = self.forms.vars['layered_profile'].get(f'result_modulus_{i}')
            if v is not None:
                v.set(f"{float(val):.2f}")
        self._write_log("[✓] Back Calculation (Single) Finished!\n")
        self._render_single_result(modulus_values, arr_main)

    def _render_single_result(self, modulus_values, arr_main):
        frame = self.bc_view_single
        for w in frame.winfo_children():
            w.destroy()
        frame.configure(bg='#F5F2E8')

        card = tk.Frame(frame, bg='#FCFCFA', highlightbackground='#D6D3C7', highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        tk.Frame(card, bg='#2C5282', height=3).pack(side=tk.TOP, fill=tk.X)

        tk.Label(card, text="✓  Single Back-Calculation Result", font=("Georgia", 16, "bold"),
                 bg="#FCFCFA", fg="#2C5282").pack(anchor='w', padx=18, pady=(12, 2))
        tk.Label(card, text="Best-fit elastic moduli recovered from the deflection basin.(No uncertainty considered)",
                 font=("Segoe UI", 14, "italic"), bg="#FCFCFA", fg="#6B7280").pack(anchor='w', padx=18, pady=(0, 10))
        tk.Frame(card, bg='#D6D3C7', height=1).pack(fill=tk.X, padx=18)

        cols = ('Layer', 'Modulus [MPa]', 'Poisson [-]', 'Thickness [mm]')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Result.Treeview", rowheight=28, font=("Consolas", 14),
                        background='#FCFCFA', fieldbackground='#FCFCFA', foreground='#1B2631')
        style.configure("Result.Treeview.Heading", font=("Segoe UI", 14, "bold"), background='#4A5568', foreground='white')
        style.map("Result.Treeview", background=[('selected', '#EFEBDC')], foreground=[('selected', '#1B2631')])

        tree_frame = tk.Frame(card, bg='#FCFCFA')
        tree_frame.pack(padx=18, pady=10, fill=tk.X)
        tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=5, style="Result.Treeview")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=170, anchor='center')

        tree.tag_configure('odd', background='#FAF8F0')
        tree.tag_configure('even', background='#FCFCFA')

        for i in range(5):
            mod = modulus_values[i] if i < len(modulus_values) else 0.0
            poisson = self.forms._f(self.forms.vars['layered_profile'][f'poisson_{i}'])
            thk_var = self.forms.vars['layered_profile'].get(f'thickness_{i}')
            thk_text = 'semi-inf' if thk_var is None else f"{self.forms._f(thk_var):.1f}"
            tag = 'odd' if i % 2 == 0 else 'even'
            tree.insert('', tk.END, values=(f"Layer {i+1}", f"{float(mod):,.2f}", f"{poisson:.3f}", thk_text), tags=(tag,))
        tree.pack(fill=tk.X)

        tk.Frame(card, bg='#D6D3C7', height=1).pack(fill=tk.X, padx=18, pady=(10, 4))
        plot_frame = tk.Frame(card, bg='#FCFCFA')
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=4)
        
        self._draw_layered_profile(plot_frame, result_moduli=modulus_values)

    # ------- Monte Carlo calculation (Threaded) -------
    def bc_run_mc(self):
        errors = self._validate_physical(self.forms, check_noise=True)
        if errors:
            self.bc_show('mc_result')
            self._report_errors_dialog(errors, self.bc_view_mc)
            return

        arr_main = self.forms.build_arr_main()
        arr_noise = self.forms.build_arr_noise()

        self.bc_show('mc_result')
        for w in self.bc_view_mc.winfo_children():
            w.destroy()

        tk.Label(self.bc_view_mc, text="⏳  Running Monte Carlo back-calculation...\nThis may take a while.",
                 fg="#5D6D7E", bg="#F5F2E8", font=("Georgia", 13, "italic"), justify="center").place(relx=0.5, rely=0.4, anchor="center")
        pb = ttk.Progressbar(self.bc_view_mc, mode='indeterminate')
        pb.place(relx=0.5, rely=0.55, width=300, anchor="center")
        pb.start(15)
        self.back_window.update()

        threading.Thread(target=self._thread_mc_task, args=(arr_main, arr_noise), daemon=True).start()

    def _thread_mc_task(self, arr_main, arr_noise):
        try:
            try:
                pd.DataFrame(arr_main).to_csv('back_calc_mc_data.csv', index=False)
                pd.DataFrame(arr_noise).to_csv('noise_prior_mc_data.csv', index=False)
            except Exception as e:
                self.back_window.after(0, lambda err=e: self._write_log(f"[WARN] Could not save MC prior CSVs: {err}\n", clear=False))

            num_threads = multiprocessing.cpu_count()
            res = WuWan_pavement_montecarlo.ParalleMonteCarlo(arr_main, arr_noise, num_threads)

            means = np.mean(res, axis=0)
            medians = np.median(res, axis=0)
            ci_lower = np.percentile(res, 2.5, axis=0)
            ci_upper = np.percentile(res, 97.5, axis=0)
            q1 = np.percentile(res, 25, axis=0)
            q3 = np.percentile(res, 75, axis=0)

            z_95 = stats.norm.ppf(0.975)
            z_75 = stats.norm.ppf(0.75)
            uncertainty_val = (np.round(ci_upper, 2) - np.round(ci_lower, 2)) / np.round(q3 - q1, 2) - z_95 / z_75

            full_df = pd.DataFrame({
                'Layer':   [f'L{j+1}' for j in range(5)],
                'Thickness': arr_main[2:7, 3],
                'Mean':    np.round(means, 2),
                'Median':  np.round(medians, 2),
                'Q1(25%)': np.round(q1, 2),
                'Q3(75%)': np.round(q3, 2),
                'IQR':     np.round(q3 - q1, 2),
                'CI(2.5%)': np.round(ci_lower, 2),
                'CI(97.5%)': np.round(ci_upper, 2),
                'UR':      np.round(uncertainty_val, 4),
                'RR (%)':  np.round(q3 - q1, 2) / medians * 100,
            })
            full_df.at[full_df.index[-1], 'Thickness'] = np.inf

            self.back_window.after(0, self._finish_mc_task, res, means, medians, ci_lower, ci_upper, q1, q3, full_df)
        except Exception as e:
            self.back_window.after(0, lambda err=e: self._write_log(f"[ERROR] Monte Carlo calculation failed: {err}\n"))

    def _finish_mc_task(self, res, means, medians, ci_lower, ci_upper, q1, q3, full_df):
        for i, val in enumerate(medians):
            v = self.forms.vars['layered_profile'].get(f'result_modulus_{i}')
            if v is not None:
                v.set(f"{float(val):.2f}")

        summary_str = "\n" + "=" * 80 + "\n"
        summary_str += "           MONTE CARLO BACK CALCULATION RESULTS\n"
        summary_str += "=" * 80 + "\n"
        summary_str += "[UR] Tail Risk   : < 0.3 (Excellent) | 0.3-0.8 (Acceptable) | > 0.8 (Poor)\n"
        summary_str += "[RR] Core Spread : < 20% (Excellent) | 20-50% (Acceptable)  | > 50% (Poor)\n\n"

        g1 = full_df[['Layer', 'Thickness', 'Mean', 'Median']]
        summary_str += tabulate(g1, headers='keys', tablefmt='psql', showindex=False, numalign="right") + "\n\n"

        g2 = full_df[['Layer', 'Q1(25%)', 'Q3(75%)', 'IQR']]
        summary_str += tabulate(g2, headers='keys', tablefmt='psql', showindex=False, numalign="right") + "\n\n"

        g3 = full_df[['Layer', 'CI(2.5%)', 'CI(97.5%)', 'UR', 'RR (%)']]
        summary_str += tabulate(g3, headers='keys', tablefmt='psql', showindex=False, numalign="right") + "\n"
        summary_str += "=" * 80 + "\n"

        self._write_log("[✓] Monte Carlo Back Calculation Finished!\n")
        self.bc_output_text.insert(tk.END, summary_str)

        self._render_mc_result(res, means, medians, ci_lower, ci_upper, q1, q3, full_df)

    def _render_mc_result(self, res, means, medians, ci_lower, ci_upper, q1, q3, summary_df):
        frame = self.bc_view_mc
        for w in frame.winfo_children():
            w.destroy()
        frame.configure(bg='#F5F2E8')

        card = tk.Frame(frame, bg='#FCFCFA', highlightbackground='#D6D3C7', highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        tk.Frame(card, bg='#2D5F3F', height=3).pack(side=tk.TOP, fill=tk.X)

        title_row = tk.Frame(card, bg='#FCFCFA')
        title_row.pack(fill=tk.X, padx=18, pady=(10, 4))
        tk.Label(title_row, text="✓  Monte Carlo Back-Calculation Result", font=("Georgia", 16, "bold"),
                 bg="#FCFCFA", fg="#2D5F3F").pack(side=tk.LEFT)
        tk.Label(title_row, text=f"N = {res.shape[0]} samples · {res.shape[1]} layers",
                 font=("Segoe UI", 14, "italic"), bg="#FCFCFA", fg="#6B7280").pack(side=tk.RIGHT)

        tk.Frame(card, bg='#D6D3C7', height=1).pack(fill=tk.X, padx=18)

        violin_frame = tk.Frame(card, bg='#FCFCFA')
        violin_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)
        card.update_idletasks()
        self._draw_violin_plot(violin_frame, res, means, medians, ci_lower, ci_upper, q1, q3)

    def _draw_layered_profile(self, parent, result_moduli=None):
        for w in parent.winfo_children():
            w.destroy()

        moduli, poissons, thicknesses = [], [], []
        for i in range(5):
            if result_moduli is not None and i < len(result_moduli):
                moduli.append(result_moduli[i])
            else:
                moduli.append(self.forms._f(self.forms.vars['layered_profile'].get(f'modulus_{i}')))
            #moduli.append(self.forms._f(self.forms.vars['layered_profile'].get(f'modulus_{i}')))
            poissons.append(self.forms._f(self.forms.vars['layered_profile'].get(f'poisson_{i}')))
            tv = self.forms.vars['layered_profile'].get(f'thickness_{i}')
            thicknesses.append(None if tv is None else self.forms._f(tv))

        stress = self.forms._f(self.forms.vars['loading'].get('stress'))
        radius = self.forms._f(self.forms.vars['loading'].get('radius'))

        fig = Figure(figsize=(8.4, 2.6), dpi=90)
        fig.patch.set_facecolor('#FCFCFA')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#FCFCFA')

        finite = [t for t in thicknesses if t]
        semi_h = (max(finite) if finite else 300.0)
        raw = [t if t is not None else semi_h for t in thicknesses]

        total_raw = sum(raw)
        ratios = [h / total_raw for h in raw]
        MIN_RATIO, MAX_RATIO = 0.10, 0.40
        clipped = [max(MIN_RATIO, min(MAX_RATIO, r)) for r in ratios]
        clip_total = sum(clipped)
        normalized = [r / clip_total for r in clipped]
        total_disp = total_raw
        disp = [r * total_disp for r in normalized]

        width = 1200.0
        colors = ['#D8C9AC', '#C3B391', '#AC9B76', '#94835F', '#7B6A4A']

        y = 0.0
        for i in range(5):
            h = disp[i]
            ax.add_patch(mpatches.Rectangle((0, y - h), width, h,
                         facecolor=colors[i], edgecolor='#3A3A3A',
                         linewidth=1.1, zorder=2))
            thk_txt = 'semi-infinite' if thicknesses[i] is None else f'{thicknesses[i]:.0f} mm'
            ax.text(width * 0.97, y - h / 2, f'Layer {i+1}',
                    ha='right', va='center', fontsize=10, fontweight='bold',
                    family='serif', color='#1B2631', zorder=4)
            ax.text(width * 0.50, y - h / 2,
                    f'Moduli = {moduli[i]:,.0f} MPa     Poissons = {poissons[i]:.2f}     h = {thk_txt}',
                    ha='center', va='center', fontsize=9,
                    family='serif', color='#2C2C2C', zorder=4)
            y -= h
        bottom = y

        ax.plot([0, width], [0, 0], color='#1B2631', lw=1.8, zorder=3)

        cx = width * 0.28
        r_disp = min(radius, width * 0.18)
        arrow_h = (0 - bottom) * 0.12
        plate_left, plate_right = cx - r_disp, cx + r_disp

        ax.add_patch(mpatches.Rectangle((plate_left, 0), 2 * r_disp, arrow_h * 0.25,
                     facecolor='#2C5282', alpha=0.3, edgecolor='#2C5282',
                     lw=1.2, zorder=5))
        n_arrows = 5
        for k in range(n_arrows):
            x = plate_left + (2 * r_disp) * k / (n_arrows - 1)
            ax.annotate('', xy=(x, 0), xytext=(x, arrow_h),
                        arrowprops=dict(arrowstyle='->', color='#2C5282', lw=1.6),
                        zorder=6)
        ax.text(cx, arrow_h * 0.8, f'stress = {stress:.3f} MPa',
                ha='center', va='bottom', fontsize=10, color='#2C5282',
                fontweight='bold', family='serif')

        ax.annotate('', xy=(plate_right, arrow_h * 0.4), xytext=(cx, arrow_h * 0.4),
                    arrowprops=dict(arrowstyle='<->', color='#2D5F3F', lw=1.4), zorder=6)
        ax.text((cx + plate_right) / 2 * 1.5, arrow_h * 0.8, f'r = {radius:.0f} mm',
                ha='center', va='top', fontsize=9, color='#2D5F3F',
                fontweight='bold', family='serif')

        ax.set_xlim(-30, width + 30)
        ax.set_ylim(bottom * 1.05, arrow_h * 1.6 + 5)
        ax.set_xlabel('Horizontal distance  [mm]', fontsize=10, family='serif')
        ax.set_ylabel('Depth  [mm]', fontsize=10, family='serif')

        tick_positions = [0.0]
        tick_labels = ['0']
        y_tmp = 0.0
        real_depth = 0.0
        for i in range(5):
            y_tmp -= disp[i]
            tick_positions.append(y_tmp)
            if thicknesses[i] is None:
                tick_labels.append('∞')
            else:
                real_depth += thicknesses[i]
                tick_labels.append(f'{real_depth:.0f}')

        ax.set_yticks(tick_positions)
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_locator(mticker.FixedLocator(tick_positions))
        ax.set_yticklabels(tick_labels)

        fig.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.draw_idle()

    def _draw_violin_plot(self, parent, res, means, medians, ci_lower, ci_upper, q1, q3):
        for w in parent.winfo_children():
            w.destroy()
        num_layers = res.shape[1]
        layer_color = '#4A5F7E'

        def get_violin_width_at_y(violin_body, y_val):
            path = violin_body.get_paths()[0]
            verts = path.vertices
            ys, xs = verts[:, 1], verts[:, 0]
            x_at = []
            for k in range(len(verts) - 1):
                y0, y1 = ys[k], ys[k + 1]
                if (y0 <= y_val <= y1) or (y1 <= y_val <= y0):
                    if abs(y1 - y0) < 1e-12:
                        x_at.append(xs[k])
                    else:
                        t = (y_val - y0) / (y1 - y0)
                        x_at.append(xs[k] + t * (xs[k + 1] - xs[k]))
            if len(x_at) >= 2:
                return min(x_at), max(x_at)
            elif len(x_at) == 1:
                return x_at[0], x_at[0]
            return None, None

        def clip_violin(vb, y_low, y_high, fc, alpha, ax):
            ys = np.linspace(y_low, y_high, 10)
            lxs, rxs = [], []
            for y in ys:
                xl, xr = get_violin_width_at_y(vb, y)
                if xl is not None:
                    lxs.append(xl)
                    rxs.append(xr)
                else:
                    lxs.append(np.nan)
                    rxs.append(np.nan)
            lxs, rxs = np.array(lxs), np.array(rxs)
            m = ~(np.isnan(lxs) | np.isnan(rxs))
            yv = ys[m]
            if len(yv) < 2:
                return
            ax.fill(np.concatenate([lxs[m], rxs[m][::-1]]), np.concatenate([yv, yv[::-1]]),
                    facecolor=fc, alpha=alpha, edgecolor='none', zorder=3)

        fig = Figure(figsize=(8, 4.3), dpi=95)
        fig.patch.set_facecolor('#FCFCFA')
        fig.suptitle('Distribution of Back-Calculated Elastic Modulus', fontsize=12, fontweight='bold', y=0.97, family='serif')
        axes = [fig.add_subplot(1, num_layers, j + 1) for j in range(num_layers)]

        for j, ax in enumerate(axes):
            ax.set_facecolor('#FAF8F0')
            vp = ax.violinplot(res[:, j], positions=[1], showmeans=False, showmedians=False, showextrema=False, widths=0.75)
            vb = None
            for pc in vp['bodies']:
                pc.set_facecolor(layer_color)
                pc.set_edgecolor(layer_color)
                pc.set_alpha(0.4)
                pc.set_linewidth(1.0)
                pc.set_zorder(2)
                vb = pc
            if vb is not None:
                clip_violin(vb, q1[j], q3[j], layer_color, 0.3, ax)
            ax.vlines(1, ci_lower[j], ci_upper[j], color='#333', lw=2.0, zorder=5)
            ax.hlines([ci_lower[j], ci_upper[j]], 0.94, 1.06, color='#333', lw=1.5, zorder=5)
            for y_val, color in [(means[j], '#A3C98E'), (medians[j], '#AE3019')]:
                if vb is not None:
                    xl, xr = get_violin_width_at_y(vb, y_val)
                    if xl is not None:
                        m = (xr - xl) * 0.03
                        xl += m
                        xr -= m
                    else:
                        xl, xr = 0.7, 1.3
                else:
                    xl, xr = 0.7, 1.3
                ax.hlines(y_val, xl, xr, color=color, lw=2.0, zorder=6)
            yr = ci_upper[j] - ci_lower[j]
            if yr == 0:
                yr = 1
            ax.set_title(f'Layer {j+1}', fontsize=10, fontweight='bold', color=layer_color, pad=8, family='serif')
            ax.set_ylabel('Elastic Modulus (MPa)' if j == 0 else '', fontsize=8, family='serif')
            ax.set_xlim(0.3, 1.7)
            ax.set_xticks([])
            ax.grid(axis='y', linestyle='--', alpha=0.35)
            ax.tick_params(axis='y', labelsize=7)
            pad = yr * 0.15
            ax.set_ylim(ci_lower[j] - pad * 1.5, ci_upper[j] + pad * 1.5)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#333', lw=2.0, label='95% CI'),
            mpatches.Patch(facecolor='gray', alpha=0.35, label='IQR (25–75%)'),
            Line2D([0], [0], color='#A3C98E', lw=2.0, label='Mean'),
            Line2D([0], [0], color='#AE3019', lw=2.0, label='Median'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=8, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.01), edgecolor='lightgray', handlelength=2.5)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.draw_idle()

    def _write_log(self, text, clear=True):
        if clear:
            self.bc_output_text.delete(1.0, tk.END)
        self.bc_output_text.insert(tk.END, text)

    # ------- Forward profile helpers -------
    def toggle_profile_view(self):
        if not self.is_profile_open:
            self._ensure_deflection_computed()
            self._draw_profile_plot()
            self.forward_window.geometry('1250x550')
            self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
            self.profile_btn.config(text="<< Hide Profile Plot")
            self.is_profile_open = True
        else:
            self.plot_frame.pack_forget()
            self.forward_window.geometry('850x550')
            self.profile_btn.config(text="Show Profile Plot >>")
            self.is_profile_open = False

    def _ensure_deflection_computed(self):
        model_data = self.table_app.model.data
        has_data = False
        for r in range(1, 11):
            v = str(model_data.get(r, {}).get('Deflection [μm]', ''))
            if v not in ('', 'nan', 'None', '0', '0.0'):
                has_data = True
                break
        if not has_data:
            self.table_app.get_data()

    def _draw_profile_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        model_data = self.table_app.model.data
        r_values, defl_values = [], []
        for r in range(1, 11):
            row_data = model_data.get(r, {})
            try:
                r_val = float(str(row_data.get('r [mm]', '')))
                d_val = float(str(row_data.get('Deflection [μm]', '')))
                r_values.append(r_val)
                defl_values.append(d_val)
            except (ValueError, TypeError):
                continue
        if not r_values:
            tk.Label(self.plot_frame, text="No valid data to plot.", fg="#e74c3c", bg="white", font=("Arial", 13)).place(relx=0.5, rely=0.5, anchor="center")
            return

        fig = Figure(figsize=(5.2, 4.2), dpi=100)
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.scatter(r_values, [-d for d in defl_values], marker='o', color='#2C3E50', zorder=3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=-max(defl_values) * 1.15, top=0)
        ax.set_xlabel('r  [mm]', fontsize=11, fontfamily='serif', labelpad=8)
        ax.set_ylabel('Deflection  [μm]', fontsize=11, fontfamily='serif', labelpad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.tick_params(axis='both', which='major', direction='in', length=4, width=0.8, labelsize=9)
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--', linewidth=0.4, color='#BFBFBF', alpha=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{abs(x):.0f}'))
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.draw_idle()

    def return_to_main(self, window):
        window.destroy()
        self.root.deiconify()

    def coming_soon(self):
        messagebox.showinfo("Coming Soon", "  This feature is under development")

    # Sensor Location Optimization — ACADEMIC LAYOUT (mirrors Back Calculation)
    def open_sensor_location_optimization(self):
        self.root.withdraw()

        self.slo_window = tk.Toplevel(self.root)
        self.slo_window.geometry('960x800')
        self.slo_window.title('WuWan v0.3 - Sensor Location Optimization')
        self.slo_window.configure(bg='#F5F2E8')

        # --- Top toolbar ---
        top_bar = tk.Frame(self.slo_window, height=42, bg="#2C3E50")
        top_bar.pack(side=tk.TOP, fill=tk.X)
        top_bar.pack_propagate(False)

        FlatButton(top_bar, text="← Main Menu",
                   command=lambda: self.return_to_main(self.slo_window),
                   font=("Segoe UI", 9, "bold"),
                   bg="#E74C3C", fg="white",
                   activebg="#C0392B", activefg="white",
                   relief=tk.FLAT, padx=14, pady=4).pack(side=tk.LEFT, padx=10, pady=8)

        tk.Label(top_bar, text="◈  Sensor Location Optimization Module",
                 font=("Georgia", 12, "bold"),
                 bg="#2C3E50", fg="white").pack(side=tk.LEFT, padx=15)

        # --- Main container ---
        main_container = tk.Frame(self.slo_window, bg='#F5F2E8')
        main_container.pack(fill=tk.BOTH, expand=True)

        # --- Left sidebar ---
        sidebar = tk.Frame(main_container, width=240, bg="#E8E4D8",
                           highlightbackground="#BDB7A4", highlightthickness=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # --- Right area: whiteboard + output ---
        right_area = tk.Frame(main_container, bg='#F5F2E8')
        right_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bottom output
        output_container = tk.Frame(right_area, bg="#FCFCFA", height=170)
        output_container.pack(side=tk.BOTTOM, fill=tk.X)
        output_container.pack_propagate(False)

        out_header = tk.Frame(output_container, bg="#2C3E50", height=24)
        out_header.pack(side=tk.TOP, fill=tk.X)
        out_header.pack_propagate(False)
        tk.Label(out_header, text="  ▼ Output Log",
                 font=("Segoe UI", 9, "bold"),
                 bg="#2C3E50", fg="white", anchor="w"
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.slo_output_text = tk.Text(output_container, height=10,
                                       bg="#FCFCFA", fg="#1B2631",
                                       font=("Consolas", 9),
                                       relief=tk.FLAT, borderwidth=0)
        self.slo_output_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Top whiteboard
        self.slo_whiteboard = tk.Frame(right_area, bg='#F5F2E8')
        self.slo_whiteboard.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Input forms manager (own instance — same defaults as Back Calculation) ---
        self.slo_forms = InputFormsManager(self.slo_whiteboard, default_profile='slo')

        # --- Result view ---
        self.slo_view_result = tk.Frame(self.slo_whiteboard, bg='#F5F2E8')
        self._build_placeholder(self.slo_view_result,
                                "Click 'Run Sensor Location Optimization' to compute.\n"
                                "Results will appear here; charts are under PREVIEW.")

        # --- Preview views (populated on demand, only reachable after a finished run) ---
        self.slo_view_preview_conv = tk.Frame(self.slo_whiteboard, bg='#F5F2E8')
        self.slo_view_preview_dist = tk.Frame(self.slo_whiteboard, bg='#F5F2E8')
        self.slo_view_preview_layout = tk.Frame(self.slo_whiteboard, bg='#F5F2E8')

        self._last_slo_res = None

        self._build_slo_sidebar(sidebar)
        self.slo_show('deflections')

        self.slo_window.protocol(
            "WM_DELETE_WINDOW",
            lambda: self.return_to_main(self.slo_window))

    def _build_slo_sidebar(self, parent):
        tk.Label(parent, text="WuWan",
                 font=("Georgia", 32, "bold"),
                 bg="#E8E4D8", fg="#2C3E50", pady=8).pack(fill=tk.X)
        tk.Label(parent, text="Sensor Location Optimization",
                 font=("Georgia", 14, "italic"),
                 bg="#E8E4D8", fg="#5D6D7E", pady=2).pack(fill=tk.X)

        # --- INPUT DATA ---
        self._sidebar_section(parent, "INPUT DATA", "#2C5282")
        self._sidebar_btn(parent, "Ⅰ  Deflection Bowl & Loading System", lambda: self.slo_show('deflections'), "#2C5282")
        self._sidebar_btn(parent, "Ⅱ  Layered System", lambda: self.slo_show('layered_profile'), "#2C5282")

        # --- SEARCH SETTINGS ---
        self._sidebar_section(parent, "SEARCH SETTINGS", "#8B6F47")
        self._sidebar_btn(parent, "Ⅲ  Sensor Search Space & DE Settings", lambda: self.slo_show('slo_settings'), "#8B6F47")

        # --- MODULUS PRIOR ---
        self._sidebar_section(parent, "MODULUS PRIOR  (log-uniform)", "#5D4E37")
        self._sidebar_btn(parent, "Ⅳ  Layered System Noise & Setting", lambda: self.slo_show('layer_noise'), "#5D4E37")
        self._sidebar_btn(parent, "Ⅴ  Deflections & Loading Noise", lambda: self.slo_show('defl_load_noise'), "#5D4E37")

        # --- COMPUTE ---
        self._sidebar_section(parent, "COMPUTE", "#2D5F3F")
        self._sidebar_btn(parent, "▶  Run Sensor Location Optimization", self.slo_run, "#2D5F3F")

        # --- PREVIEW (only clickable after a finished run) ---
        self._sidebar_section(parent, "PREVIEW", "#6A4C93")
        is_ready = lambda: getattr(self, '_last_slo_res', None) is not None
        not_ready_msg = "Please run Sensor Location Optimization first."
        btn_conv = self._sidebar_btn_guarded(parent, "①  DE Convergence Curve",
                                             self.slo_preview_convergence, "#6A4C93", is_ready, not_ready_msg)
        btn_dist = self._sidebar_btn_guarded(parent, "②  Modulus Distribution Comparison",
                                             self.slo_preview_distribution, "#6A4C93", is_ready, not_ready_msg)
        btn_lay = self._sidebar_btn_guarded(parent, "③  Sensor Layout Comparison",
                                            self.slo_preview_layout, "#6A4C93", is_ready, not_ready_msg)
        self.slo_preview_btns = [btn_conv, btn_dist, btn_lay]
        self._refresh_slo_preview_buttons()

        # --- SAVE ---
        self._sidebar_section(parent, "SAVE", "#34495E")
        self._sidebar_btn(parent, "💾  Save Input/Output Data", self.slo_save_data, "#34495E")

    def _sidebar_btn_guarded(self, parent, text, command, color, ready_fn, not_ready_msg):
        """Sidebar button that only fires `command` once `ready_fn()` is True;
        otherwise pops up a dialog telling the user to compute first."""
        btn = tk.Label(parent, text=text,
                       font=("Segoe UI", 9, "bold"),
                       bg="#F5F2E8", fg="#A6A08D",
                       relief=tk.FLAT, borderwidth=0, highlightthickness=0,
                       cursor="arrow", anchor="w", padx=14, pady=8)
        btn.pack(fill=tk.X, padx=10, pady=2)

        def on_click(e):
            if ready_fn():
                command()
            else:
                messagebox.showinfo("Calculation Required", not_ready_msg)

        def on_enter(e):
            if ready_fn():
                btn.config(bg='#FFFFFF', fg=color)

        def on_leave(e):
            btn.config(bg="#F5F2E8", fg=("#1B2631" if ready_fn() else "#A6A08D"))

        btn.bind('<Button-1>', on_click)
        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        return btn

    def _refresh_slo_preview_buttons(self):
        ready = getattr(self, '_last_slo_res', None) is not None
        for btn in getattr(self, 'slo_preview_btns', []):
            if ready:
                btn.config(fg="#1B2631", cursor="hand2")
            else:
                btn.config(fg="#A6A08D", cursor="arrow")

    def slo_show(self, name):
        self.slo_forms.hide_all()
        for v in (self.slo_view_result, self.slo_view_preview_conv,
                  self.slo_view_preview_dist, self.slo_view_preview_layout):
            v.pack_forget()

        if name == 'slo_result':
            self.slo_view_result.pack(fill=tk.BOTH, expand=True)
        elif name == 'preview_convergence':
            self.slo_view_preview_conv.pack(fill=tk.BOTH, expand=True)
        elif name == 'preview_distribution':
            self.slo_view_preview_dist.pack(fill=tk.BOTH, expand=True)
        elif name == 'preview_layout':
            self.slo_view_preview_layout.pack(fill=tk.BOTH, expand=True)
        else:
            self.slo_forms.show(name)
        self.slo_window.update_idletasks()

    # ------- Validation -------
    def _validate_slo(self):
        errors = self._validate_physical(self.slo_forms, check_noise=True)

        f_opt = self.slo_forms._f_or_none
        tm = self.slo_forms.vars.get('true_modulus', {})
        for i in range(5):
            val = f_opt(tm.get(f'true_modulus_{i}'))
            if val is not None and not (0 < val < 1_000_000):
                errors.append(f"Layer {i+1}: True Modulus {val:g} MPa out of range (0 < E < 1,000,000).")

        f = self.slo_forms._f
        sp = self.slo_forms.vars['slo_settings']
        num_fixed = f(sp.get('num_fixed'))
        r_min = f(sp.get('r_min'))
        r_max = f(sp.get('r_max'))
        min_gap = f(sp.get('min_gap'))
        n_saa = f(sp.get('n_saa'))
        np_mult = f(sp.get('np_mult'))
        maxiter = f(sp.get('maxiter'))
        tol = f(sp.get('tol'))

        if not (1 <= num_fixed <= 9):
            errors.append(f"Number of Fixed Sensors must be between 1 and 9 (got {num_fixed:g}).")
        if r_min <= 0 or r_max <= r_min:
            errors.append(f"Search range invalid: r_min={r_min:g}, r_max={r_max:g} (need 0 < r_min < r_max).")
        if min_gap <= 0:
            errors.append(f"Min Sensor Gap must be > 0 (got {min_gap:g}).")
        if 1 <= num_fixed <= 9:
            k_free = 10 - int(num_fixed)
            if (r_max - r_min - (k_free - 1) * min_gap) <= 0:
                errors.append("Search range too small for the number of free sensors and minimum gap; "
                              "widen [r_min, r_max] or reduce Min Sensor Gap.")
        if n_saa < 4:
            errors.append(f"SAA Samples should be at least 4 (got {n_saa:g}).")
        if np_mult < 1:
            errors.append(f"DE Population Multiplier must be >= 1 (got {np_mult:g}).")
        if maxiter < 1:
            errors.append(f"DE Max Iterations must be >= 1 (got {maxiter:g}).")
        if tol <= 0:
            errors.append(f"DE Tolerance must be > 0 (got {tol:g}).")
        return errors

    def slo_save_data(self):
        folder = filedialog.askdirectory(title="Select folder to save data")
        if not folder:
            return
        try:
            arr_main = self.slo_forms.build_arr_main()
            arr_noise = self.slo_forms.build_arr_noise()
            params = self.slo_forms.build_slo_params()
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            pd.DataFrame(arr_main).to_csv(os.path.join(folder, f"slo_input_main_{ts}.csv"), index=False)
            pd.DataFrame(arr_noise).to_csv(os.path.join(folder, f"slo_input_noise_{ts}.csv"), index=False)

            res = getattr(self, '_last_slo_res', None)
            with open(os.path.join(folder, f"slo_{ts}.log"), 'w', encoding='utf-8') as f:
                f.write("WuWan Sensor Location Optimization Data\nSaved: " + ts + "\n\n=== SLO Settings ===\n")
                f.write("\n".join(f"{k} = {v}" for k, v in params.items()) + "\n\n")
                f.write("=== INPUT arr_main ===\n" + pd.DataFrame(arr_main).to_string() + "\n\n")
                f.write("=== INPUT arr_noise ===\n" + pd.DataFrame(arr_noise).to_string() + "\n\n")
                if res is not None:
                    f.write("=== RESULT ===\n")
                    f.write(f"initial positions : {np.round(res.initial_pos, 1)}\n")
                    f.write(f"optimized positions: {np.round(res.final_pos, 1)}\n")
                    f.write(f"D_eff_final = {res.d_eff_final:.4f}   volume_final = {res.vol_final:.1f}%\n")
                f.write("\n=== Output Log ===\n")
                f.write(self.slo_output_text.get(1.0, tk.END))

            messagebox.showinfo("Saved", f"Data saved to folder:\n{folder}")
            self._write_slo_log(f"[✓] Input/Output data saved to: {folder}\n")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save data:\n{e}")
            self._write_slo_log(f"[ERROR] Save failed: {e}\n")

    # ------- Run Sensor Location Optimization (Threaded) -------
    def slo_run(self):
        errors = self._validate_slo()
        if errors:
            self.slo_show('slo_result')
            self._report_errors_dialog(errors, self.slo_view_result)
            return

        arr_main = self.slo_forms.build_arr_main()
        arr_noise = self.slo_forms.build_arr_noise()
        params = self.slo_forms.build_slo_params()
        true_modulus_ref = self.slo_forms.build_true_moduli()  # optional known values, for the distribution preview only

        self.slo_show('slo_result')
        for w in self.slo_view_result.winfo_children():
            w.destroy()

        maxiter = max(int(params.get('maxiter', 1)), 1)
        self.slo_progress_label = tk.Label(
            self.slo_view_result,
            text=f"⏳  Running Differential Evolution...   iteration 0/{maxiter}  (0%)",
            fg="#5D6D7E", bg="#F5F2E8", font=("Georgia", 13, "italic"), justify="center")
        self.slo_progress_label.place(relx=0.5, rely=0.42, anchor="center")
        self.slo_progress_bar = ttk.Progressbar(self.slo_view_result, mode='determinate',
                                                maximum=100, value=0)
        self.slo_progress_bar.place(relx=0.5, rely=0.55, width=320, anchor="center")
        self.slo_window.update()

        threading.Thread(target=self._thread_slo_task,
                         args=(arr_main, arr_noise, params, true_modulus_ref), daemon=True).start()

    def _thread_slo_task(self, arr_main, arr_noise, params, true_modulus_ref):
        maxiter = max(int(params.get('maxiter', 1)), 1)

        def progress_cb(step, value, best, convergence):
            pct = min(100, int(round(step / maxiter * 100)))
            self.slo_window.after(0, self._update_slo_progress, step, maxiter, pct)

        try:
            res = WuWan_pavement_slo.optimize_sensor_layout(arr_main, arr_noise, callback=progress_cb, **params)
            res.true_modulus_ref = true_modulus_ref  # optional known values; None per-layer if unknown

            self.slo_window.after(0, self._update_slo_progress_text,
                                  "⏳  Running Monte Carlo comparison (initial layout)...")
            res.mc_initial = WuWan_pavement_slo.run_monte_carlo_at(
                arr_main, arr_noise, res.initial_pos, res.eval_moduli)

            self.slo_window.after(0, self._update_slo_progress_text,
                                  "⏳  Running Monte Carlo comparison (optimized layout)...")
            res.mc_final = WuWan_pavement_slo.run_monte_carlo_at(
                arr_main, arr_noise, res.final_pos, res.eval_moduli)

            self.slo_window.after(0, self._finish_slo_task, res)
        except Exception as e:
            self.slo_window.after(0, self._slo_task_failed, str(e))

    def _update_slo_progress(self, step, maxiter, pct):
        if getattr(self, 'slo_progress_bar', None) is not None and self.slo_progress_bar.winfo_exists():
            self.slo_progress_bar['value'] = pct
        if getattr(self, 'slo_progress_label', None) is not None and self.slo_progress_label.winfo_exists():
            self.slo_progress_label.config(
                text=f"⏳  Running Differential Evolution...   iteration {step}/{maxiter}  ({pct}%)")

    def _update_slo_progress_text(self, text):
        if getattr(self, 'slo_progress_bar', None) is not None and self.slo_progress_bar.winfo_exists():
            self.slo_progress_bar['value'] = 100
        if getattr(self, 'slo_progress_label', None) is not None and self.slo_progress_label.winfo_exists():
            self.slo_progress_label.config(text=text)

    def _slo_task_failed(self, msg):
        self._write_slo_log(f"[ERROR] Sensor location optimization failed: {msg}\n")
        for w in self.slo_view_result.winfo_children():
            w.destroy()
        tk.Label(self.slo_view_result, text=f"⚠  Optimization failed:\n{msg}",
                 fg="#C0392B", bg="#F5F2E8", font=("Georgia", 13, "italic"),
                 justify="center").place(relx=0.5, rely=0.5, anchor="center")

    def _finish_slo_task(self, res):
        self._last_slo_res = res
        self._refresh_slo_preview_buttons()

        summary = "\n" + "=" * 72 + "\n"
        summary += "          SENSOR LOCATION OPTIMIZATION RESULTS\n"
        summary += "=" * 72 + "\n"
        summary += f"DE iterations run                 : {res.n_iter}  ({res.de_message})\n"
        summary += f"D-efficiency (final vs initial)    : {res.d_eff_final:.4f}\n"
        summary += f"95% confidence-ellipsoid volume    : {res.vol_final:.1f}% of initial\n\n"
        summary += f"Robust SAA objective -E[ln det]      initial = {res.initial_saa:8.4f}   final = {res.final_saa:8.4f}\n"
        summary += f"log10 det FIM (true moduli)          initial = {res.init_logdet10:8.4f}   final = {res.final_logdet10:8.4f}\n"
        summary += f"condition number                     initial = {res.init_cond:12,.0f}   final = {res.final_cond:12,.0f}\n\n"
        summary += f"initial   positions : {np.round(res.initial_pos, 1)}\n"
        summary += f"optimized positions : {np.round(res.final_pos, 1)}\n"
        summary += "=" * 72 + "\n"
        summary += "See PREVIEW in the sidebar for the convergence curve and before/after comparison charts.\n"

        self._write_slo_log("[✓] Sensor Location Optimization Finished!\n")
        self.slo_output_text.insert(tk.END, summary)

        self._render_slo_result(res)

    def _render_slo_result(self, res):
        frame = self.slo_view_result
        for w in frame.winfo_children():
            w.destroy()
        frame.configure(bg='#F5F2E8')

        card = tk.Frame(frame, bg='#FCFCFA', highlightbackground='#D6D3C7', highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        tk.Frame(card, bg='#8B6F47', height=3).pack(side=tk.TOP, fill=tk.X)

        title_row = tk.Frame(card, bg='#FCFCFA')
        title_row.pack(fill=tk.X, padx=18, pady=(10, 4))
        tk.Label(title_row, text="✓  Sensor Location Optimization Result", font=("Georgia", 16, "bold"),
                 bg="#FCFCFA", fg="#8B6F47").pack(side=tk.LEFT)
        tk.Label(title_row, text=f"D_eff = {res.d_eff_final:.2f}  ·  CI volume → {res.vol_final:.0f}%",
                 font=("Segoe UI", 14, "italic"), bg="#FCFCFA", fg="#6B7280").pack(side=tk.RIGHT)
        tk.Frame(card, bg='#D6D3C7', height=1).pack(fill=tk.X, padx=18)

        # --- metrics summary table ---
        metrics_frame = tk.Frame(card, bg='#FCFCFA')
        metrics_frame.pack(fill=tk.X, padx=18, pady=(8, 4))

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("SLOMetric.Treeview", rowheight=26, font=("Consolas", 13),
                        background='#FCFCFA', fieldbackground='#FCFCFA', foreground='#1B2631')
        style.configure("SLOMetric.Treeview.Heading", font=("Segoe UI", 13, "bold"),
                        background='#4A5568', foreground='white')

        cols = ('Metric', 'Initial', 'Optimized')
        tree_m = ttk.Treeview(metrics_frame, columns=cols, show='headings', height=3, style="SLOMetric.Treeview")
        for c, w in zip(cols, (380, 140, 140)):
            tree_m.heading(c, text=c)
            tree_m.column(c, width=w, anchor='w' if c == 'Metric' else 'center')
        rows = [
            ('Robust SAA objective  -E[ln det]  (lower = better)', f"{res.initial_saa:.4f}", f"{res.final_saa:.4f}"),
            ('Single-point log10 det FIM (true moduli, higher = better)', f"{res.init_logdet10:.4f}", f"{res.final_logdet10:.4f}"),
            ('Condition number', f"{res.init_cond:,.0f}", f"{res.final_cond:,.0f}"),
        ]
        tree_m.tag_configure('odd', background='#FAF8F0')
        tree_m.tag_configure('even', background='#FCFCFA')
        for i, (m, iv, fv) in enumerate(rows):
            tree_m.insert('', tk.END, values=(m, iv, fv), tags=('odd' if i % 2 == 0 else 'even',))
        tree_m.pack(fill=tk.X)

        tk.Frame(card, bg='#D6D3C7', height=1).pack(fill=tk.X, padx=18, pady=(10, 4))

        # --- sensor positions table ---
        pos_frame = tk.Frame(card, bg='#FCFCFA')
        pos_frame.pack(fill=tk.X, padx=18, pady=(4, 4))
        tk.Label(pos_frame, text="Sensor Layout — Initial vs Optimized", font=("Georgia", 13, "bold"),
                 bg="#FCFCFA", fg="#8B6F47").pack(anchor='w', pady=(0, 4))

        style.configure("SLOPos.Treeview", rowheight=24, font=("Consolas", 12),
                        background='#FCFCFA', fieldbackground='#FCFCFA', foreground='#1B2631')
        style.configure("SLOPos.Treeview.Heading", font=("Segoe UI", 12, "bold"),
                        background='#4A5568', foreground='white')

        cols2 = ('Point', 'Status', 'Initial r [mm]', 'Optimized r [mm]', 'Δr [mm]')
        tree_p = ttk.Treeview(pos_frame, columns=cols2, show='headings', height=10, style="SLOPos.Treeview")
        for c, w in zip(cols2, (60, 80, 130, 150, 110)):
            tree_p.heading(c, text=c)
            tree_p.column(c, width=w, anchor='center')
        tree_p.tag_configure('odd', background='#FAF8F0')
        tree_p.tag_configure('even', background='#FCFCFA')
        for i in range(10):
            status = 'Fixed' if i < res.num_fixed else 'Free'
            ip, fp = res.initial_pos[i], res.final_pos[i]
            tree_p.insert('', tk.END, values=(f"P{i+1}", status, f"{ip:.1f}", f"{fp:.1f}", f"{fp - ip:+.1f}"),
                         tags=('odd' if i % 2 == 0 else 'even',))
        tree_p.pack(fill=tk.X)

        tk.Frame(card, bg='#D6D3C7', height=1).pack(fill=tk.X, padx=18, pady=(10, 4))
        hint = tk.Frame(card, bg='#EFEBDC', highlightthickness=1, highlightbackground='#C8C2B0')
        hint.pack(fill=tk.X, padx=18, pady=(4, 14))
        tk.Label(hint, text="ℹ  Open PREVIEW in the sidebar for the DE convergence curve (①), the "
                 "before/after modulus distribution comparison (②), and the sensor layout comparison (③).",
                 font=("Segoe UI", 11, "italic"), bg='#EFEBDC', fg='#5D4E37',
                 wraplength=640, justify='left').pack(fill=tk.X, padx=12, pady=10)

    # ------- Preview: button handlers -------
    def slo_preview_convergence(self):
        self.slo_show('preview_convergence')
        frame = self.slo_view_preview_conv
        for w in frame.winfo_children():
            w.destroy()
        frame.configure(bg='#F5F2E8')
        _card, body = self._simple_card(frame, "①  DE Convergence Curve", "#D4652A",
                                        subtitle="D-efficiency of the FIM vs. DE iteration")
        self._draw_slo_convergence(body, self._last_slo_res)

    def slo_preview_distribution(self):
        self.slo_show('preview_distribution')
        frame = self.slo_view_preview_dist
        for w in frame.winfo_children():
            w.destroy()
        frame.configure(bg='#F5F2E8')
        _card, body = self._simple_card(frame, "②  Modulus Distribution Comparison", "#4A5F7E",
                                        subtitle="Back-calculated modulus — initial vs optimized layout")
        self._draw_slo_distribution_comparison(body, self._last_slo_res)

    def slo_preview_layout(self):
        self.slo_show('preview_layout')
        frame = self.slo_view_preview_layout
        for w in frame.winfo_children():
            w.destroy()
        frame.configure(bg='#F5F2E8')
        _card, body = self._simple_card(frame, "③  Sensor Layout Comparison", "#2D5F3F",
                                        subtitle="Initial (top) vs optimized (bottom) sensor positions")
        self._draw_slo_layout_comparison(body, self._last_slo_res)

    def _simple_card(self, parent, title, accent, subtitle=None):
        card = tk.Frame(parent, bg='#FCFCFA', highlightbackground='#D6D3C7', highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        tk.Frame(card, bg=accent, height=3).pack(side=tk.TOP, fill=tk.X)

        title_row = tk.Frame(card, bg='#FCFCFA')
        title_row.pack(fill=tk.X, padx=18, pady=(10, 4))
        tk.Label(title_row, text=title, font=("Georgia", 16, "bold"),
                 bg="#FCFCFA", fg=accent).pack(side=tk.LEFT)
        if subtitle:
            tk.Label(title_row, text=subtitle, font=("Segoe UI", 13, "italic"),
                     bg="#FCFCFA", fg="#6B7280").pack(side=tk.RIGHT)
        tk.Frame(card, bg='#D6D3C7', height=1).pack(fill=tk.X, padx=18)

        body = tk.Frame(card, bg='#FCFCFA')
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)
        return card, body

    def _draw_slo_convergence(self, parent, res):
        for w in parent.winfo_children():
            w.destroy()

        fig = Figure(figsize=(7.6, 3.6), dpi=95)
        fig.patch.set_facecolor('#FCFCFA')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#FAF8F0')

        d_eff = res.d_eff
        steps = np.arange(len(d_eff))
        orange, grey, ink = '#D4652A', '#9AA0A6', '#2B2B2B'

        ax.fill_between(steps, 1.0, d_eff, color=orange, alpha=0.12, zorder=1)
        ax.plot(steps, d_eff, color=orange, lw=2.2, zorder=3, solid_capstyle='round')
        ax.axhline(1.0, color=grey, ls='--', lw=1.0, zorder=2)
        ax.text(steps[-1], 1.003, 'initial design', ha='right', va='bottom',
                fontsize=8, color=grey, style='italic')
        ax.scatter([steps[-1]], [res.d_eff_final], s=36, color=orange,
                  edgecolor='white', linewidth=1.0, zorder=5)
        ax.annotate(f'$D_{{eff}}$ = {res.d_eff_final:.2f}\nvolume → {res.vol_final:.0f}%',
                   xy=(steps[-1], res.d_eff_final), xytext=(-12, -30),
                   textcoords='offset points', ha='right', va='top', fontsize=9, color=ink,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=grey, alpha=0.9),
                   arrowprops=dict(arrowstyle='-', color=grey, lw=0.8))

        ax.set_xlim(-1, max(int(steps[-1]), 1) + 1)
        ax.set_ylim(min(1.0, float(d_eff.min())), max(res.d_eff_final * 1.05, 1.05))
        ax.set_xlabel('DE iteration', fontsize=10, family='serif', color=ink)
        ax.set_ylabel('D-efficiency  (vs. initial layout)', fontsize=10, family='serif', color=ink)
        ax.grid(axis='y', linestyle='--', alpha=0.35, color=grey)
        ax.grid(axis='x', linestyle='--', alpha=0.2, color=grey)
        ax.tick_params(labelsize=9, colors=ink)
        ax.spines['top'].set_visible(False)
        ax.set_title('Robust D-optimal sensor layout — DE convergence',
                    fontsize=12, fontweight='bold', color=ink, family='serif', pad=10)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.draw_idle()

    def _draw_slo_layout_comparison(self, parent, res):
        for w in parent.winfo_children():
            w.destroy()

        from matplotlib.lines import Line2D

        fig = Figure(figsize=(7.6, 3.2), dpi=95)
        fig.patch.set_facecolor('#FCFCFA')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#FAF8F0')

        color_before, color_after = '#4A5F7E', '#D4652A'
        ip, fp = res.initial_pos, res.final_pos

        ax.scatter(ip, np.ones_like(ip), color=color_before, s=60, zorder=5,
                  edgecolor='white', linewidth=1.0)
        ax.scatter(fp, -np.ones_like(fp), color=color_after, s=60, zorder=5,
                  edgecolor='white', linewidth=1.0)

        for k in range(len(ip)):
            ax.annotate(f'P{k+1}', xy=(ip[k], 1.0), xytext=(0, 6), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7.5, color=color_before, family='serif')
            ax.annotate(f'P{k+1}', xy=(fp[k], -1.0), xytext=(0, -6), textcoords='offset points',
                       ha='center', va='top', fontsize=7.5, color=color_after, family='serif')

        ax.axhline(0, color='#333333', lw=1.2, zorder=2)
        ax.set_ylim(-1.6, 1.6)
        x_max = max(float(ip.max()), float(fp.max())) * 1.08
        ax.set_xlim(-30, x_max)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Optimized', '', 'Initial'], fontsize=9.5, family='serif')
        ax.set_xlabel('r  [mm]', fontsize=10.5, family='serif')
        ax.grid(axis='x', linestyle='--', alpha=0.3, color='#9AA0A6')
        ax.tick_params(axis='x', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        legend_elements = [
            Line2D([0], [0], color=color_before, lw=2.2, marker='o', label='Initial layout'),
            Line2D([0], [0], color=color_after, lw=2.2, marker='o', label='Optimized layout'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8.5, frameon=True,
                 framealpha=0.9, edgecolor='lightgray')
        ax.set_title('Sensor Layout Comparison — Initial (top) vs Optimized (bottom)',
                    fontsize=12, fontweight='bold', family='serif', pad=10)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.draw_idle()

    def _draw_slo_distribution_comparison(self, parent, res):
        for w in parent.winfo_children():
            w.destroy()

        from matplotlib.lines import Line2D

        res1, res2 = res.mc_initial, res.mc_final
        true_values = getattr(res, 'true_modulus_ref', None) or [None] * res1.shape[1]
        num_layers = res1.shape[1]
        any_true_value = any(v is not None for v in true_values)

        means1, medians1 = np.mean(res1, axis=0), np.median(res1, axis=0)
        ci_lower1, ci_upper1 = np.percentile(res1, 2.5, axis=0), np.percentile(res1, 97.5, axis=0)
        q1_1, q3_1 = np.percentile(res1, 25, axis=0), np.percentile(res1, 75, axis=0)

        means2, medians2 = np.mean(res2, axis=0), np.median(res2, axis=0)
        ci_lower2, ci_upper2 = np.percentile(res2, 2.5, axis=0), np.percentile(res2, 97.5, axis=0)
        q1_2, q3_2 = np.percentile(res2, 25, axis=0), np.percentile(res2, 75, axis=0)

        color_res1, color_res2 = '#4A5F7E', '#D4652A'   # Initial / Optimized

        def get_violin_width_at_y(violin_body, y_val):
            verts = violin_body.get_paths()[0].vertices
            ys, xs = verts[:, 1], verts[:, 0]
            x_at_y = []
            for k in range(len(verts) - 1):
                y0, y1 = ys[k], ys[k + 1]
                if (y0 <= y_val <= y1) or (y1 <= y_val <= y0):
                    if abs(y1 - y0) < 1e-12:
                        x_at_y.append(xs[k])
                    else:
                        t = (y_val - y0) / (y1 - y0)
                        x_at_y.append(xs[k] + t * (xs[k + 1] - xs[k]))
            if len(x_at_y) >= 2:
                return min(x_at_y), max(x_at_y)
            elif len(x_at_y) == 1:
                return x_at_y[0], x_at_y[0]
            return None, None

        def clip_half_violin(violin_body, side, facecolor, edgecolor, alpha, ax,
                            center=1.0, y_low=None, y_high=None):
            verts = violin_body.get_paths()[0].vertices.copy()
            ys_all = verts[:, 1]
            y_min_v, y_max_v = ys_all.min(), ys_all.max()
            y_samples = np.linspace(y_min_v, y_max_v, 200)

            contour_left, contour_right = [], []
            for y_val in y_samples:
                xl, xr = get_violin_width_at_y(violin_body, y_val)
                if xl is not None:
                    contour_left.append((xl, y_val))
                    contour_right.append((xr, y_val))
                else:
                    contour_left.append((center, y_val))
                    contour_right.append((center, y_val))
            contour_left = np.array(contour_left)
            contour_right = np.array(contour_right)

            if side == 'left':
                half_x = np.concatenate([contour_left[:, 0], [center, center]])
                half_y = np.concatenate([contour_left[:, 1], [y_max_v, y_min_v]])
            else:
                half_x = np.concatenate([[center, center], contour_right[::-1, 0]])
                half_y = np.concatenate([[y_min_v, y_max_v], contour_right[::-1, 1]])
            ax.fill(half_x, half_y, facecolor=facecolor, alpha=alpha,
                    edgecolor=edgecolor, linewidth=1.0, zorder=2)

            if y_low is not None and y_high is not None:
                idx = np.where((y_samples >= y_low) & (y_samples <= y_high))[0]
                if len(idx) >= 2:
                    if side == 'left':
                        poly_x = np.concatenate([contour_left[idx, 0], [center, center]])
                        poly_y = np.concatenate([contour_left[idx, 1],
                                                [contour_left[idx[-1], 1], contour_left[idx[0], 1]]])
                    else:
                        poly_x = np.concatenate([[center, center], contour_right[idx[::-1], 0]])
                        poly_y = np.concatenate([[contour_right[idx[0], 1], contour_right[idx[-1], 1]],
                                                contour_right[idx[::-1], 1]])
                    ax.fill(poly_x, poly_y, facecolor=facecolor, alpha=alpha + 0.15,
                            edgecolor='none', zorder=3)
            return contour_left, contour_right

        def get_half_width_at_y(contour_arr, y_val, center, side):
            x_interp = np.interp(y_val, contour_arr[:, 1], contour_arr[:, 0])
            return (x_interp, center) if side == 'left' else (center, x_interp)

        def draw_safe_hline(ax, y_val, xl, xr, center, color, ls, lw, side):
            min_width = 0.03
            if xr - xl < min_width:
                if side == 'left':
                    xl, xr = center - min_width, center
                else:
                    xl, xr = center, center + min_width
            margin = (xr - xl) * 0.05
            ax.hlines(y_val, xl + margin, xr - margin, color=color, linestyle=ls, lw=lw, zorder=6)

        # --- Figure proportions ---------------------------------------------------
        fig_w = max(9.5, 2.35 * num_layers + 1.4)
        fig_h = fig_w / 1.5
        fig = Figure(figsize=(fig_w, fig_h), dpi=100)
        fig.patch.set_facecolor('#FCFCFA')
        fig.suptitle('Distribution of Back-Calculated Elastic Modulus  '
                    '(Initial vs Optimized Layout)',
                    fontsize=13, fontweight='bold', y=0.955, family='serif')
        axes = [fig.add_subplot(1, num_layers, j + 1) for j in range(num_layers)]

        fig.subplots_adjust(left=0.055, right=0.95, top=0.86, bottom=0.165, wspace=0.92)

        pos, vw, w_offset = 1.0, 1.0, 0.04   
        xlim = (0.40, 1.60)                  

        for j, ax in enumerate(axes):
            ax.set_facecolor('#FAFAFA')
            med_j = medians2[j] if medians2[j] != 0 else 1.0

            def pct(v, med_j=med_j):
                return (v - med_j) / med_j * 100.0

            pct_data1 = pct(res1[:, j])
            pct_ci_lower1, pct_ci_upper1 = pct(ci_lower1[j]), pct(ci_upper1[j])
            pct_q1_1, pct_q3_1 = pct(q1_1[j]), pct(q3_1[j])
            pct_mean1, pct_median1 = pct(means1[j]), pct(medians1[j])

            pct_data2 = pct(res2[:, j])
            pct_ci_lower2, pct_ci_upper2 = pct(ci_lower2[j]), pct(ci_upper2[j])
            pct_q1_2, pct_q3_2 = pct(q1_2[j]), pct(q3_2[j])
            pct_mean2, pct_median2 = pct(means2[j]), 0.0
            pct_true = pct(true_values[j]) if true_values[j] is not None else None

            all_ci_low, all_ci_high = min(pct_ci_lower1, pct_ci_lower2), max(pct_ci_upper1, pct_ci_upper2)
            all_q1, all_q3 = min(pct_q1_1, pct_q1_2), max(pct_q3_1, pct_q3_2)
            ci_span = all_ci_high - all_ci_low
            pct_iqr = all_q3 - all_q1
            use_inset = (ci_span > 3 * pct_iqr) if pct_iqr > 0 else False

            data_min_pct = min(np.min(pct_data1), np.min(pct_data2))
            data_max_pct = max(np.max(pct_data1), np.max(pct_data2))
            data_span = (data_max_pct - data_min_pct) if (data_max_pct - data_min_pct) > 0 else 1.0

            if use_inset:
                pad_iqr = pct_iqr * 0.8
                y_display_low, y_display_high = all_q1 - pad_iqr, all_q3 + pad_iqr
            else:
                pad_ci = ci_span * 0.1 if ci_span > 0 else 5.0
                y_display_low, y_display_high = all_ci_low - pad_ci, all_ci_high + pad_ci

            y_display_low = max(y_display_low, data_min_pct - data_span * 0.04)
            y_display_high = min(y_display_high, data_max_pct + data_span * 0.04)
            if y_display_high <= y_display_low:
                y_display_low, y_display_high = data_min_pct - 5, data_max_pct + 5

            vp1 = ax.violinplot(pct_data1, positions=[pos], showmeans=False,
                                showmedians=False, showextrema=False, widths=vw)
            violin_body1 = [pc.set_visible(False) or pc for pc in vp1['bodies']][0]
            vp2 = ax.violinplot(pct_data2, positions=[pos], showmeans=False,
                                showmedians=False, showextrema=False, widths=vw)
            violin_body2 = [pc.set_visible(False) or pc for pc in vp2['bodies']][0]

            contour_l1, _ = clip_half_violin(violin_body1, side='left', center=pos,
                                            facecolor=color_res1, edgecolor=color_res1,
                                            alpha=0.4, ax=ax, y_low=pct_q1_1, y_high=pct_q3_1)
            _, contour_r2 = clip_half_violin(violin_body2, side='right', center=pos,
                                            facecolor=color_res2, edgecolor=color_res2,
                                            alpha=0.4, ax=ax, y_low=pct_q1_2, y_high=pct_q3_2)

            ax.vlines(pos - w_offset, pct_ci_lower1, pct_ci_upper1, color=color_res1, lw=1.7, zorder=5)
            ax.hlines([pct_ci_lower1, pct_ci_upper1], pos - w_offset - 0.05, pos - w_offset + 0.05,
                    color=color_res1, lw=1.2, zorder=5)
            ax.vlines(pos + w_offset, pct_ci_lower2, pct_ci_upper2, color=color_res2, lw=1.7, zorder=5)
            ax.hlines([pct_ci_lower2, pct_ci_upper2], pos + w_offset - 0.05, pos + w_offset + 0.05,
                    color=color_res2, lw=1.2, zorder=5)

            markers_left = [(pct_mean1, '#A3C98E'), (pct_median1, '#AE3019')]
            markers_right = [(pct_mean2, '#A3C98E'), (pct_median2, '#AE3019')]
            if pct_true is not None:
                markers_left.append((pct_true, '#52AADC'))
                markers_right.append((pct_true, '#52AADC'))
            for y_val, color in markers_left:
                xl, xr = get_half_width_at_y(contour_l1, y_val, pos, 'left')
                draw_safe_hline(ax, y_val, xl, xr, pos, color, '-', 1.8, 'left')
            for y_val, color in markers_right:
                xl, xr = get_half_width_at_y(contour_r2, y_val, pos, 'right')
                draw_safe_hline(ax, y_val, xl, xr, pos, color, '--', 1.8, 'right')

            ax.axvline(pos, color='#AAAAAA', lw=0.5, ls=':', alpha=0.6, zorder=1)
            ax.set_xlim(*xlim)
            ax.set_xticks([])                       
            ax.grid(axis='y', linestyle='--', alpha=0.35, color='gray')
            ax.set_ylim(y_display_low, y_display_high)

            base_ticks = np.array([0.0, pct_median2, all_q1, all_q3, all_ci_low, all_ci_high])
            base_ticks = base_ticks[(base_ticks >= y_display_low) & (base_ticks <= y_display_high)]
            base_ticks = sorted(set(np.round(base_ticks, 1)))
            min_tick_dist = (y_display_high - y_display_low) * 0.08
            final_ticks = []
            for val in sorted(base_ticks, key=lambda x: (abs(x - 0.0), x)):
                if all(abs(val - ft) >= min_tick_dist for ft in final_ticks):
                    final_ticks.append(val)
            final_ticks = sorted(final_ticks) or [y_display_low, y_display_high]
            ax.set_yticks(final_ticks)
            ax.set_yticklabels(['0%' if abs(v) < 1e-6 else f'{v:+.1f}%' for v in final_ticks], fontsize=8)
            ax.tick_params(axis='y', labelsize=8, length=3)
            for sp in ax.spines.values():
                sp.set_edgecolor('#555555')
                sp.set_linewidth(0.9)
            if j == 0:
                ax.set_ylabel('Deviation from Median (%)', fontsize=10, family='serif')

            right_val = np.array(final_ticks) / 100.0 * med_j + med_j
            for k, (yt, rv) in enumerate(zip(final_ticks, right_val)):
                label = f'{rv:.0f} MPa' if k == len(final_ticks) - 1 else f'{rv:.0f}'
                ax.text(0.97, yt, label, transform=ax.get_yaxis_transform(),
                        ha='right', va='center', fontsize=6.3, color='#5D6D7E', family='serif',
                        bbox=dict(boxstyle='round,pad=0.08', fc='#FFFFFF', ec='none', alpha=0.7), zorder=8)

            ax.set_title(f'Layer {j+1}', fontsize=12, fontweight='bold', color='#333333', pad=6, family='serif')

            if use_inset:
                inset_ax = ax.inset_axes([0.56, 0.62, 0.40, 0.33])
                inset_ax.set_facecolor('#FAFAFA')
                full_pad = ci_span * 0.05 if ci_span > 0 else 1.0
                vp_in1 = inset_ax.violinplot(pct_data1, positions=[pos], showmeans=False,
                                            showmedians=False, showextrema=False, widths=vw * 0.8)
                inset_vb1 = [pc.set_visible(False) or pc for pc in vp_in1['bodies']][0]
                vp_in2 = inset_ax.violinplot(pct_data2, positions=[pos], showmeans=False,
                                            showmedians=False, showextrema=False, widths=vw * 0.8)
                inset_vb2 = [pc.set_visible(False) or pc for pc in vp_in2['bodies']][0]
                clip_half_violin(inset_vb1, side='left', center=pos, facecolor=color_res1,
                                edgecolor=color_res1, alpha=0.35, ax=inset_ax, y_low=pct_q1_1, y_high=pct_q3_1)
                clip_half_violin(inset_vb2, side='right', center=pos, facecolor=color_res2,
                                edgecolor=color_res2, alpha=0.35, ax=inset_ax, y_low=pct_q1_2, y_high=pct_q3_2)
                inset_ax.vlines(pos - w_offset, pct_ci_lower1, pct_ci_upper1, color=color_res1, lw=1.1, zorder=5)
                inset_ax.vlines(pos + w_offset, pct_ci_lower2, pct_ci_upper2, color=color_res2, lw=1.1, zorder=5)
                inset_ax.axhspan(y_display_low, y_display_high, color='#FFD700', alpha=0.15, zorder=1)
                inset_ax.axhline(y_display_low, color='#DAA520', ls='--', lw=0.7, alpha=0.6, zorder=4)
                inset_ax.axhline(y_display_high, color='#DAA520', ls='--', lw=0.7, alpha=0.6, zorder=4)
                inset_ax.set_ylim(all_ci_low - full_pad, all_ci_high + full_pad)
                inset_ax.set_xlim(0.4, 1.6)
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                for spine in inset_ax.spines.values():
                    spine.set_edgecolor('lightgray')
                    spine.set_linewidth(0.5)

        legend_elements = [
            mpatches.Patch(facecolor=color_res1, alpha=0.5, edgecolor=color_res1, label='Initial (left half)'),
            mpatches.Patch(facecolor=color_res2, alpha=0.5, edgecolor=color_res2, label='Optimized (right half)'),
            Line2D([0], [0], color='#333333', lw=1.7, label='95% CI'),
            mpatches.Patch(facecolor='gray', alpha=0.45, edgecolor='gray', label='IQR (25-75%)'),
            Line2D([0], [0], color='#A3C98E', lw=1.8, label='Mean'),
            Line2D([0], [0], color='#AE3019', lw=1.8, label='Median'),
        ]
        if any_true_value:
            legend_elements.append(Line2D([0], [0], color='#52AADC', lw=1.8, label='True Value'))
        legend_elements += [
            Line2D([0], [0], color='black', lw=1.4, ls='-', label='solid = Initial'),
            Line2D([0], [0], color='black', lw=1.4, ls='--', label='dashed = Optimized'),
            mpatches.Patch(facecolor='#FFD700', alpha=0.3, edgecolor='#DAA520', label='Inset range'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=8.5,
                frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, 0.022),
                edgecolor='lightgray', handlelength=2.0, columnspacing=1.6)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.draw_idle()

    def _write_slo_log(self, text, clear=True):
        if clear:
            self.slo_output_text.delete(1.0, tk.END)
        self.slo_output_text.insert(tk.END, text)


# Input Forms Manager
class InputFormsManager:
    COLOR_INPUT       = '#2C5282'
    COLOR_UNCERTAINTY = '#8B6F47'
    COLOR_TEXT        = '#1B2631'
    COLOR_MUTED       = '#6B7280'
    COLOR_BORDER      = '#D6D3C7'
    COLOR_CARD_BG     = '#FCFCFA'
    COLOR_PAGE_BG     = '#F5F2E8'
    COLOR_HEADER_BG   = '#EFEBDC'
    COLOR_ENTRY_BG    = '#FFFFFF'
    COLOR_ROW_ALT     = '#FAF8F0'
    COLOR_READONLY    = '#EFEBDC'

    def __init__(self, container, default_profile='backcalc'):
        self.container = container
        self.default_profile = default_profile  
        self.vars = {
            'layered_profile':  {}, 'loading':          {}, 'deflections':      {},
            'layer_noise':      {}, 'loading_noise':    {}, 'deflection_noise': {},
            'slo_settings':     {}, 'true_modulus':     {},
        }
        self.modules = {}
        self._build_layered_profile()
        self._build_deflections()       
        self._build_layer_noise()
        #self._build_loading_noise()
        #self._build_deflection_noise()
        self._build_defl_load_noise()
        self._build_slo_settings()       

    def hide_all(self):
        for f in self.modules.values():
            f.pack_forget()

    def show(self, name):
        self.hide_all()
        if name in self.modules:
            self.modules[name].pack(fill=tk.BOTH, expand=True)

    def _card(self, title, subtitle, accent, subtitle_font=("Segoe UI", 16, "italic")):
        outer = tk.Frame(self.container, bg=self.COLOR_PAGE_BG)
        inner = tk.Frame(outer, bg=self.COLOR_CARD_BG, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        tk.Frame(inner, bg=accent, height=3).pack(side=tk.TOP, fill=tk.X)
        title_frame = tk.Frame(inner, bg=self.COLOR_CARD_BG)
        title_frame.pack(side=tk.TOP, fill=tk.X, padx=18, pady=(10, 2))
        tk.Label(title_frame, text=title, font=("Georgia", 18, "bold"),
                 bg=self.COLOR_CARD_BG, fg=accent).pack(side=tk.LEFT, anchor='w')
        if subtitle:
            info_icon = tk.Label(title_frame, text=' ⓘ', font=("Segoe UI", 15, "bold"),
                                 bg=self.COLOR_CARD_BG, fg=accent, cursor='hand2')
            info_icon.pack(side=tk.LEFT, anchor='w', padx=(8, 0))
            ToolTip(info_icon, subtitle)
        tk.Frame(inner, bg=self.COLOR_BORDER, height=1).pack(side=tk.TOP, fill=tk.X, padx=18, pady=(6, 6))
        body = tk.Frame(inner, bg=self.COLOR_CARD_BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=18, pady=(0, 10))
        return outer, body

    def _entry(self, parent, var, width=14, readonly=False):
        e = tk.Entry(parent, textvariable=var, width=width, font=('Consolas', 16), relief='flat',
                     bg=self.COLOR_ENTRY_BG, fg=self.COLOR_TEXT, justify='center',
                     highlightthickness=1, highlightbackground='#C8C2B0', highlightcolor=self.COLOR_INPUT, insertbackground=self.COLOR_TEXT)
        if readonly:
            e.config(state='readonly', readonlybackground=self.COLOR_READONLY,
                     disabledforeground='#8A8473', fg='#5D5747')
        return e

    def _badge(self, parent, text, bg=None):
        if bg is None:
            bg = '#4A5568'
        return tk.Label(parent, text=text, font=("Segoe UI", 16, "bold"), bg=bg, fg='#F7F5EE', padx=10, pady=2)

    def _col_header(self, parent, text, col, span=1, row=0):
        cell = tk.Label(parent, text=text, font=("Segoe UI", 16, "bold"), bg=self.COLOR_HEADER_BG, fg=self.COLOR_TEXT, anchor='center', pady=5)
        cell.grid(row=row, column=col, columnspan=span, padx=1, pady=(0, 2), sticky='nsew')
        return cell

    def _disabled_box(self, parent, text='—'):
        return tk.Label(parent, text=text, font=('Consolas', 16, 'italic'), bg='#EFEBDC', fg='#8A8473', relief='flat', pady=3)

    def _build_layered_profile(self):
        subtitle = (
            "Here, the user inputs:\n"
            "initial layer moduli, Poisson ratios, and thicknesses.\n "
            "The initial moduli are utilized as an initial guess in the search for optimal moduli that best match calculated and measured deflections.\n "
            "To eliminate a layer (except the top), set its thickness to zero.\n"
            "The back-calculation is based on minimizing the sum of the squares of the individual relative errors."
        )
        if self.default_profile == 'slo':
            subtitle += (
                "\nTrue Modulus [MPa] (below the reference box) is OPTIONAL: the actual/known modulus "
                "used only to draw the 'True Value' marker in the distribution comparison preview. "
                "Leave blank if unknown — it is the Initial Modulus above that the optimizer searches around."
            )
        outer, body = self._card("Ⅱ  Layered System", subtitle, self.COLOR_INPUT)

        input_frame = tk.Frame(body, bg=self.COLOR_CARD_BG)
        input_frame.pack(side=tk.TOP, fill=tk.X)

        for c, h in enumerate(['Layer', 'Initial Modulus [MPa]', 'Poisson [-]', 'Thickness [mm]']):
            self._col_header(input_frame, h, c)
        poisson_def = ['0.30', '0.35', '0.35', '0.40', '0.40']
        modulus_def = ['5000', '1000', '600', '300', '100']
        if self.default_profile == 'slo':
            thickness_def = ['257', '241', '392', '478', None]
        else:
            thickness_def = ['50', '100', '200', '500', None]
        store = {}
        for i in range(5):
            r = i + 1
            self._badge(input_frame, f'L{i+1}').grid(row=r, column=0, padx=4, pady=2)
            vm = tk.StringVar(value=modulus_def[i])
            store[f'modulus_{i}'] = vm
            self._entry(input_frame, vm).grid(row=r, column=1, padx=4, pady=2, sticky='ew')
            vp = tk.StringVar(value=poisson_def[i])
            store[f'poisson_{i}'] = vp
            self._entry(input_frame, vp).grid(row=r, column=2, padx=4, pady=2, sticky='ew')
            if i == 4:
                self._disabled_box(input_frame, 'semi-infinite').grid(row=r, column=3, padx=4, pady=2, sticky='ew')
                store[f'thickness_{i}'] = None
            else:
                vt = tk.StringVar(value=thickness_def[i])
                store[f'thickness_{i}'] = vt
                self._entry(input_frame, vt).grid(row=r, column=3, padx=4, pady=2, sticky='ew')

        for c in range(4):
            input_frame.columnconfigure(c, weight=1)
        self.vars['layered_profile'] = store

        #tk.Frame(body, bg='#D6D3C7', height=1).pack(fill=tk.X, pady=(10, 4))
        #self.layer_plot_frame = tk.Frame(body, bg=self.COLOR_CARD_BG)
        #self.layer_plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        hint_frame = tk.Frame(body, bg='#EFEBDC', highlightthickness=1, highlightbackground='#C8C2B0')
        hint_frame.pack(fill=tk.X, pady=(20, 5))

        tk.Label(hint_frame, text="ℹ  Reference: Typical Initial Moduli for 5-Layer Flexible Pavements",
                 font=("Segoe UI", 9, "bold", "italic"), bg='#EFEBDC', fg='#5D4E37', anchor='w').pack(fill=tk.X, padx=12, pady=(8, 3))

        ref_text = ("• L1: Asphalt Surface Course      :  3,000  -  15,000 MPa\n"
                    "• L2: Asphalt Binder/Base         :  2,000  -  10,000 MPa\n"
                    "• L3: Unbound Granular Base       :    150  -     600 MPa\n"
                    "• L4: Granular Subbase            :     80  -     300 MPa\n"
                    "• L5: Subgrade (Semi-inf)         :     30  -     150 MPa")

        tk.Label(hint_frame, text=ref_text, font=("Consolas", 12), bg='#EFEBDC', fg='#5D4E37',
                 justify='left', anchor='w').pack(fill=tk.X, padx=28, pady=(0, 8))

        self.modules['layered_profile'] = outer

        if self.default_profile == 'slo':
            true_frame = tk.Frame(body, bg=self.COLOR_CARD_BG)
            true_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
            tk.Label(true_frame, text="True Modulus [MPa]  (optional — known value for comparison only)",
                     font=("Segoe UI", 11, "bold"), bg=self.COLOR_CARD_BG, fg=self.COLOR_MUTED,
                     anchor='w').pack(fill=tk.X, padx=2, pady=(0, 4))

            true_grid = tk.Frame(true_frame, bg=self.COLOR_CARD_BG)
            true_grid.pack(side=tk.TOP, fill=tk.X)
            true_def = ['17504', '2305', '574', '224', '101']
            true_store = {}
            for i in range(5):
                self._badge(true_grid, f'L{i+1}', bg=self.COLOR_MUTED).grid(row=0, column=i, padx=4, pady=(0, 2))
                vtm = tk.StringVar(value=true_def[i])
                true_store[f'true_modulus_{i}'] = vtm
                self._entry(true_grid, vtm).grid(row=1, column=i, padx=4, pady=2, sticky='ew')
            for c in range(5):
                true_grid.columnconfigure(c, weight=1)
            self.vars['true_modulus'] = true_store

    def _build_deflections(self):
        outer, body = self._card(
            "Ⅰ  Deflection Bowl & Loading System",
            "Here, the user inputs:\n"
            "a list of FWD sensor offsets from the load center,\n"
            "peak deflections, peak loading stress, and loading radius.\n",
            self.COLOR_INPUT
        )

        # --- Load System ---
        load_frame = tk.Frame(body, bg=self.COLOR_CARD_BG)
        load_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        tk.Label(load_frame, text="Load System", font=("Georgia", 13, "bold"),
                 bg=self.COLOR_CARD_BG, fg=self.COLOR_INPUT).grid(row=0, column=0, columnspan=2, sticky='w', padx=4, pady=(0, 4))
        lstore = {}
        for i, (lbl, k, dv) in enumerate([('Stress [MPa]', 'stress', '0.95'), ('Loading Radius [mm]', 'radius', '150')]):
            tk.Label(load_frame, text=lbl, font=("Segoe UI", 16, "bold"), bg=self.COLOR_CARD_BG, fg=self.COLOR_TEXT).grid(row=1, column=i, padx=18, pady=(2, 2), sticky='w')
            v = tk.StringVar(value=dv)
            lstore[k] = v
            self._entry(load_frame, v, width=18).grid(row=2, column=i, padx=18, pady=(0, 6), sticky='ew')
        for c in range(2):
            load_frame.columnconfigure(c, weight=1)
        self.vars['loading'] = lstore

        tk.Frame(body, bg=self.COLOR_BORDER, height=1).pack(fill=tk.X, pady=(2, 6))

        # --- Deflection table ---
        table = tk.Frame(body, bg=self.COLOR_CARD_BG)
        table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        for c, h in enumerate(['Point', 'r [mm]', 'Deflection [μm]']):
            self._col_header(table, h, c)
        r_def = ['0', '100', '200', '300', '450', '600', '900', '1200', '1500', '1800']
        if self.default_profile == 'slo':
            d_def = ['170.4', '167.6', '158.0', '152.0', '143.9', '136.4', '122.8', '110.7', '100.1', '90.7']
        else:
            d_def = ['744.3', '675.0', '516.3', '399.6', '303.1', '249.0', '186.1', '148.2', '122.0', '102.9']
        store = {}
        for i in range(10):
            r = i + 1
            self._badge(table, f'P{i+1}').grid(row=r, column=0, padx=4, pady=1)
            vr = tk.StringVar(value=r_def[i])
            store[f'r_{i}'] = vr
            self._entry(table, vr).grid(row=r, column=1, padx=4, pady=1, sticky='ew')
            vd = tk.StringVar(value=d_def[i])
            store[f'defl_{i}'] = vd
            self._entry(table, vd).grid(row=r, column=2, padx=4, pady=1, sticky='ew')
        for c in range(3):
            table.columnconfigure(c, weight=1)
        self.vars['deflections'] = store
        self.modules['deflections'] = outer

    def _build_layer_noise(self):
        outer, body = self._card(
            "Ⅲ  Layered System Noise & Setting  (Triangular Distribution)",
            "Modulus Lower/Upper [MPa]: bounds of the triangular prior for each layer's modulus.\n"
            "Modulus [MPa]: read-only, mirrors INPUT DATA (Layered System).\n"
            "Thickness [mm]: read-only, mirrors INPUT DATA (Layered System).\n"
            "Thickness Noise ± [mm]: symmetric measurement noise on layer thickness.",
            self.COLOR_UNCERTAINTY
        )
        lp = self.vars['layered_profile']

        # --- Modulus：Lower | Modulus | Upper ---
        for c, h in enumerate(['Layer', 'Modulus Lower [MPa]', 'Modulus [MPa]', 'Modulus Upper [MPa]']):
            self._col_header(body, h, c)
        mod_low_def = ['500', '50', '25', '10', '5']
        mod_up_def  = ['50000', '25000', '12500', '8000', '1000']
        store = {}
        for i in range(5):
            r = i + 1
            self._badge(body, f'L{i+1}', bg='#8B6F47').grid(row=r, column=0, padx=4, pady=2)
            vlo = tk.StringVar(value=mod_low_def[i])
            store[f'mod_lower_{i}'] = vlo
            self._entry(body, vlo).grid(row=r, column=1, padx=4, pady=2, sticky='ew')
            self._entry(body, lp.get(f'modulus_{i}'), readonly=True).grid(row=r, column=2, padx=4, pady=2, sticky='ew')
            vup = tk.StringVar(value=mod_up_def[i])
            store[f'mod_upper_{i}'] = vup
            self._entry(body, vup).grid(row=r, column=3, padx=4, pady=2, sticky='ew')

        # --- Thickness：Thickness | Thickness Noise ---
        tk.Frame(body, bg='#D6D3C7', height=1).grid(row=6, column=0, columnspan=4, sticky='ew', padx=4, pady=(12, 4))
        tk.Label(body, text="Thickness Setting & Noise", font=("Segoe UI", 12, "bold"),
                 bg=self.COLOR_CARD_BG, fg=self.COLOR_UNCERTAINTY, anchor='w').grid(
            row=7, column=0, columnspan=4, sticky='w', padx=4, pady=(0, 4))
        for c, h in enumerate(['Layer', 'Thickness [mm]', 'Thickness Noise ± [mm]']):
            self._col_header(body, h, c, row=8)
        thk_n_def = ['10', '10', '10', '10', None]
        for i in range(5):
            r = 9 + i
            self._badge(body, f'L{i+1}', bg='#8B6F47').grid(row=r, column=0, padx=4, pady=2)
            thk_var = lp.get(f'thickness_{i}')
            if i == 4:
                self._disabled_box(body, 'semi-infinite').grid(row=r, column=1, padx=4, pady=2, sticky='ew')
                self._disabled_box(body, '— N/A —').grid(row=r, column=2, padx=4, pady=2, sticky='ew')
                store[f'thk_noise_{i}'] = None
            else:
                # Thickness
                self._entry(body, thk_var, readonly=True).grid(row=r, column=1, padx=4, pady=2, sticky='ew')
                vt = tk.StringVar(value=thk_n_def[i])
                store[f'thk_noise_{i}'] = vt
                self._entry(body, vt).grid(row=r, column=2, padx=4, pady=2, sticky='ew')

        for c in range(4):
            body.columnconfigure(c, weight=1)
        self.vars['layer_noise'] = store
        self.modules['layer_noise'] = outer

    def _build_defl_load_noise(self):
        outer, body = self._card(
            "Ⅳ  Deflections & Loading Noise Setting  (Triangular)",
            "Set the uncertainty boundaries for the measured deflection basin and load system.\n"
            "Stress Noise [-]: relative coefficient of load uncertainty (e.g. 0.025 = ±2.5%).\n"
            "r noise ± [mm] / Defl noise ± [μm]: measurement uncertainty of FWD geophones.",
            self.COLOR_UNCERTAINTY
        )

        # 1. Loading Noise 
        load_frame = tk.Frame(body, bg=self.COLOR_CARD_BG)
        load_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        
        tk.Label(load_frame, text="Load Uncertainty", font=("Georgia", 13, "bold"),
                 bg=self.COLOR_CARD_BG, fg=self.COLOR_UNCERTAINTY).grid(row=0, column=0, columnspan=2, sticky='w', padx=4, pady=(0, 4))
                 
        tk.Label(load_frame, text='Stress [MPa]', font=("Segoe UI", 14, "bold"), bg=self.COLOR_CARD_BG, fg=self.COLOR_TEXT).grid(row=1, column=0, padx=18, pady=(2, 2), sticky='w')
        tk.Label(load_frame, text='Stress Noise Level [-]', font=("Segoe UI", 14, "bold"), bg=self.COLOR_CARD_BG, fg=self.COLOR_TEXT).grid(row=1, column=1, padx=18, pady=(2, 2), sticky='w')
        
        load_store = {}
        # Stress
        self._entry(load_frame, self.vars['loading'].get('stress'), width=18, readonly=True).grid(row=2, column=0, padx=18, pady=(0, 6), sticky='ew')
        # Stress noise 
        v_stress = tk.StringVar(value='0.025')
        load_store['stress_noise'] = v_stress
        self._entry(load_frame, v_stress, width=18).grid(row=2, column=1, padx=18, pady=(0, 6), sticky='ew')
        
        for c in range(2):
            load_frame.columnconfigure(c, weight=1)
        self.vars['loading_noise'] = load_store

        tk.Frame(body, bg=self.COLOR_BORDER, height=1).pack(fill=tk.X, pady=(4, 10))

        # 2. Deflections Noise
        defl_title_frame = tk.Frame(body, bg=self.COLOR_CARD_BG)
        defl_title_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(defl_title_frame, text="Deflection Basin Uncertainty", font=("Georgia", 13, "bold"),
                 bg=self.COLOR_CARD_BG, fg=self.COLOR_UNCERTAINTY).pack(side=tk.LEFT, padx=4, pady=(0, 4))
        

        glob = tk.Frame(body, bg='#EFEBDC', highlightthickness=1, highlightbackground='#C8C2B0')
        glob.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        tk.Label(glob, text="Global error:", bg='#EFEBDC', fg='#5D4E37', font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=8, pady=6)
        tk.Label(glob, text="r ± [mm]:", bg='#EFEBDC', fg='#5D4E37', font=("Segoe UI", 14)).pack(side=tk.LEFT)
        self.global_r_noise = tk.StringVar(value='2')
        self._entry(glob, self.global_r_noise, width=8).pack(side=tk.LEFT, padx=4)
        tk.Label(glob, text="Defl ± [μm]:", bg='#EFEBDC', fg='#5D4E37', font=("Segoe UI", 14)).pack(side=tk.LEFT)
        self.global_d_noise = tk.StringVar(value='1')
        self._entry(glob, self.global_d_noise, width=8).pack(side=tk.LEFT, padx=4)
        FlatButton(glob, text="Apply to All", command=self._apply_global_defl_noise,
                   font=("Segoe UI", 12, "bold"), bg='#8B6F47', fg='white',
                   activebg='#6F5736', activefg='white',
                   relief=tk.FLAT, padx=10, pady=2).pack(side=tk.LEFT, padx=8)


        dfl = self.vars['deflections']
        table = tk.Frame(body, bg=self.COLOR_CARD_BG)
        table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        for c, h in enumerate(['Point', 'r [mm]', 'r noise ± [mm]', 'Deflection [μm]', 'Defl noise ± [μm]']):
            self._col_header(table, h, c)
            
        defl_store = {}
        for i in range(10):
            r = i + 1
            self._badge(table, f'P{i+1}', bg='#8B6F47').grid(row=r, column=0, padx=3, pady=1)
            # r 
            # r noise 
            if i == 0:
                vr = tk.StringVar(value='0')
                defl_store[f'r_noise_{i}'] = vr
                self._entry(table, vr, width=9, readonly=True).grid(row=r, column=2, padx=3, pady=1, sticky='ew')
            else:
                vr = tk.StringVar(value='10')
                defl_store[f'r_noise_{i}'] = vr
                self._entry(table, vr, width=9).grid(row=r, column=2, padx=3, pady=1, sticky='ew')
            self._entry(table, dfl.get(f'r_{i}'), width=9, readonly=True).grid(row=r, column=1, padx=3, pady=1, sticky='ew')
            
            self._entry(table, dfl.get(f'defl_{i}'), width=9, readonly=True).grid(row=r, column=3, padx=3, pady=1, sticky='ew')
            vd = tk.StringVar(value='1')
            defl_store[f'd_noise_{i}'] = vd
            self._entry(table, vd, width=9).grid(row=r, column=4, padx=3, pady=1, sticky='ew')
            
        for c in range(5):
            table.columnconfigure(c, weight=1)
        self.vars['deflection_noise'] = defl_store
        
        self.modules['defl_load_noise'] = outer

    def _build_slo_settings(self):
        outer, body = self._card(
            "Ⅲ  Sensor Search Space & DE Settings",
            "Number of Fixed Sensors: how many of the FIRST evaluation points (by Point order in "
            "Deflection Bowl & Loading System) are held fixed near the load; the remaining sensors are "
            "free and re-optimized.\n"
            "Search Range r_min / r_max [mm]: lower/upper bound for the free sensor positions.\n"
            "Min Sensor Gap [mm]: minimum spacing enforced between adjacent free sensors.\n"
            "SAA Samples: number of log-uniform modulus draws (taken from the Modulus Prior "
            "lower/upper bounds in Layered System Noise) used to approximate the robust design "
            "objective E[ln det FIM].\n"
            "DE Population Multiplier / Max Iterations / Tolerance / Seed: Differential Evolution "
            "settings used to search the free sensor positions.",
            self.COLOR_UNCERTAINTY
        )

        grid = tk.Frame(body, bg=self.COLOR_CARD_BG)
        grid.pack(side=tk.TOP, fill=tk.X)

        rows = [
            [('Number of Fixed Sensors [-]',   'num_fixed', '3'),
             ('Min Sensor Gap [mm]',            'min_gap',   '100')],
            [('Search Range Lower r_min [mm]', 'r_min',     '300'),
             ('Search Range Upper r_max [mm]', 'r_max',     '3000')],
            [('SAA Samples [-]',               'n_saa',     '32'),
             ('DE Population Multiplier [-]',  'np_mult',   '10')],
            [('DE Max Iterations [-]',         'maxiter',   '200'),
             ('DE Tolerance [-]',              'tol',       '1e-5'),
             ('Random Seed [-]',               'seed',      '42')],
        ]
        n_cols = 3
        store = {}
        for r, row_fields in enumerate(rows):
            for c, (lbl, key, dv) in enumerate(row_fields):
                tk.Label(grid, text=lbl, font=("Segoe UI", 13, "bold"), bg=self.COLOR_CARD_BG,
                         fg=self.COLOR_TEXT).grid(row=2 * r, column=c, padx=14, pady=(8, 2), sticky='w')
                v = tk.StringVar(value=dv)
                store[key] = v
                self._entry(grid, v, width=16).grid(row=2 * r + 1, column=c, padx=14, pady=(0, 8), sticky='ew')
        for c in range(n_cols):
            grid.columnconfigure(c, weight=1)

        hint = tk.Frame(body, bg='#EFEBDC', highlightthickness=1, highlightbackground='#C8C2B0')
        hint.pack(fill=tk.X, pady=(16, 5))
        tk.Label(hint, text="ℹ  Fixed sensors are taken from the FIRST N rows of the Deflections table "
                 "(by Point order); the remaining (10 − N) sensors are optimized within [r_min, r_max].",
                 font=("Segoe UI", 11, "italic"), bg='#EFEBDC', fg='#5D4E37',
                 wraplength=560, justify='left').pack(fill=tk.X, padx=12, pady=8)

        self.vars['slo_settings'] = store
        self.modules['slo_settings'] = outer

    def build_slo_params(self):
        s = self.vars['slo_settings']
        return dict(
            num_fixed=int(self._f(s.get('num_fixed'))),
            r_min=self._f(s.get('r_min')),
            r_max=self._f(s.get('r_max')),
            min_gap=self._f(s.get('min_gap')),
            n_saa=int(self._f(s.get('n_saa'))),
            np_mult=int(self._f(s.get('np_mult'))),
            maxiter=int(self._f(s.get('maxiter'))),
            tol=self._f(s.get('tol')),
            seed=int(self._f(s.get('seed'))),
        )

    def _build_loading_noise(self):
        outer, body = self._card(
            "Ⅳ  Loading Noise Setting (Triangular Distribution)",
            "Stress [MPa]: read-only, mirrors INPUT DATA (Load System).\n"
            "Stress Noise [-]: relative coefficient of load uncertainty (e.g. 0.025 = ±2.5%).",
            self.COLOR_UNCERTAINTY
        )
        store = {}
        # Stress
        tk.Label(body, text='Stress [MPa]', font=("Segoe UI", 16, "bold"),
                 bg=self.COLOR_CARD_BG, fg=self.COLOR_TEXT).grid(row=0, column=0, padx=18, pady=(6, 2), sticky='w')
        self._entry(body, self.vars['loading'].get('stress'), width=18, readonly=True).grid(
            row=1, column=0, padx=18, pady=(0, 10), sticky='ew')
        # Stress noise
        tk.Label(body, text='Stress Noise Level [-]', font=("Segoe UI", 16, "bold"),
                 bg=self.COLOR_CARD_BG, fg=self.COLOR_TEXT).grid(row=0, column=1, padx=18, pady=(6, 2), sticky='w')
        v = tk.StringVar(value='0.025')
        store['stress_noise'] = v
        self._entry(body, v, width=18).grid(row=1, column=1, padx=18, pady=(0, 10), sticky='ew')

        info = tk.Frame(body, bg='#EFEBDC', highlightthickness=1, highlightbackground='#C8C2B0')
        info.grid(row=2, column=0, columnspan=2, padx=18, pady=10, sticky='ew')
        tk.Label(info, text='ℹ  Stress noise is a relative coefficient. Stress mirrors INPUT DATA.', font=("Segoe UI", 16, "italic"), bg='#EFEBDC', fg='#5D4E37', padx=12, pady=8).pack(anchor='w')
        for c in range(2):
            body.columnconfigure(c, weight=1)
        self.vars['loading_noise'] = store
        self.modules['loading_noise'] = outer

    def _build_deflection_noise(self):
        outer, body = self._card(
            "Ⅴ  Deflections Noise Setting  (Triangular Distribution)",
            "r [mm] / Deflection [μm]: read-only, mirror INPUT DATA.\n"
            "r noise ± [mm]: uncertainty in geophone position from load center.\n"
            "Deflection noise ± [μm]: measurement uncertainty of each geophone reading.",
            self.COLOR_UNCERTAINTY
        )
        glob = tk.Frame(body, bg='#EFEBDC', highlightthickness=1, highlightbackground='#C8C2B0')
        glob.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        tk.Label(glob, text="Global error (applies to all 10 points):", bg='#EFEBDC', fg='#5D4E37', font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT, padx=8, pady=6)
        tk.Label(glob, text="r ± [mm]:", bg='#EFEBDC', fg='#5D4E37', font=("Segoe UI", 16)).pack(side=tk.LEFT)
        self.global_r_noise = tk.StringVar(value='2')
        self._entry(glob, self.global_r_noise, width=8).pack(side=tk.LEFT, padx=4)
        tk.Label(glob, text="Defl ± [μm]:", bg='#EFEBDC', fg='#5D4E37', font=("Segoe UI", 16)).pack(side=tk.LEFT)
        self.global_d_noise = tk.StringVar(value='1')
        self._entry(glob, self.global_d_noise, width=8).pack(side=tk.LEFT, padx=4)
        FlatButton(glob, text="Apply to All", command=self._apply_global_defl_noise,
                   font=("Segoe UI", 16, "bold"), bg='#8B6F47', fg='white',
                   activebg='#6F5736', activefg='white',
                   relief=tk.FLAT, padx=10, pady=2).pack(side=tk.LEFT, padx=8)

        dfl = self.vars['deflections']
        table = tk.Frame(body, bg=self.COLOR_CARD_BG)
        table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        for c, h in enumerate(['Point', 'r [mm]', 'r noise ± [mm]', 'Deflection [μm]', 'Defl noise ± [μm]']):
            self._col_header(table, h, c)
        store = {}
        for i in range(10):
            r = i + 1
            self._badge(table, f'P{i+1}', bg='#8B6F47').grid(row=r, column=0, padx=3, pady=1)
            # r
            self._entry(table, dfl.get(f'r_{i}'), width=9, readonly=True).grid(row=r, column=1, padx=3, pady=1, sticky='ew')
            vr = tk.StringVar(value='10')
            store[f'r_noise_{i}'] = vr
            self._entry(table, vr, width=9).grid(row=r, column=2, padx=3, pady=1, sticky='ew')
            # Deflection
            self._entry(table, dfl.get(f'defl_{i}'), width=9, readonly=True).grid(row=r, column=3, padx=3, pady=1, sticky='ew')
            vd = tk.StringVar(value='1')
            store[f'd_noise_{i}'] = vd
            self._entry(table, vd, width=9).grid(row=r, column=4, padx=3, pady=1, sticky='ew')
        for c in range(5):
            table.columnconfigure(c, weight=1)
        self.vars['deflection_noise'] = store
        self.modules['deflection_noise'] = outer

    def _apply_global_defl_noise(self):
        rv, dv = self.global_r_noise.get(), self.global_d_noise.get()
        for i in range(10):
            if i != 0:
                self.vars['deflection_noise'][f'r_noise_{i}'].set(rv)
                
            self.vars['deflection_noise'][f'd_noise_{i}'].set(dv)

    @staticmethod
    def _f(var, default=0.0):
        if var is None:
            return 0.0
        try:
            s = var.get().strip()
            return float(s) if s else 0.0
        except (ValueError, AttributeError):
            return default

    @staticmethod
    def _f_or_none(var):
        """Like _f, but returns None for blank/invalid instead of 0.0 — used where
        'blank' has a distinct meaning (e.g. True Modulus: unknown, don't compare)."""
        if var is None:
            return None
        try:
            s = var.get().strip()
            return float(s) if s else None
        except (ValueError, AttributeError):
            return None

    def build_true_moduli(self):
        """Optional known modulus per layer (SLO 'True Modulus' row); None where left blank."""
        store = self.vars.get('true_modulus', {})
        return [self._f_or_none(store.get(f'true_modulus_{i}')) for i in range(5)]

    def build_arr_main(self):
        arr = np.zeros((11, 8), dtype=np.float64)
        for i in range(5):
            arr[i + 2, 0] = i + 1
            arr[i + 2, 1] = self._f(self.vars['layered_profile'].get(f'modulus_{i}'))
            arr[i + 2, 2] = self._f(self.vars['layered_profile'].get(f'poisson_{i}'))
            arr[i + 2, 3] = self._f(self.vars['layered_profile'].get(f'thickness_{i}'))
        arr[10, 1] = self._f(self.vars['loading'].get('stress'))
        arr[10, 3] = self._f(self.vars['loading'].get('radius'))
        for i in range(10):
            arr[i + 1, 5] = i + 1
            arr[i + 1, 6] = self._f(self.vars['deflections'].get(f'r_{i}'))
            arr[i + 1, 7] = self._f(self.vars['deflections'].get(f'defl_{i}'))
        return np.ascontiguousarray(arr)

    def build_arr_noise(self, zero_errors=False):
        arr = np.zeros((11, 7), dtype=np.float64)
        for i in range(5):
            arr[i + 2, 0] = i + 1
            arr[i + 2, 1] = self._f(self.vars['layer_noise'].get(f'mod_lower_{i}'))
            arr[i + 2, 2] = self._f(self.vars['layer_noise'].get(f'mod_upper_{i}'))
            arr[i + 2, 3] = 0.0 if zero_errors else self._f(self.vars['layer_noise'].get(f'thk_noise_{i}'))
        arr[10, 1] = 0.0 if zero_errors else self._f(self.vars['loading_noise'].get('stress_noise'))
        arr[10, 2] = 0.0
        for i in range(10):
            arr[i + 1, 4] = i + 1
            if zero_errors:
                arr[i + 1, 5] = arr[i + 1, 6] = 0.0
            else:
                arr[i + 1, 5] = self._f(self.vars['deflection_noise'].get(f'r_noise_{i}'))
                arr[i + 1, 6] = self._f(self.vars['deflection_noise'].get(f'd_noise_{i}'))
        return np.ascontiguousarray(arr)


# Forward TableApp
class TableApp:
    def __init__(self, root):
        self.root = root
        self.frame = tk.Frame(root, bg='#FCFCFA')
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.model = TableModel()
        self.table = TableCanvas(self.frame, model=self.model, rowheight=30,
                                 grid_color="black", linewidth=2.5,
                                 rowselectedcolor="light blue",
                                 cellbackgr="#CCCCCC")
        self.table.show()
        self.table.tablecolheader.grid_forget()
        self.table.tablerowheader.grid_forget()

        original_drawText = self.table.drawText

        def custom_drawText(row, col, celltxt, fgcolor=None, align=None):
            col_name = self.model.columnNames[col]

            if col_name in ['r [mm]', 'Deflection [μm]']:
                align = 'center'
                
            elif col_name in ['Layer', 'Modulus [MPa]', 'Poisson [-]', 'Thickness [mm]']:
                
                if row == 10:
                    align = 'e'  
                
                elif col_name in ['Modulus [MPa]', 'Poisson [-]', 'Thickness [mm]']:
                    align = 'center'
                    
            original_drawText(row, col, celltxt, fgcolor, align)

        self.table.drawText = custom_drawText

        self.load_data()
        self.adjust_column_widths()
        self.set_cell_colors()
        self.custom_draw_merged_header()

        button_frame = tk.Frame(root, bg='#FCFCFA')
        button_frame.pack(pady=10)
        FlatButton(button_frame, text="Compute!", command=self.get_data,
                   font=("Segoe UI", 10, "bold"), bg="#2C5282", fg="white",
                   activebg="#1A365D", activefg="white",
                   relief=tk.FLAT, padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        FlatButton(button_frame, text="Save Data", command=self.save_data,
                   font=("Segoe UI", 10, "bold"), bg="#2D5F3F", fg="white",
                   activebg="#1E4029", activefg="white",
                   relief=tk.FLAT, padx=20, pady=5).pack(side=tk.LEFT, padx=5)

        legend = tk.Frame(root, bg='#FCFCFA')
        legend.pack(pady=(0, 4))
        for color, label in [('#FFFFFF', 'Editable Input'), ('#CCCCCC', 'Fixed Label'), ('#D6AFB9', 'Calculation Result')]:
            sw = tk.Frame(legend, bg=color, width=14, height=14,
                          highlightbackground='#888888', highlightthickness=1)
            sw.pack_propagate(False)
            sw.pack(side=tk.LEFT, padx=(8, 2))
            tk.Label(legend, text=label, font=("Segoe UI", 9), bg='#FCFCFA').pack(side=tk.LEFT, padx=(0, 8))
        self.result_text = tk.Text(root, height=3, width=200, bg='#F5F2E8',
                                   relief=tk.FLAT, font=("Consolas", 10))
        self.result_text.pack(pady=10, padx=10)

    def load_data(self):
        data = {
            'Layer':           ['Layered System', 'layer #', '1', '2', '3', '4', '5', '', '', '', 'Stress[MPa]='],
            'Modulus [MPa]':   ['Layered System', 'Modulus [MPa]', '8000', '400', '300', '200', '100', '', '', '', '0.95'],
            'Poisson [-]':     ['Layered System', 'Poisson [-]', '0.30', '0.35', '0.35', '0.40', '0.40', '', '', '', 'Radius[mm]='],
            'Thickness [mm]':  ['Layered System', 'Thickness [mm]', '150', '240', '300', '500', 'semi-inf', '', '', '', '150'],
            '':                ['', '', '', '', '', '', '', '', '', '', ''],
            'Evaluation point':['Evaluation point #', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'r [mm]':          ['r [mm]', '0', '100', '200', '300', '450', '600', '900', '1200', '1500', '1800'],
            'Deflection [μm]': ['Deflection [μm]', '', '', '', '', '', '', '', '', '', ''],
        }
        df = pd.DataFrame(data)
        self.model.data = {}
        self.model.columnNames = []
        self.model.reclist = []
        self.model.importDict(df.to_dict(orient='index'))
        self.table.redraw()

    def custom_draw_merged_header(self):
        merged = ['Layer', 'Modulus [MPa]', 'Poisson [-]', 'Thickness [mm]']
        total = sum(self.model.columnwidths.get(c, 100) for c in merged)
        if not hasattr(self, 'merged_label'):
            self.merged_label = tk.Label(self.frame, text="Layered System",
                font=("Arial", 10, "bold"), bg='#CCCCCC', relief='solid', borderwidth=1)
            self.merged_label.place(x=4, y=4, width=total, height=30)
        if not hasattr(self, 'merged_label_2'):
            self.merged_label_2 = tk.Label(self.frame, text="Loading",
                font=("Arial", 10, "bold"), bg='#CCCCCC', relief='solid', borderwidth=1)
            self.merged_label_2.place(x=4, y=30 + 30 * 8 + 4, width=total, height=30)
        if not hasattr(self, 'merged_label_3'):
            self.merged_label_3 = tk.Label(self.frame, text="Evaluation point #",
                font=("Arial", 10, "bold"), bg='#CCCCCC', relief='solid', borderwidth=1)
            self.merged_label_3.place(x=4 + 490, y=4, width=100, height=30)
        if not hasattr(self, 'merged_label_4'):
            self.merged_label_4 = tk.Label(self.frame, text="r [mm]",
                font=("Arial", 10, "bold"), bg='#CCCCCC', relief='solid', borderwidth=1)
            self.merged_label_4.place(x=4 + 590, y=4, width=80, height=30)
        if not hasattr(self, 'merged_label_5'):
            self.merged_label_5 = tk.Label(self.frame, text="Deflection [μm]",
                font=("Arial", 10, "bold"), bg='#CCCCCC', relief='solid', borderwidth=1)
            self.merged_label_5.place(x=4 + 670, y=4, width=100, height=30)

    def adjust_column_widths(self):
        cw = {'Layer': 100, 'Modulus [MPa]': 110, 'Poisson [-]': 90,
              'Thickness [mm]': 110, 'Evaluation point': 100,
              'r [mm]': 80, 'Deflection [μm]': 100}
        for k, v in cw.items():
            if k in self.model.columnNames:
                self.model.columnwidths[k] = v
        for col in self.model.columnNames:
            if col not in cw:
                self.model.columnwidths[col] = max(len(col) * 8, 80)
        self.table.redraw()

    def col_idx(self, col_name):
        try:
            return self.model.columnNames.index(col_name)
        except ValueError:
            return None

    def color_cells(self, rows, cols, color):
        if rows is not None and not isinstance(rows, (list, tuple)):
            rows = [rows]
        if cols is not None and not isinstance(cols, (list, tuple)):
            cols = [cols]
        self.table.setcellColor(rows, cols, color, key='bg')

    def set_cell_colors(self):
        grey, white, pink = '#CCCCCC', '#FFFFFF', '#D6AFB9'
        # Fixed labels 
        for rc in [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(6,3),(1,1),(1,2),(1,3),
                   (10,0),(10,2),(1,5),(2,5),(3,5),(4,5),(5,5),(6,5),
                   (7,5),(8,5),(9,5),(10,5),(0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4),(9,4),(10,4),
                   (7,0),(7,1),(7,2),(7,3),(8,0),(8,1),(8,2),(8,3)]:
            self.color_cells(rc[0], rc[1], grey)
        # Editable 
        for rc in [(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3),
                   (5,1),(5,2),(5,3),(6,1),(6,2),(10,1),(10,3),
                   (10,6),(9,6),(8,6),(7,6),(6,6),(5,6),(4,6),(3,6),(2,6),(1,6)]:
            self.color_cells(rc[0], rc[1], white)
        # Results 
        for rc in [(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(8,7),(9,7),(10,7)]:
            self.color_cells(rc[0], rc[1], pink)
        self.table.redraw()

    def highlight_results(self):
        c = self.col_idx('Deflection [μm]')
        if c is None:
            return
        rows = [r for r in range(11)
                if str(self.model.data.get(r, {}).get('Deflection [μm]', '')) not in ('', 'nan', 'None')]
        if rows:
            self.color_cells(rows, c, '#D6AFB9')
        self.table.redraw()

    def validate_inputs(self):
        errors = []
        for li, row in enumerate(range(2, 7)):
            d = self.model.data.get(row, {})
            # Modulus
            try:
                mod = float(str(d.get('Modulus [MPa]', '')))
                if not (0 < mod < 1_000_000):
                    errors.append(f"Layer {li+1}: Modulus {mod:g} MPa out of range (0 < E < 1,000,000).")
            except (ValueError, TypeError):
                errors.append(f"Layer {li+1}: Modulus is not a valid number.")
            # Poisson
            try:
                pois = float(str(d.get('Poisson [-]', '')))
                if not (-1.0 <= pois <= 0.5):
                    errors.append(f"Layer {li+1}: Poisson {pois:g} out of physical range (-1 to 0.5).")
            except (ValueError, TypeError):
                errors.append(f"Layer {li+1}: Poisson is not a valid number.")
            # Thickness
            if li < 4:
                thk_raw = str(d.get('Thickness [mm]', ''))
                try:
                    thk = float(thk_raw)
                    if li == 0:
                        if not (0 < thk <= 1_000_000):
                            errors.append(f"Layer 1: thickness {thk:g} mm must be > 0 and <= 1,000,000.")
                    else:
                        if not (0 <= thk <= 1_000_000):
                            errors.append(f"Layer {li+1}: thickness {thk:g} mm must be 0 to 1,000,000.")
                except (ValueError, TypeError):
                    errors.append(f"Layer {li+1}: thickness is not a valid number.")
        return errors

    def get_data(self):
        self.frame.focus_set()

        errors = self.validate_inputs()
        if errors:
            messagebox.showerror("Invalid Input",
                                 "Please fix the following before computing:\n\n" +
                                 "\n".join(f"• {e}" for e in errors))
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "[ERROR] Calculation aborted. See dialog for details.")
            return

        df = pd.DataFrame.from_dict(self.model.data, orient='index')
        try:
            df.to_csv('data.csv', index=False)
        except Exception:
            pass

        df_num = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        arr = np.ascontiguousarray(df_num.iloc[:, :8].to_numpy(), dtype=np.float64)

        try:
            result = WuWan_pavement_forward.Calculation(arr, calc_grad=False)
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"[ERROR] Calculation failed: {e}")
            return

        u = list(result.result_displacement)
        values_um = [round(float(v) * 1000, 1) for v in u]

        for i in range(1, 11):
            if i in self.model.data:
                idx = i - 1
                if idx < len(u):
                    self.model.data[i]['Deflection [mm]'] = str(float(u[idx]))
                    self.model.data[i]['Deflection [μm]'] = str(values_um[idx])

        self.highlight_results()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, 'Compute Finished!   ')
        if hasattr(self, 'parent_menu') and getattr(self.parent_menu, 'is_profile_open', False):
            self.parent_menu._draw_profile_plot()

    def save_data(self):
        folder = filedialog.askdirectory(title="Select folder to save data")
        if not folder:
            return
        try:
            df = pd.DataFrame.from_dict(self.model.data, orient='index')
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(folder, f"forward_data_{ts}.csv")
            log_path = os.path.join(folder, f"forward_data_{ts}.log")
            df.to_csv(csv_path, index=False)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("WuWan Forward Calculation Data\nSaved: " + ts + "\n\n=== Input / Output Table ===\n")
                f.write(df.to_string() + "\n")
            messagebox.showinfo("Saved", f"Data saved to:\n{csv_path}\n{log_path}")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Data saved to: {folder}   ")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save data:\n{e}")


if __name__ == '__main__':
    root = tk.Tk()
    main_menu = MainMenu(root)

    def on_closing():
        root.quit()
        root.destroy()
        import sys
        sys.exit()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()