import numpy as np
import pandas as pd
import tkinter as tk 
from tkintertable.Tables import TableCanvas
from tkintertable.TableModels import TableModel
from tkintertable import TableCanvas, TableModel
from tkinter import ttk
import matplotlib.ticker as mticker
from tkinter import *  
import WuWan_pavement_forward
from src import WuWan_pavement_inverse
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
  
class MainMenu:  
    def __init__(self, root):  
        self.root = root  
        self.root.title('WuWan v0.3 - Main Menu')  
        self.root.geometry('600x600')  
          
        # Create main frame  
        self.main_frame = ttk.Frame(root, padding="20")  
        self.main_frame.pack(fill=tk.BOTH, expand=True)  
          
        # Title  
        title_label = tk.Label(  
            self.main_frame,  
            text="WuWan Analysis System",  
            font=("Arial", 24, "bold"),  
            fg="#2C3E50"  
        )  
        title_label.pack(pady=30)  
          
        # Subtitle  
        subtitle_label = tk.Label(  
            self.main_frame,  
            text="Please Select Analysis Module",  
            font=("Arial", 18),  
            fg="#2C3E50"  
        )  
        subtitle_label.pack(pady=10)  
          
        # Button frame  
        button_frame = ttk.Frame(self.main_frame)  
        button_frame.pack(pady=30)  
          
        # Forward Calculation button  
        forward_btn = tk.Button(  
            button_frame,  
            text="Forward Calculation",  
            command=self.open_forward_calculation,  
            font=("Arial", 12, "bold"),  
            bg="#3498DB",  
            fg="black",  
            width=25,  
            height=3,  
            relief=tk.RAISED,  
            borderwidth=3,  
            cursor="hand2"  
        )  
        forward_btn.pack(pady=10)  
          
        # Back Calculation button - now enabled  
        back_btn = tk.Button(  
            button_frame,  
            text="Back Calculation",  
            command=self.open_back_calculation,  
            font=("Arial", 12, "bold"),  
            bg="#9B59B6",  
            fg="black",  
            width=25,  
            height=3,  
            relief=tk.RAISED,  
            borderwidth=3,  
            cursor="hand2"  
        )  
        back_btn.pack(pady=10)  
          
        # Other feature button  
        other_btn = tk.Button(  
            button_frame,  
            text="Bayesian Uncertainty (Coming Soon)",  
            command=self.coming_soon,  
            font=("Arial", 12),  
            bg="#95A5A6",  
            fg="white",  
            width=25,  
            height=3,  
            relief=tk.RAISED,  
            borderwidth=3,  
            state=tk.DISABLED  
        )  
        other_btn.pack(pady=10)  
          
        # Bottom information  
        info_label = tk.Label(  
            self.main_frame,  
            text="Version 0.3 | © 2026",  
            font=("Arial", 9),  
            fg="#BDC3C7"  
        )  
        info_label.pack(side=tk.BOTTOM, pady=10)  
  
    def open_forward_calculation(self):  
        """Open the forward calculation window"""  
        self.root.withdraw()  
          
        self.forward_window = tk.Toplevel(self.root)  
        self.forward_window.geometry('800x500')  
        self.forward_window.title('WuWan v0.2 - Forward Calculation')  
        self.is_profile_open = False  
          
        # === Top toolbar ===  
        top_bar = tk.Frame(self.forward_window, height=40, bg="#ecf0f1")  
        top_bar.pack(side=tk.TOP, fill=tk.X)  
          
        return_btn = tk.Button(  
            top_bar,  
            text="← Return to Main Menu",  
            command=lambda: self.return_to_main(self.forward_window),  
            font=("Arial", 9),  
            bg="#E74C3C",  
            fg="black",  
            padx=10,  
            pady=3  
        )  
        return_btn.pack(side=tk.LEFT, padx=5, pady=5)  
  
        self.profile_btn = tk.Button(  
            top_bar,  
            text="Show Profile Plot >>",  
            command=self.toggle_profile_view,  
            font=("Arial", 9, "bold"),  
            bg="#27AE60",  
            fg="black",  
            padx=10,  
            pady=3  
        )  
        self.profile_btn.pack(side=tk.LEFT, padx=5, pady=5)  
  
        # === Content container ===  
        content_container = tk.Frame(self.forward_window)  
        content_container.pack(fill=tk.BOTH, expand=True)  
  
        # === Left area (calculation table) ===  
        self.calc_frame = tk.Frame(content_container, width=800)  
        self.calc_frame.pack_propagate(0)   
        self.calc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)   
          
        # Load TableApp and store reference  
        self.table_app = TableApp(self.calc_frame)  
          
        # === Right area (plot area) ===  
        self.plot_frame = tk.Frame(content_container, bg="white", width=800)  
        self.plot_frame.pack_propagate(0)  
          
        # Placeholder label  
        self.plot_placeholder_label = tk.Label(  
            self.plot_frame,   
            text="Deflection profile will be shown here\nafter computation",  
            fg="#7f8c8d",  
            bg="white",  
            font=("Arial", 14)  
        )  
        self.plot_placeholder_label.place(relx=0.5, rely=0.5, anchor="center")  
  
        self.forward_window.protocol("WM_DELETE_WINDOW", lambda: self.return_to_main(self.forward_window))  
  
    def open_back_calculation(self):  
        """Open the back calculation window"""  
        self.root.withdraw()  
          
        self.back_window = tk.Toplevel(self.root)  
        self.back_window.geometry('800x500')  
        self.back_window.title('WuWan v0.2 - Back Calculation')  
        self.is_noise_prior_open = False  
          
        # === Top toolbar ===  
        top_bar = tk.Frame(self.back_window, height=40, bg="#ecf0f1")  
        top_bar.pack(side=tk.TOP, fill=tk.X)  
          
        return_btn = tk.Button(  
            top_bar,  
            text="← Return to Main Menu",  
            command=lambda: self.return_to_main(self.back_window),  
            font=("Arial", 9),  
            bg="#E74C3C",  
            fg="black",  
            padx=10,  
            pady=3  
        )  
        return_btn.pack(side=tk.LEFT, padx=5, pady=5)  
  
        # Noise and Prior Level button  
        self.noise_prior_btn = tk.Button(  
            top_bar,  
            text="Show Noise & Prior Level >>",  
            command=self.toggle_noise_prior_view,  
            font=("Arial", 9, "bold"),  
            bg="#F39C12",  
            fg="black",  
            padx=10,  
            pady=3  
        )  
        self.noise_prior_btn.pack(side=tk.LEFT, padx=5, pady=5)  
  
        # === Content container ===  
        self.back_content_container = tk.Frame(self.back_window)  
        self.back_content_container.pack(fill=tk.BOTH, expand=True)  
  
        # === Left area (calculation table) ===  
        self.back_calc_frame = tk.Frame(self.back_content_container, width=800)  
        self.back_calc_frame.pack_propagate(0)   
        self.back_calc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)   
          
        # Load BackCalculationTableApp  
        self.back_table_app = BackCalculationTableApp(self.back_calc_frame)
          
        # === Right area (Noise and Prior Level table) ===  
        self.noise_prior_frame = tk.Frame(self.back_content_container, bg="white", width=1200)   
        self.noise_prior_frame.pack_propagate(False)  # Prevent child widgets from changing frame size  
        self.noise_prior_frame.grid_propagate(False)   # Also prevent grid from changing size  
          
        # Load NoisePriorTableApp  
        self.noise_prior_table_app = NoisePriorTableApp(self.noise_prior_frame)  
        self.back_table_app.noise_app = self.noise_prior_table_app
        self.back_window.protocol("WM_DELETE_WINDOW", lambda: self.return_to_main(self.back_window))  
  
    def toggle_noise_prior_view(self):  
        """Toggle view logic - Back Calculation Noise & Prior"""  
        if not self.is_noise_prior_open:  
            self.back_window.geometry('800x500')  
            self.noise_prior_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)  
            self.noise_prior_btn.config(text="<< Hide Noise & Prior Level")  
            self.is_noise_prior_open = True  
        else:  
            self.noise_prior_frame.pack_forget()  
            self.back_window.geometry('1200x500')  
            self.noise_prior_btn.config(text="Show Noise & Prior Level >>")  
            self.is_noise_prior_open = False  

    def toggle_profile_view(self):  
        """Toggle view logic - Forward Calculation Profile Plot"""  
        if not self.is_profile_open:  
            # Check if deflection data has been computed; if not, run compute first  
            self._ensure_deflection_computed()  
            self._draw_profile_plot()  
            self.forward_window.geometry('1200x500')   
            self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)  
            self.profile_btn.config(text="<< Hide Profile Plot")  
            self.is_profile_open = True  
        else:  
            self.plot_frame.pack_forget()  
            self.forward_window.geometry('800x500')  
            self.profile_btn.config(text="Show Profile Plot >>")  
            self.is_profile_open = False  

    def _ensure_deflection_computed(self):  
        """Check whether deflection data has been computed; if not, trigger compute."""  
        model_data = self.table_app.model.data  
        has_data = False  
        for r in range(1, 11):  
            v = str(model_data.get(r, {}).get('Deflection [μm]', ''))  
            if v not in ('', 'nan', 'None', '0', '0.0'):  
                has_data = True  
                break  
        if not has_data:  
            # Deflection data not yet computed, execute compute first  
            self.table_app.get_data()  

    def _draw_profile_plot(self):  
        """Draw the deflection profile plot in the right panel."""  
        # Clear previous widgets in plot_frame  
        for widget in self.plot_frame.winfo_children():  
            widget.destroy()  

        # Extract r and deflection data from the table model  
        model_data = self.table_app.model.data  
        r_values = []  
        defl_values = []  
        for r in range(1, 11):  
            row_data = model_data.get(r, {})  
            r_str = str(row_data.get('r [mm]', ''))  
            d_str = str(row_data.get('Deflection [μm]', ''))  
            try:  
                r_val = float(r_str)  
                d_val = float(d_str)  
                r_values.append(r_val)  
                defl_values.append(d_val)  
            except (ValueError, TypeError):  
                continue  

        if not r_values:  
            placeholder = tk.Label(  
                self.plot_frame,  
                text="No valid data to plot.",  
                fg="#e74c3c", bg="white", font=("Arial", 13)  
            )  
            placeholder.place(relx=0.5, rely=0.5, anchor="center")  
            return  

        # Create matplotlib figure with a clean, publication-quality style  
        fig = Figure(figsize=(5.2, 4.2), dpi=100) 
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111) 

        # Plot deflection: positive deflection drawn downward (negative y-axis)  
        ax.plot(r_values, [-d for d in defl_values],  
                marker='o', markersize=5, linewidth=1.8,  
                color='#2C3E50', markerfacecolor='#E74C3C',  
                markeredgecolor='#2C3E50', markeredgewidth=0.8,  
                zorder=3)  

        # Configure axes: only show positive x-axis and negative y-axis  
        ax.set_xlim(left=0)  
        y_min = -max(defl_values) * 1.15  
        ax.set_ylim(bottom=y_min, top=0)  

        # Axis labels  
        ax.set_xlabel('r  [mm]', fontsize=11, fontfamily='serif', labelpad=8)  
        ax.set_ylabel('Deflection  [μm]', fontsize=11, fontfamily='serif', labelpad=8)  

        # Spine configuration: only keep bottom and left spines  
        ax.spines['top'].set_visible(False)  
        ax.spines['right'].set_visible(False)  
        ax.spines['left'].set_linewidth(0.8)  
        ax.spines['bottom'].set_linewidth(0.8)  

        # Tick parameters  
        ax.tick_params(axis='both', which='major', direction='in',  
                       length=4, width=0.8, labelsize=9)  
        ax.tick_params(axis='both', which='minor', direction='in',  
                       length=2, width=0.5)  
        ax.minorticks_on()  

        # Light grid  
        ax.grid(True, which='major', linestyle='--', linewidth=0.4,  
                color='#BFBFBF', alpha=0.7)  
        ax.grid(True, which='minor', linestyle=':', linewidth=0.25,  
                color='#D9D9D9', alpha=0.5)  

        # Y-axis tick labels: display as positive values (absolute deflection)  
        #def abs_formatter(x, pos):
            #return f'{abs(x):.0f}'
        
        #ax.yaxis.set_major_formatter(mticker.FuncFormatter(abs_formatter))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{abs(x):.0f}'))


        fig.tight_layout(pad=1.5)  

        # Embed matplotlib figure into tkinter frame  
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)  
        canvas.draw()  
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  

        # Store reference to prevent garbage collection  
        self._profile_fig = fig  
        self._profile_canvas = canvas  

    def toggle_noise_prior_view(self):  
        """Toggle view logic - Back Calculation Noise & Prior"""  
        if not self.is_noise_prior_open:  
            self.back_window.geometry('1400x500')   
            self.noise_prior_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)  
            self.noise_prior_btn.config(text="<< Hide Noise & Prior Level")  
            self.is_noise_prior_open = True  
        else:  
            self.noise_prior_frame.pack_forget()  
            self.back_window.geometry('800x500')  
            self.noise_prior_btn.config(text="Show Noise & Prior Level >>")  
            self.is_noise_prior_open = False  
  
    def return_to_main(self, window):  
        window.destroy()  
        self.root.deiconify()  
  
    def coming_soon(self):  
        from tkinter import messagebox  
        messagebox.showinfo("Coming Soon", "  This feature is under development")  
  
  
class TableApp:  
    """Table application for Forward Calculation"""  
    def __init__(self, root):  
        self.frame = ttk.Frame(root)  
        self.frame.pack(fill=tk.BOTH, expand=True)  
  
        self.model = TableModel()  
        self.table = TableCanvas(self.frame, model=self.model, rowheight=30, grid_color="black", linewidth=2.5, rowselectedcolor="light blue", cellbackgr="#E84445")  
        self.table.show()  
        self.table.tablecolheader.grid_forget()  
        self.table.tablerowheader.grid_forget()  
  
        self.load_data()  
        self.adjust_column_widths()  
        self.set_cell_colors()  
        self.custom_draw_merged_header()  
          
        button_frame = ttk.Frame(root)  
        button_frame.pack(pady=5)  
        tk.Button(  
            button_frame,  
            text="Compute!",  
            command=self.get_data,  
            font=("Arial", 10, "bold"),  
            bg="#696969",  
            fg="black",  
            padx=20,  
            pady=5  
        ).pack()  
  
        self.result_text = tk.Text(root, height=3, width=200)  
        self.result_text.pack(pady=10)  
  
    def load_data(self):  
        data = {  
            'Layer':            ['Layered System', 'layer #', '1', '2', '3', '4', '5', '', '', '','Stress [MPa]'],  
            'Modulus [MPa]':    ['Layered System', 'Modulus [MPa]', '4000', '400', '300', '200', '100', '', '', '','0.95'],  
            'Poisson [-]':      ['Layered System', 'Poisson [-]', '0.30', '0.35', '0.35', '0.40', '0.40', '', '', '', 'Radius [mm]'],  
            'Thickness [mm]':   ['Layered System', 'Thickness [mm]', '150', '240', '300', '500', 'semi-inf', '', '', '','150'],  
            '':  ['', '', '', '', '', '', '', '', '', '', ''],  
            'Evaluation point': ['Evaluation point #', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10' ],  
            'r [mm]':           ['r [mm]', '0', '300', '600', '900', '1200', '1500', '1800', '2000', '3000', '4000'],  
            'Deflection [μm]':  ['Deflection [μm]', '', '', '', '', '', '', '', '', '', ''],  
        }  
        df = pd.DataFrame(data)  
  
        self.model.data = {}  
        self.model.columnNames = []  
        self.model.reclist = []  
        self.model.importDict(df.to_dict(orient='index'))  
        self.table.redraw()  
  
    def custom_draw_merged_header(self):  
        merged_cols = ['Layer', 'Modulus [MPa]', 'Poisson [-]', 'Thickness [mm]']  
          
        if not hasattr(self, 'merged_label'):  
            self.merged_label = tk.Label(  
                self.frame,   
                text="Layered System",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',  
                relief='solid',  
                borderwidth=1  
            )  
            total_width = sum(self.model.columnwidths.get(col, 100) for col in merged_cols)  
            self.merged_label.place(x=4, y=4, width=total_width, height=30)  
  
        if not hasattr(self, 'merged_label_2'):  
            self.merged_label_2 = tk.Label(  
                self.frame,   
                text="Loading",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            row_height = 30  
            header_height = 30    
            y_position = header_height + (row_height * 8)    
            total_width = sum(self.model.columnwidths.get(col, 100) for col in merged_cols)  
            self.merged_label_2.place(x=4, y=y_position + 4, width=total_width, height=row_height)  
  
        if not hasattr(self, 'merged_label_3'):  
            self.merged_label_3 = tk.Label(  
                self.frame,   
                text="Evaluation point #",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            row_height = 30  
            width = 490  
            self.merged_label_3.place(x=4 + width, y=4, width=100, height=row_height)  
  
        if not hasattr(self, 'merged_label_4'):  
            self.merged_label_4 = tk.Label(  
                self.frame,   
                text="r [mm]",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            row_height = 30  
            width = 590  
            self.merged_label_4.place(x=4 + width, y=4, width=80, height=row_height)  
  
        if not hasattr(self, 'merged_label_5'):  
            self.merged_label_5 = tk.Label(  
                self.frame,   
                text="Deflection [μm]",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            row_height = 30  
            width = 670  
            self.merged_label_5.place(x=4 + width, y=4, width=100, height=row_height)  
  
    def adjust_column_widths(self):  
        column_widths = {  
            'Layer': 100,  
            'Modulus [MPa]': 110,  
            'Poisson [-]': 90,  
            'Thickness [mm]': 110,  
            'Evaluation point': 100,  
            'r [mm]': 80,  
            'Deflection [μm]': 100,  
        }  
          
        for col_name, width in column_widths.items():  
            if col_name in self.model.columnNames:  
                self.model.columnwidths[col_name] = width  
          
        for col in self.model.columnNames:  
            if col not in column_widths:  
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
        grey   = '#CCCCCC'     
        yellow = '#FFE0C1'    
  
        self.color_cells(1, 0, grey)   
        self.color_cells(2, 0, grey)   
        self.color_cells(3, 0, grey)  
        self.color_cells(4, 0, grey)  
        self.color_cells(5, 0, grey)   
        self.color_cells(6, 0, grey)     
        self.color_cells(6, 3, grey)      
        self.color_cells(1, 1, grey)  
        self.color_cells(1, 2, grey)  
        self.color_cells(1, 3, grey)  
        self.color_cells(10, 0, grey)  
        self.color_cells(10, 2, grey)  
        self.color_cells(1, 5, grey)  
        self.color_cells(2, 5, grey)  
        self.color_cells(3, 5, grey)  
        self.color_cells(4, 5, grey)  
        self.color_cells(5, 5, grey)  
        self.color_cells(6, 5, grey)  
        self.color_cells(7, 5, grey)  
        self.color_cells(8, 5, grey)  
        self.color_cells(9, 5, grey)  
        self.color_cells(10, 5, grey)  
        self.color_cells(2, 1, yellow)   
        self.color_cells(2, 2, yellow)   
        self.color_cells(2, 3, yellow)   
        self.color_cells(3, 1, yellow)   
        self.color_cells(3, 2, yellow)   
        self.color_cells(3, 3, yellow)   
        self.color_cells(4, 1, yellow)   
        self.color_cells(4, 2, yellow)   
        self.color_cells(4, 3, yellow)  
        self.color_cells(5, 1, yellow)   
        self.color_cells(5, 2, yellow)   
        self.color_cells(5, 3, yellow)    
        self.color_cells(6, 1, yellow)   
        self.color_cells(6, 2, yellow)   
        self.color_cells(10, 1, yellow)   
        self.color_cells(10, 3, yellow)   
        self.color_cells(10, 6, yellow)   
        self.color_cells(9, 6, yellow)  
        self.color_cells(8, 6, yellow)  
        self.color_cells(7, 6, yellow)   
        self.color_cells(6, 6, yellow)  
        self.color_cells(5, 6, yellow)  
        self.color_cells(4, 6, yellow)   
        self.color_cells(3, 6, yellow)  
        self.color_cells(2, 6, yellow)  
        self.color_cells(1, 6, yellow)  
  
        self.table.redraw()  
  
    def highlight_results(self):  
        deflect_col = self.col_idx('Deflection [μm]')  
        if deflect_col is None:  
            return  
        rows_to_hl = []  
        for r in range(11):  
            v = str(self.model.data.get(r, {}).get('Deflection [μm]', ''))  
            if v not in ('', 'nan', 'None'):  
                rows_to_hl.append(r)  
        if rows_to_hl:  
            self.color_cells(rows_to_hl, deflect_col, '#D6AFB9')    
        self.table.redraw()  
  
    def get_data(self):  
        df = pd.DataFrame.from_dict(self.model.data, orient='index')  
        df.to_csv('data.csv', index=False)  
  
        df_num = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)  
        arr = np.ascontiguousarray(df_num.to_numpy(), dtype=np.float64)  
          
        # Note: the Calculation function must be imported  
        input_data = np.ascontiguousarray(arr)
        result = WuWan_pavement_forward.Calculation(input_data, calc_grad=False)
        u1, u2, u3, u4, u5, u6, u7, u8, u9, u10 = result.result_displacement
        values_mm  = [u1, u2, u3, u4, u5, u6, u7, u8, u9, u10]
        values_um  = [round(v * 1000, 1) for v in values_mm]
  
        df.loc[1:10, 'Deflection [mm]'] = values_mm  
        df.loc[1:10, 'Deflection [μm]'] = values_um  
        self.model.data = df.to_dict(orient='index')  
  
        self.highlight_results()  
  
        self.result_text.insert(tk.END, f'Compute Finished!   ')  
        print("Computation Finished!")  
  
  
class BackCalculationTableApp:  
    """Table application for Back Calculation"""  
    def __init__(self, root):  
        self.frame = ttk.Frame(root)  
        self.frame.pack(fill=tk.BOTH, expand=True)  
  
        self.model = TableModel()  
        self.table = TableCanvas(self.frame, model=self.model, rowheight=30, grid_color="black", linewidth=2.5, rowselectedcolor="light blue", cellbackgr="#E84445")  
        self.table.show()  
        self.table.tablecolheader.grid_forget()  
        self.table.tablerowheader.grid_forget()  
  
        self.load_data()  
        self.adjust_column_widths()  
        self.set_cell_colors()  
        self.custom_draw_merged_header()  
          
        button_frame = ttk.Frame(root)  
        button_frame.pack(pady=5)  
        tk.Button(  
            button_frame,  
            text="Back Calculate!",  
            command=self.get_data,  
            font=("Arial", 10, "bold"),  
            bg="#9B59B6",  
            fg="black",  
            padx=20,  
            pady=5  
        ).pack()  
  
        self.result_text = tk.Text(root, height=3, width=200)  
        self.result_text.pack(pady=10)  
  
    def load_data(self):  
        data = {  
            'Layer':            ['Layered System', 'layer #', '1', '2', '3', '4', '5', '', '', '','Stress [MPa]'],  
            'Modulus [MPa]':    ['Layered System', 'Modulus [MPa]', '', '', '', '', '', '', '', '','0.95'],  
            'Poisson [-]':      ['Layered System', 'Poisson [-]', '0.30', '0.35', '0.35', '0.40', '0.40', '', '', '', 'Radius [mm]'],  
            'Thickness [mm]':   ['Layered System', 'Thickness [mm]', '50', '200', '300', '600', 'semi-inf', '', '', '','150'],  
            '':  ['', '', '', '', '', '', '', '', '', '', ''],  
            'Evaluation point': ['Evaluation point #', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10' ],  
            'r [mm]':           ['r [mm]', '0', '300', '600', '900', '1200', '1500', '1800', '2000', '3000', '4000'],  
            'Deflection [μm]':  ['Deflection [μm]', '660.51133', '344.71229', '220.93542', '170.4919', '139.30202', '117.31299', '100.74293', '91.83003', '62.17525', '46.11301'],  
        }  
        df = pd.DataFrame(data)  
  
        self.model.data = {}  
        self.model.columnNames = []  
        self.model.reclist = []  
        self.model.importDict(df.to_dict(orient='index'))  
        self.table.redraw()  
  
    def custom_draw_merged_header(self):  
        merged_cols = ['Layer', 'Modulus [MPa]', 'Poisson [-]', 'Thickness [mm]']  
          
        if not hasattr(self, 'merged_label'):  
            self.merged_label = tk.Label(  
                self.frame,   
                text="Layered System",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',  
                relief='solid',  
                borderwidth=1  
            )  
            total_width = sum(self.model.columnwidths.get(col, 100) for col in merged_cols)  
            self.merged_label.place(x=4, y=4, width=total_width, height=30)  
  
        if not hasattr(self, 'merged_label_2'):  
            self.merged_label_2 = tk.Label(  
                self.frame,   
                text="Loading",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            row_height = 30  
            header_height = 30    
            y_position = header_height + (row_height * 8)    
            total_width = sum(self.model.columnwidths.get(col, 100) for col in merged_cols)  
            self.merged_label_2.place(x=4, y=y_position + 4, width=total_width, height=row_height)  
  
        if not hasattr(self, 'merged_label_3'):  
            self.merged_label_3 = tk.Label(  
                self.frame,   
                text="Evaluation point #",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            row_height = 30  
            width = 490  
            self.merged_label_3.place(x=4 + width, y=4, width=100, height=row_height)  
  
        if not hasattr(self, 'merged_label_4'):  
            self.merged_label_4 = tk.Label(  
                self.frame,   
                text="r [mm]",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            row_height = 30  
            width = 590  
            self.merged_label_4.place(x=4 + width, y=4, width=80, height=row_height)  
  
        if not hasattr(self, 'merged_label_5'):  
            self.merged_label_5 = tk.Label(  
                self.frame,   
                text="Deflection [μm]",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            row_height = 30  
            width = 670  
            self.merged_label_5.place(x=4 + width, y=4, width=100, height=row_height)  
  
    def adjust_column_widths(self):  
        column_widths = {  
            'Layer': 100,  
            'Modulus [MPa]': 110,  
            'Poisson [-]': 90,  
            'Thickness [mm]': 110,  
            'Evaluation point': 100,  
            'r [mm]': 80,  
            'Deflection [μm]': 100,  
        }  
          
        for col_name, width in column_widths.items():  
            if col_name in self.model.columnNames:  
                self.model.columnwidths[col_name] = width  
          
        for col in self.model.columnNames:  
            if col not in column_widths:  
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
        grey   = '#CCCCCC'     
        yellow = '#FFE0C1'    
        light_green = '#90EE90'    # For displaying input deflection data  
        light_purple = '#DDA0DD'   # For displaying back-calculated modulus results  
  
        self.color_cells(1, 0, grey)   
        self.color_cells(2, 0, grey)   
        self.color_cells(3, 0, grey)  
        self.color_cells(4, 0, grey)  
        self.color_cells(5, 0, grey)   
        self.color_cells(6, 0, grey)     
        self.color_cells(6, 3, grey)      
        self.color_cells(1, 1, grey)  
        self.color_cells(1, 2, grey)  
        self.color_cells(1, 3, grey)  
        self.color_cells(10, 0, grey)  
        self.color_cells(10, 2, grey)  
        self.color_cells(1, 5, grey)  
        self.color_cells(2, 5, grey)  
        self.color_cells(3, 5, grey)  
        self.color_cells(4, 5, grey)  
        self.color_cells(5, 5, grey)  
        self.color_cells(6, 5, grey)  
        self.color_cells(7, 5, grey)  
        self.color_cells(8, 5, grey)  
        self.color_cells(9, 5, grey)  
        self.color_cells(10, 5, grey)  
          
        # Poisson and Thickness - yellow (input)  
        self.color_cells(2, 2, yellow)   
        self.color_cells(2, 3, yellow)   
        self.color_cells(3, 2, yellow)   
        self.color_cells(3, 3, yellow)   
        self.color_cells(4, 2, yellow)   
        self.color_cells(4, 3, yellow)  
        self.color_cells(5, 2, yellow)   
        self.color_cells(5, 3, yellow)    
        self.color_cells(6, 2, yellow)   
          
        # Loading parameters - yellow  
        self.color_cells(10, 1, yellow)   
        self.color_cells(10, 3, yellow)   
          
        # r values - yellow  
        self.color_cells(10, 6, yellow)   
        self.color_cells(9, 6, yellow)  
        self.color_cells(8, 6, yellow)  
        self.color_cells(7, 6, yellow)   
        self.color_cells(6, 6, yellow)  
        self.color_cells(5, 6, yellow)  
        self.color_cells(4, 6, yellow)   
        self.color_cells(3, 6, yellow)  
        self.color_cells(2, 6, yellow)  
        self.color_cells(1, 6, yellow)  
          
        # Modulus cells - light purple (output / to be calculated)  
        self.color_cells(2, 1, light_purple)   
        self.color_cells(3, 1, light_purple)   
        self.color_cells(4, 1, light_purple)   
        self.color_cells(5, 1, light_purple)   
        self.color_cells(6, 1, light_purple)   
          
        # Deflection cells - light green (input for back calculation)  
        deflect_col = self.col_idx('Deflection [μm]')  
        if deflect_col is not None:  
            for r in range(1, 11):  
                self.color_cells(r, deflect_col, light_green)  
  
        self.table.redraw()  
  
    def highlight_results(self):  
        """Highlight the back-calculated modulus results"""  
        modulus_col = self.col_idx('Modulus [MPa]')  
        if modulus_col is None:  
            return  
        rows_to_hl = [2, 3, 4, 5, 6]  # Modulus for layers 1-5  
        self.color_cells(rows_to_hl, modulus_col, '#98FB98')  # Light green indicating computation complete  
        self.table.redraw()  
  
    def get_data(self):  
        df_left = pd.DataFrame.from_dict(self.model.data, orient='index')
        df_left.to_csv('back_calc_data.csv', index=False) 
        df_num = df_left.apply(pd.to_numeric, errors='coerce').fillna(0.0) 
        arr_main = np.ascontiguousarray(df_num.to_numpy(), dtype=np.float64) 
        arr_noise = None

        noise_dict = self.noise_app.model.data
        df_noise = pd.DataFrame.from_dict(noise_dict, orient='index')
        df_noise.to_csv('noise_prior_data.csv', index=False)
        df_noise_num = df_noise.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        arr_noise = np.ascontiguousarray(df_noise_num.to_numpy(), dtype=np.float64)
        
        inverse_result = WuWan_pavement_inverse.Backcalculation(arr_main, arr_noise, forward_module=WuWan_pavement_forward) 
        
        modulus_values = inverse_result
          
        for i, val in enumerate(modulus_values):  
            df_left.loc[i+2, 'Modulus [MPa]'] = val  
          
        self.model.data = df_left.to_dict(orient='index')  
        self.highlight_results()  
  
        self.result_text.insert(tk.END, f'Back Calculation Finished!   ')  
        print("Back Calculation Finished!")  
  
  
class NoisePriorTableApp:  
    """Table application for Noise and Prior Level settings"""  
    def __init__(self, root):  
        self.frame = ttk.Frame(root)  
        self.frame.pack(fill=tk.BOTH, expand=True)  
  
        self.model = TableModel()  
        self.table = TableCanvas(self.frame, model=self.model, rowheight=30, grid_color="black", linewidth=2.5, rowselectedcolor="light blue", cellbackgr="#FFFFFF")  
        self.table.show()  
        self.table.tablecolheader.grid_forget()  
        self.table.tablerowheader.grid_forget()  
  
        self.load_data()  
        self.adjust_column_widths()  
        self.set_cell_colors()  
        self.custom_draw_merged_header()  
  
    def load_data(self):  
        data = {  
            'Layer': ['Layered System', 'layer #', '1', '2', '3', '4', '5', '', '', '', 'Stress noise level [%]'],  
            'Modulus prior range lower [MPa]': ['Layered System', 'Modulus [MPa]', '1000', '100', '80', '20', '15', '', '', '','0.025 '],  
            'Modulus prior range upper [MPa]': ['Layered System', 'Modulus [MPa]', '25000', '8000', '600', '500', '150', '', '', '',''],  
            'Thickness noise level [mm]': ['Layered System', 'Thickness [mm]', '10', '10', '10', '10', '-', '', '', '', ''],  
            'Evaluation point': ['Evaluation point #', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],  
            'r noise level [mm]': ['noise [mm]', '0', '2', '2', '2', '2', '2', '2', '2', '2', '2'],  
            'Deflection noise level [μm]': ['noise [μm]', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],  
        }  
        df = pd.DataFrame(data)  
  
        self.model.data = {}  
        self.model.columnNames = []  
        self.model.reclist = []  
        self.model.importDict(df.to_dict(orient='index'))  
        self.table.redraw()  
  
    def custom_draw_merged_header(self):  
        merged_cols = ['Layer', 'Modulus prior range lower [MPa]', 'Modulus prior range upper [MPa]', 'Thickness noise level [mm]']  
        row_height = 30  
          
        # Left merged header - Prior Settings  
        if not hasattr(self, 'noise_merged_label_1'):  
            self.noise_merged_label_1 = tk.Label(  
                self.frame,   
                text="Prior & Noise Settings for Layered System",   
                font=("Arial", 10, "bold"),  
                bg='#F39C12',  
                relief='solid',  
                borderwidth=1  
            )  
            total_width = sum(self.model.columnwidths.get(col, 100) for col in merged_cols)  
            self.noise_merged_label_1.place(x=4, y=4, width=total_width, height=row_height)  
  
        # Loading noise settings header  
        if not hasattr(self, 'noise_merged_label_2'):  
            self.noise_merged_label_2 = tk.Label(  
                self.frame,   
                text="Loading Noise Settings",   
                font=("Arial", 10, "bold"),  
                bg='#E67E22',    
                relief='solid',  
                borderwidth=1  
            )  
            header_height = 30    
            y_position = header_height + (row_height * 8)    
            total_width = sum(self.model.columnwidths.get(col, 100) for col in merged_cols)  
            self.noise_merged_label_2.place(x=4, y=y_position + 4, width=total_width, height=row_height)  
  
        # Right side Evaluation point header  
        if not hasattr(self, 'noise_merged_label_3'):  
            self.noise_merged_label_3 = tk.Label(  
                self.frame,   
                text="point #",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            # Calculate total width of left-side columns  
            left_cols_width = sum(self.model.columnwidths.get(col, 100) for col in merged_cols)  
            x_position = 4 + left_cols_width  
            self.noise_merged_label_3.place(x=x_position, y=4, width=50, height=row_height)  
  
        # r noise level header  
        if not hasattr(self, 'noise_merged_label_4'):  
            self.noise_merged_label_4 = tk.Label(  
                self.frame,   
                text="r noise [mm]",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            left_cols_width = sum(self.model.columnwidths.get(col, 100) for col in merged_cols)  
            eval_col_width = self.model.columnwidths.get('Evaluation point', 100)  
            x_position = 4 + left_cols_width + eval_col_width  
            self.noise_merged_label_4.place(x=x_position, y=4, width=80, height=row_height)  
  
        # Deflection noise level header  
        if not hasattr(self, 'noise_merged_label_5'):  
            self.noise_merged_label_5 = tk.Label(  
                self.frame,   
                text="Deflection noise [μm]",   
                font=("Arial", 10, "bold"),  
                bg='#CCCCCC',    
                relief='solid',  
                borderwidth=1  
            )  
            left_cols_width = sum(self.model.columnwidths.get(col, 100) for col in merged_cols)    
            eval_col_width = self.model.columnwidths.get('Evaluation point', 100)  
            r_noise_col_width = self.model.columnwidths.get('r noise level [mm]', 80)  
            x_position = 4 + left_cols_width + eval_col_width + r_noise_col_width  
            self.noise_merged_label_5.place(x=x_position, y=4, width=120, height=row_height)  
  
    def adjust_column_widths(self):  
        column_widths = {  
            'Layer': 80,  
            'Modulus prior range lower [MPa]': 80,  
            'Modulus prior range upper [MPa]': 80,  
            'Thickness noise level [mm]': 80,  
            'Evaluation point': 50,  
            'r noise level [mm]': 80,  
            'Deflection noise level [μm]': 120,  
        }  
          
        for col_name, width in column_widths.items():  
            if col_name in self.model.columnNames:  
                self.model.columnwidths[col_name] = width  
          
        for col in self.model.columnNames:  
            if col not in column_widths:  
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
        grey = '#CCCCCC'     
        orange_light = '#FDEBD0'   # Light orange for prior input  
        yellow_light = '#FFF9C4'   # Light yellow for noise input  
        green_light = '#E8F5E9'    # Light green for evaluation point noise  
          
        # === Left side Layered System area ===  
        # Header row (row 1)  
        self.color_cells(1, 0, grey)   
        self.color_cells(1, 1, grey)   
        self.color_cells(1, 2, grey)   
        self.color_cells(1, 3, grey)   
          
        # Layer numbers (column 0, rows 2-6)  
        for r in range(2, 7):  
            self.color_cells(r, 0, grey)  
          
        # Prior range values - orange light (columns 1-2, rows 2-6)  
        for r in range(2, 7):  
            self.color_cells(r, 1, orange_light)   # Lower bound  
            self.color_cells(r, 2, orange_light)   # Upper bound  
          
        # Thickness noise level - yellow light (column 3, rows 2-6)  
        for r in range(2, 7):  
            self.color_cells(r, 3, yellow_light)  
          
        # === Loading Noise Settings (row 10) ===  
        self.color_cells(10, 0, grey)          # Stress noise level [%] label  
        self.color_cells(10, 1, yellow_light)  # Stress noise value  
        self.color_cells(10, 2, grey)          # Radius noise level [mm] label  
        self.color_cells(10, 3, yellow_light)  # Radius noise value  
          
        # === Right side Evaluation point area ===  
        # Evaluation point column  
        eval_col = self.col_idx('Evaluation point')  
        if eval_col is not None:  
            self.color_cells(1, eval_col, grey)    # Header  
            for r in range(2, 11):  
                self.color_cells(r, eval_col, grey)  # Point numbers  
          
        # r noise level column - green light  
        r_noise_col = self.col_idx('r noise level [mm]')  
        if r_noise_col is not None:  
            self.color_cells(1, r_noise_col, grey)   # Header  
            for r in range(2, 11):  
                self.color_cells(r, r_noise_col, green_light)  
          
        # Deflection noise level column - green light  
        defl_noise_col = self.col_idx('Deflection noise level [μm]')  
        if defl_noise_col is not None:  
            self.color_cells(1, defl_noise_col, grey)  # Header  
            for r in range(2, 11):  
                self.color_cells(r, defl_noise_col, green_light)  
  
        self.table.redraw()  
  
    def get_noise_prior_data(self):  
        """Retrieve noise and prior settings data"""  
        df = pd.DataFrame.from_dict(self.model.data, orient='index')  
        return df  
 
  
  
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
