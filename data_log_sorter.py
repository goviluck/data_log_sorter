# COHU_data_log_sorter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import csv
import os
import seaborn as sns
import mplcursors
import time
import pickle
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog



class data_log_sorter():
    sns.set_theme()
    CONFIDENCE_INTERVAL = 1.96
    # Mapping LSB values
    FV_MAP = {100.0 : 3.17379639 * 1e-3,
            50.0 :1.586898195 * 1e-3,
            25.0 :793.4490975 * 1e-6,
            10.0 :317.3828125 * 1e-6,
            5.0 :158.6914063 * 1e-6,
            2.5 :79.34570313 * 1e-6
            }
    MV_MAP = {100.0 :3.88661931 * 1e-3,
                50.0 :1.94330966 * 1e-3,
                25.0 :971.654828 * 1e-6,
                10.0 :389.099121 * 1e-6,
                5.0 :194.549561 * 1e-6,
                2.5 :97.2747803 * 1e-6,
                1.25 :48.6373901 * 1e-6
                }
    FV_MV_LSB_MAP = {'FV':FV_MAP,
                    'MV':MV_MAP
                    }
    # Map for gridlines
    TICKS = {
            'max_FV' : [],
            'min_FV' : [],
            'max_FV_error' : [],
            'min_FV_error' : [],
            'max_MV_error' : [],
            'min_MV_error' : []
            }
    
    def __init__(self, file_name, save_folder, header=34):
        self.start_time = time.time() # for keeping time
        self.file_name = file_name
        self.header = header
        # confirm csv file and save folder
        if self.file_name[-4:] != ".csv":
            self.file_name = file_name + '.csv'
        self.save_folder = save_folder
        if self.save_folder[-1] != "\\":
            self.save_folder = save_folder + "\\"
        self.sort_columns = []
        self.FV_MV_pickle_list = [[0],[]]
        # parse csv for info
        with open(self.file_name, newline='') as csvfile: 
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if "Number_of_Tests" in row[0]:
                    self.nrows = int(row[1])   
                if "Number_of_Runs" in row[0]:
                    self.runs = int(row[1])
                if "INDEX" in row[0]:
                    self.header = i
                    break
        # create data frame
        self.df = pd.read_csv(
            filepath_or_buffer= f'{self.file_name}', 
            header = self.header, 
            skipinitialspace = True, # parse cells after initinal space
            nrows = self.nrows,
            index_col = 'INDEX')
        
        # clean columns names
        self.df.columns = [col.strip() for col in self.df.columns] 
        for col in self.df.columns:
            # remove empty columns
            if pd.isna(self.df.loc[self.df.first_valid_index(), col]):
                self.df.drop(columns=col, inplace=True) 

        # get unamed columns
        unnamed_columns = [col for col in self.df.columns if 'Unnamed' in col] 
        if unnamed_columns: #rename to test numbers
            new_names = [f'Run Num: {i+1}' for i in range(self.runs)]
            self.df.rename(columns={col: new_name for col, new_name in zip(unnamed_columns, new_names)}, inplace=True)

    # adds column of force voltages to dataframe
    def force_voltage_col_adder(self): 
        # pattern to search for in test name cell
        pattern = r'Force (-?\d+\.?\d*)V' 
        # extract number from test name and create column in data frame of values
        force_vol_list = self.df['TNAME'].str.extract(pattern, expand=False).astype(float) 
        # add to data frame
        self.df['Force Vol'] = force_vol_list 

    # adds column of pin numbers to dataframe
    def pin_num_col_adder(self): 
        pattern = r'Pin(\d+)'
        pin_list = self.df['TNAME'].str.extract(pattern,expand=False).astype(int)
        self.df['Pin Number'] = pin_list
        # get number of pins
        self.num_pins = len(set(pin_list))

    # adds column of force range values to column
    def range_col_adder(self): 
        pattern = r'(-?\d+\.?\d*)V Range'
        range_list = self.df['TNAME'].str.extract(pattern, expand=False).astype(float)
        # Convert extracted values to float and fill NaN with 0
        range_list = range_list.astype(float).fillna(0)
        self.df['Range'] = range_list

    # adds column of varible values to column
    def var_col_adder(self,var_sort,before_or_after):
        var_sort = var_sort.strip()
        if before_or_after == 'after':
            pattern = r'\s*(-?\d+\.?\d*)+[mun]?[VA]?\s?' + var_sort
        if before_or_after == 'before':
            pattern = var_sort + r'\s*(-?\d+\.?\d*)+[mun]?[VA]?\s?'
        var_list = self.df['TNAME'].str.extract(pattern, expand=False).astype(float)
        var_list = var_list.astype(float).fillna(0)
        self.df[var_sort] = var_list

    # sorts dataframe
    def first_sort(self): 
        save_location = self.save_folder + "pin_force_tnum_" + self.file_name
        sort_columns = ['Pin Number','Force Vol','TNUM']
        # sort by importance
        self.df.sort_values(by=sort_columns,ignore_index=True,inplace=True) 

    # sorts new dataframe
    def second_sort(self):    
        save_location = self.save_folder + "pin_range_force_" + self.file_name
        sort_columns = ['Pin Number','Range','Force Vol']
        self.df_New.sort_values(by=sort_columns,ignore_index=True,inplace=True)

    # sorts dataframe by pin number 
    def pin_sort(self):
        if 'Pin Number' not in self.sort_columns:
            self.sort_columns += ['Pin Number']
        self.df.sort_values(by=self.sort_columns,ignore_index=True, inplace=True)

    # sorts dataframe by variable
    def var_sort(self,var_sort):
        var_sort = var_sort.strip()
        if var_sort not in self.sort_columns:
            self.sort_columns += [var_sort]
        self.df.sort_values(by=self.sort_columns,ignore_index=True, inplace=True)
        
    # save dataframe to csv
    def save_file(self,new_file_name):
        save_location = self.save_folder + new_file_name
        # confrom save location
        if save_location[-4:] != ".csv":
            save_location += ".csv"
        self.df.to_csv(f'{save_location}')
        print(f'New data file created and saved to "{self.cyan_text(save_location)}"')

    # convert all units to volts and remove unit column
    def unit_updater(self): 
        # Define a mapping for unit prefixes
        unit_map = {'mV': 1e-3, 'uV': 1e-6, 'V': 1.0,'mA': 1e-3, 'uA': 1e-6, 'A': 1.0}  
        for k in range(1, self.runs+1):
            col_name = f'Run Num: {k}'
            # Extract the unit prefix from the 'UNITS' column
            unit = self.df['UNITS']  
            # Convert measurements using vectorization based on unit prefix
            self.df[col_name] *= unit.apply(lambda x: unit_map.get(x, 1.0))
        self.df.drop(columns='UNITS',inplace=True) # drop unit column since all unnits are V or A


    def calculate_error_bar(self, values_list):
        return np.std(values_list) / np.sqrt(len(values_list)) * self.CONFIDENCE_INTERVAL
    
    # create new dataframe with MV and FV calculations
    def calculate_MV_FV_data(self): 
        # test_name = []
        data = []
        # create column headers
        columns_headers = ['Pin Number','Force Vol','Range','TNUM','TNAME','MV avg',
                   'FV avg','MV Error avg','FV Error avg', 'MV err std', 'FV err std']
        columns_headers += [f'{measType}: {i}' for measType in ['MV', 'FV', 'MV Error', 'FV Error'] for i in range(1, self.runs + 1)  ]
        
        for i in range(0,self.nrows,3):
            pin = self.df.loc[i,"Pin Number"] # get pin number
            fv = self.df.loc[i+1,f"Force Vol"] # get force voltage, convret to mV
            vol_range = self.df.loc[i+1,f"Range"] # get force voltage, convret to mV
            test_num = self.df.loc[i,"TNUM"] # get pin number
            test_name = re.sub(r'VI100 Verify/Force Voltage Measure Voltage Vrf/(.*?)(?:\s*ADC Measure\s*)', r'\1/', self.df.loc[i+1, "TNAME"])
            # Extract data for DMMmeas, ADCmeas, and ADCerror columns for the range of runs
            # DMMmeas = [0], ADCmeas = [1], ADCerror = [2]
            runs_data = self.df.loc[i:i + 2, [f"Run Num: {j}" for j in range(1, self.runs + 1)]] #data frame of each (3) measure and all runs
            # Calculate mean, std, and other statistics using vectorized operations
            avg_MV = runs_data.iloc[2].mean() #ADC error
            avg_FV = fv
            avg_MV_err = (runs_data.iloc[1] - runs_data.iloc[0]).mean() # ADC - DMM
            avg_FV_error = (runs_data.iloc[0] - fv).mean() # DMM - FV
            MV_std_err = self.calculate_error_bar(list(runs_data.iloc[1] - runs_data.iloc[0])) # ADC - DMM
            FV_std_err = self.calculate_error_bar(list(runs_data.iloc[0] - fv))
            # Append the calculated values to the data list
            t_data = [pin, fv, vol_range, test_num, test_name, avg_MV, avg_FV, avg_MV_err, avg_FV_error, MV_std_err, FV_std_err]
            t_data.extend(runs_data.iloc[2])  # MV values (ADC error)
            t_data.extend(np.full(self.runs, fv))  # FV values (FV)
            t_data.extend(runs_data.iloc[1] - runs_data.iloc[0])  # MV Error values (ADC - DMM)
            t_data.extend(runs_data.iloc[0] - fv)  # FV Error values (DMM - FV)
            
            data.append(t_data)

        self.df_New = pd.DataFrame(data,columns=columns_headers) #create new dataframe
        
    # creates list of dataframes for specific range and pin number
    def range_pin_divider(self): 
        ranges_list = self.df_New["Range"].unique()  # Get unique range values
        pin_list = self.df_New["Pin Number"].unique()  # Get unique pin numbers
        self.df_list = [] # Initialize a list to store DataFrames
        for range_val in ranges_list:
            for pin_num in pin_list:
                mask = (self.df_New["Range"] == range_val) & (self.df_New["Pin Number"] == pin_num)
                df_subset = self.df_New[mask]  # Filter the DataFrame based on range and pin number
                self.df_list.append(df_subset)  # Append the filtered DataFrame to dfList

    # exports dataframe list to excel files in multiple sheets
    def to_excel_sheets(self): 
        excel_file = f"{self.save_folder}sheet_seperated_MV_FV_data.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            for df in self.df_list:
                # get name indo and screate sheet
                range_name = df.loc[df.first_valid_index(),'Range']
                pin_num = df.loc[df.first_valid_index(), 'Pin Number']
                sheet_name = f"FV {range_name} Pin {pin_num}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'Dataframe seperated and saved to "{self.cyan_text(excel_file)}"')

    # collects grid line ticks for graphing
    def collect_ticks(self, df):
        self.TICKS['max_FV'].append(df['FV avg'].max())
        self.TICKS['min_FV'].append(df['FV avg'].min())
        self.TICKS['max_FV_error'].append(df['FV Error avg'].max())
        self.TICKS['min_FV_error'].append(df['FV Error avg'].min())
        self.TICKS['max_MV_error'].append(df['MV Error avg'].max())
        self.TICKS['min_MV_error'].append(df['MV Error avg'].min())

    # clears grid line ticks for next plot
    def clear_ticks(self,FVorMV):
        if FVorMV == 'FV':
            self.TICKS['max_FV_error'] = []
            self.TICKS['min_FV_error'] = []
        elif FVorMV == 'MV':
            self.TICKS['max_MV_error'] = []
            self.TICKS['min_MV_error'] = []
            self.TICKS['max_FV'] = []
            self.TICKS['min_FV'] = []

    #   plots data
    def create_plot_data(self, df, FVorMV,ax):
        pin_num = df.loc[df.first_valid_index(), 'Pin Number']  # get pin num

        if self.volts_or_lsb == "VOLTS":
            # convert units to uV
            df.loc[:,f'{FVorMV} Error avg'] =  df.loc[:,f'{FVorMV} Error avg'].div(1e-6)
            df.loc[:,f'{FVorMV} err std'] =  df.loc[:,f'{FVorMV} err std'].div(1e-6)
            # plot data
            df.plot(drawstyle="steps-mid", 
                    x='FV avg',
                    xlabel =  'FV avg (V)',
                    ylabel = f'{FVorMV} Error avg (uV)', 
                    y=f'{FVorMV} Error avg', 
                    yerr = f'{FVorMV} err std', 
                    label = f"Pin: {pin_num}", 
                    ax=  ax, 
                    grid = True
                    )
        if self.volts_or_lsb == "LSB":
            range_val = df.loc[df.first_valid_index(), 'Range']
            # convert units to LSB
            df.loc[:,f'{FVorMV} Error avg'] =  df.loc[:,f'{FVorMV} Error avg'].div(self.FV_MV_LSB_MAP[FVorMV][range_val])
            df.loc[:,f'{FVorMV} err std'] =  df.loc[:,f'{FVorMV} err std'].div(self.FV_MV_LSB_MAP[FVorMV][range_val])
            df.plot(drawstyle="steps-mid", 
                                x='FV avg', 
                                xlabel =  'FV avg (V)',
                                ylabel = f'{FVorMV} Error avg (LSB)', 
                                y=f'{FVorMV} Error avg', 
                                yerr = f'{FVorMV} err std', 
                                label = f"Pin: {pin_num}", 
                                ax=  ax, 
                                grid = True)       

        self.collect_ticks(df) # collect max values for gridlines
            
    # format plots
    def fortmat_save(self,FVorMV,range_name,ax,fig):
        # get max and min for all pins for creating gridlines
        max_x, min_x = max(self.TICKS['max_FV']), min(self.TICKS['min_FV'])
        if FVorMV == 'FV':
            max_y, min_y = max(self.TICKS['max_FV_error']), min(self.TICKS['min_FV_error'])
        elif FVorMV == 'MV':
            max_y, min_y = max(self.TICKS['max_MV_error']), min(self.TICKS['min_MV_error'])
        if -min_y > max_y:
            max_y = -min_y
        # format and name plot
        ax.set_title(f'FV vs. {FVorMV} err {range_name}V range')
        ax.set_xticks(np.linspace(min_x,max_x,11))
        if self.volts_or_lsb == "VOLTS":
            ax.set_yticks(np.linspace(round(-max_y,1),round(max_y,1),11))
        if self.volts_or_lsb == "LSB":
            ax.set_yticks(np.linspace(round(-max_y,3),round(max_y,3),11))
        ax.axhline(0, color='black', linewidth=.75)
        ax.legend()   

        name = f'{self.save_folder}{self.volts_or_lsb}_plots'
        # Creates the folder if it doesn't exist
        if not os.path.exists(name):
            os.makedirs(name) 
        # save figure in save folder
        name += f'\FV_vs_{FVorMV}_err_{range_name}V_range_{self.volts_or_lsb}.'
        fig.savefig(f'{name}png') 
        self.clear_ticks(FVorMV)
        # save as pickle files so figure can be opened with same functionalities
        with open(f'{name}pickle', 'wb') as file:
            pickle.dump(fig, file)
        if FVorMV == 'FV':
            self.FV_MV_pickle_list[0].append(f'{name}pickle')
        elif FVorMV == 'MV':
            self.FV_MV_pickle_list[1].append(f'{name}pickle')
        plt.close(fig)
        
    # create plots for all data
    def bow_tie_plotter(self,volts_or_lsb):
        # plots FV avg vs FV err and FV avg vs MV err 
        self.volts_or_lsb = volts_or_lsb
        fv_axes_list = []  # List for FV axes
        mv_axes_list = []  # List for MV axes
        fv_figures_list = []  # List for FV figures
        mv_figures_list = []  # List for MV figures
        for i, df in enumerate(self.df_list):
            list_index = i // self.num_pins
            # create new plot if all pins plotted on one graph
            if (i + 1) % self.num_pins == 1:
                # add figures and axes to list for future use
                fig_fv, ax_fv = plt.subplots()
                fig_mv, ax_mv = plt.subplots()
                fv_figures_list.append(fig_fv)
                mv_figures_list.append(fig_mv)
                fv_axes_list.append(ax_fv)
                mv_axes_list.append(ax_mv)
            if i>=self.num_pins:
                # skip first range for FV plots
                self.create_plot_data(df, 'FV', fv_axes_list[list_index])
            self.create_plot_data(df, 'MV', mv_axes_list[list_index])
            # save if all pins plotted
            if (i+1)%self.num_pins == 0: 
                range_name = df.loc[df.first_valid_index(),'Range']
                if i>=self.num_pins:
                    self.fortmat_save('FV', range_name, fv_axes_list[list_index], fv_figures_list[list_index])
                self.fortmat_save('MV', range_name, mv_axes_list[list_index], mv_figures_list[list_index])
        
        print(f'Bow-tie plots created and saved in "{self.cyan_text(self.save_folder+self.volts_or_lsb)}"')
        
    # for recording how long function takes to perform
    def time_getter(self):
        self.end_time = time.time() # start time is when class is created
        execution_time = self.end_time - self.start_time
        print(f"Execution time: {round(execution_time,2)} seconds!")
        self.start_time = time.time() # end time is when method is called

    # calls all necessary functions in order for Rich plots 
    def rich_func(self,volts_or_lsb): 

        self.force_voltage_col_adder()
        self.pin_num_col_adder()
        self.range_col_adder()
        self.unit_updater()
        self.first_sort()
        self.calculate_MV_FV_data()
        self.second_sort()
        self.range_pin_divider()
        self.to_excel_sheets()
        self.time_getter()
        self.bow_tie_plotter(volts_or_lsb)
        self.time_getter()

    # makes text cyan color
    def cyan_text(self,text): 
        return  '\033[96m' + text + '\033[0m'


class data_log_sorter_app:
    # constant foreground, background, and fot for labels
    fg = "black"
    bg = "lightgray"
    custom_font = ("Times", 12)
    
    def __init__(self, root):
        # set class instance variables and set entries euqal to defualt value
        self.data = None
        self.root = root
        self.var_sort = tk.StringVar(value = "V MVRange")
        self.new_file_name = tk.StringVar(value = "sorted_file")
        self.file_name = tk.StringVar(value = "VI100 Verify_20231129_NewSet")
        self.save_folder = tk.StringVar(value = "New_Set")

        # self.file_name = tk.StringVar(value = "VI100_Verify_20231117_23h46m.csv")
        # self.save_folder = tk.StringVar(value = "Old_Set")

        # self.file_name = tk.StringVar(value = "VI100_ck_BothCards_BaseLine")
        # self.save_folder = tk.StringVar(value = "Old_Files")
        self.create_main_gui()

    # label creation
    def create_label(self, window, text, row, column, pady, padx=0, sticky="",columnspan=1):
        label = tk.Label(window, text=text, font=self.custom_font, fg=self.fg, bg=self.bg)
        label.grid(row=row, column=column, pady=pady, padx=padx, sticky=sticky,columnspan=columnspan)
        return label
    
    # entry creation
    def create_entry(self,window, textvariable, width, row, column, pady, padx=0, justify="center"):
        entry = tk.Entry(window, textvariable=textvariable, width=width, justify=justify)
        entry.grid(row=row, column=column, pady=pady, padx=padx)
        return entry
    # button creation
    def create_button(self, window, text, command, row, column, pady, padx, width=0, height=0,columnspan=1):
        button = tk.Button(window, text=text, command=command, width=width, height=height,activebackground="green")
        button.grid(row=row, column=column, pady=pady, padx=padx, columnspan=columnspan)
        return button

    # Main GUI widgets and buttons setup
    def create_main_gui(self):
        # create windwow with title, location and widgets
        self.root.title("Main GUI")
        self.root.configure(bg=self.bg)
        self.root.geometry("+0+0")
        self.create_button(self.root, "Go to Plotter", self.plotter_gui, 0, 0, 15, 25, 40, 7)
        self.create_button(self.root, "Go to csv Sorter", self.sorter_gui, 1, 0, 15, 25, 40, 7)

    # Plotter GUI
    def plotter_gui(self):
        # create windwow with title, location and widgets
        win_1 = tk.Toplevel(self.root)
        win_1.title("Plotter")
        win_1.configure(bg=self.bg)
        win_1.geometry("+500+0")

        self.create_label(win_1, "Enter the File Name:", 0, 0, 2, 2, "W")
        self.create_entry(win_1, self.file_name, 35, 0, 1, 2, 2)
        self.create_button(win_1, "Browse", self.browse_file_name, 0, 2, 5, 5)

        self.create_label(win_1, "Enter the Save Folder:", 1, 0, 2, 2, "W")
        self.create_entry(win_1, self.save_folder, 35, 1, 1, 2, 2)
        self.create_button(win_1, "Browse", self.browse_save_folder, 1, 2, 5, 5)

        self.create_button(win_1, "Create Plots (V)", self.rich_test_volts, 2, 0, 5, 5,columnspan=2)
        self.create_button(win_1, "Create Plots (LSB)", self.rich_test_lsb, 2, 1, 5, 5,columnspan=2)

    # Display Plots GUI
    def show_plots_gui(self):
        # create windwow with title, location and widgets
        win_3 = tk.Toplevel(self.root)
        win_3.title("Display Plots")
        win_3.configure(bg=self.bg)
        win_3.geometry("+500+300")

        text = "Close this window before creating new plots!\n" \
        "Note: Closing the plot deletes it from memory\n" \
        "Either create the plots again or look at saved .png files"
        self.create_label(win_3, text, 0, 1, 5, 2, columnspan=5)
        # button for showing all plots
        self.create_button(win_3, "Open all plots", lambda: self.show_all(), 1, 0, 5, 5)
        # buttons for individual plots
        labels = {
            'FV': [1.25, 2.5, 5.0, 10, 25, 50, 100],
            'MV': [1.25, 2.5, 5.0, 10, 25, 50, 100]
        }
        for idx, (label, values) in enumerate(labels.items()):
            for col, value in enumerate(values):
                if not (col == 0 and idx == 0):
                    self.create_button(win_3, 
                                       f"{label} {value} V Range", 
                                       lambda idx=idx, col=col: self.show_figure(idx, col), 
                                       idx + 1, col, 5, 5)

        win_3.protocol("WM_DELETE_WINDOW", lambda: self.close_third_gui(win_3))

    # Sorter GUI
    def sorter_gui(self):
        # create windwow with title, location and widgets
        win_2 = tk.Toplevel(self.root)
        win_2.title("csv Sorter")
        win_2.configure(bg=self.bg)
        win_2.geometry("+1000+0")
        self.win_2 = win_2

        self.create_label(win_2, "Enter the File Name:", 0, 0, 2, 2, "W")
        self.create_entry(win_2, self.file_name, 35, 0, 1, 2)
        self.create_button(win_2, "Browse", self.browse_file_name, 0, 2, 5, 5)

        self.create_label(win_2, "Enter the Save Folder:", 1, 0, 2, 2, "W")
        self.create_entry(win_2, self.save_folder, 35, 1, 1, 2)
        tk.Button(win_2, text="Browse", command=self.browse_save_folder).grid(row=1, column=2, pady=2)
        self.create_button(win_2, "Browse", self.browse_save_folder, 1, 2, 5, 5)

        text="Enter text after or before the number to sort by.\n" \
        "Click sort buttons in order of importance!\n" \
        "If units have different prefix, remove prefix from entry!"
        self.create_label(win_2, text, 2, 0, 2, columnspan=3)
        self.create_entry(win_2, self.var_sort, 35, 3, 1, 5)
        self.create_button(win_2, "Text Comes Before Number", self.var_col_before, 4, 0, 5, 5)
        self.create_button(win_2, "Text Comes After Number", self.var_col_after, 4, 1, 5, 5)
        self.create_button(win_2, "Sort by Pin Number", self.pin_col, 4, 2, 5, 5)

        self.create_label(win_2, "Enter New File Name:", 5, 0, 2, 2, "W")
        self.create_entry(win_2, self.new_file_name, 35, 5, 1, 2)
        self.create_button(win_2, "Browse", self.browse_new_file_name, 5, 2, 2, 0)
        self.columns_label = self.create_label(self.win_2, "Order to sort:", 6, 0, 2, 2, "W")
        self.create_button(win_2, "Save Sorted File", self.save_new_file, 6, 0, 2, 0,columnspan=3)
        self.create_button(win_2, "Reset", self.reset, 6, 2, 2, 0,columnspan=3)

    # deletes data frame and updates columns to be sorted
    def reset(self):
        # method when reset or save button is clicked
        try:
            self.columns_label.config(text="Order to sort:")
            self.data = None
            text = "Reset!"
            print(f"{text:_^40}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # updates the order of sort columns on GUI
    def update_sort_columns(self):
        try:
            text = "Order to sort:"
            for col in self.data.sort_columns:
                text += f"\n{col}"
            self.columns_label.config(text=text)
            self.data.sort_columns
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # show all plots on button click
    def show_all(self):
        try:
            for row in self.data.FV_MV_pickle_list:
                for plot in row:
                    if plot != 0:
                        with open(plot, 'rb') as file:
                            fig = pickle.load(file)
                            fig.show()
                            mplcursors.cursor(hover=True)
        except Exception as e:
            print(f"Error occurred while showing figure: {e}")
            messagebox.showinfo("Show All", "One or more figures are destroyed!")

    # show plot on button click
    def show_figure(self, idx, col):
        try:
            with open(self.data.FV_MV_pickle_list[idx][col], 'rb') as file:
                fig = pickle.load(file)
                fig.show()
                mplcursors.cursor(hover=True)
        except Exception as e:
            print(f"Error occurred while showing figure: {e}")
            messagebox.showinfo("Show Figure", "Figure is destroyed!")

    # close plotter GUI
    def close_third_gui(self,window):
        plt.close('all')
        self.data = None
        window.destroy()

    # browse for file on button click
    def browse_file_name(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_name.set(file_path)

    # browse for file on button click
    def browse_new_file_name(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.new_file_name.set(file_path)

    # browse for folder on button click
    def browse_save_folder(self):
        file_path = filedialog.askdirectory()
        if file_path:
            self.save_folder.set(file_path)
    
    # create data log sorter object if not currently sorting
    def create_csv_object(self):
        if type(self.data) != data_log_sorter:
            self.data = data_log_sorter(
                file_name = self.file_name.get(),
                save_folder = self.save_folder.get(),
            )
        text = "Column Added!"
        print(f"{text:_^40}")

    # create data log sorter object for plotting
    def create_csv_object_plot(self):
        self.data = data_log_sorter(
            file_name = self.file_name.get(),
            save_folder = self.save_folder.get(),
        )
        text = "Plotting"
        print(f"{text:_^40}")

    # Rich test button click volts
    def rich_test_volts(self):
        try:
            self.create_csv_object_plot()
            self.data.rich_func("VOLTS")
            self.show_plots_gui()
            messagebox.showinfo("CSV Sorter", "CSV Sorter was successful!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    # Rich test button click LSB
    def rich_test_lsb(self):
        try:
            self.create_csv_object_plot()
            self.data.rich_func("LSB")
            self.show_plots_gui()
            messagebox.showinfo("CSV Sorter", "CSV Sorter was successful!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
    # Pin column button
    def pin_col(self):
        try:
            self.create_csv_object()
            self.data.pin_num_col_adder()
            self.data.pin_sort()
            self.update_sort_columns()
            # messagebox.showinfo("CSV Sorter", "CSV Sorter was successful!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # Variable column button before
    def var_col_before(self):
        try:
            self.create_csv_object()
            self.data.var_col_adder(self.var_sort.get().strip(),"before")
            self.data.var_sort(self.var_sort.get().strip())
            self.update_sort_columns()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # Variable column button after
    def var_col_after(self):
        try:
            self.create_csv_object()
            self.data.var_col_adder(self.var_sort.get().strip(),"after")
            self.data.var_sort(self.var_sort.get().strip())
            self.update_sort_columns()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # Save file button
    def save_new_file(self):
        try:
            self.data.save_file(self.new_file_name.get())
            messagebox.showinfo("CSV Sorter", "CSV Sorter was successful!")
            self.reset()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}\nAdd columns first!")


if __name__ == "__main__":
    root = tk.Tk()
    app = data_log_sorter_app(root)
    root.protocol("WM_DELETE_WINDOW", root.quit())
    root.mainloop()

# data = data_log_sorter('VI100 Verify_20231129_NewSet.csv','New_Set')
# data.rich_func("VOLTS")