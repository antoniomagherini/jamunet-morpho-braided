# This module contains the functions used for postprocessing the river data (discharge, water level, velocity)

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.dates import DateFormatter

from preprocessing.river_analysis_pre import *

warnings.filterwarnings("ignore", category=FutureWarning)

# specify path of flow data 
path_flow_data = r'data/flow/original'

# set formats for dataframes data type
format_main = '%d/%m/%Y'
format_alternative = '%d %B, %Y %I:%M:%p'
format_alternative2 = '%Y-%m'

# set formats for plotting 
format_day = '%b'
format_year = '%Y'

def plot_exlcude_missing_data(df, variable, loc, delta_t=1, date_type='D', fill_missing=False,
                              save_img=False):
    '''
    Plot the data with fragmented lines when data is missing within the record.

    Inputs: 
           df = pandas DataFrame, contains recorded data with possibly missing values
           variable = str, specifies recorded variable
                      options: 'Q', discharge
                               'WL', water level
           loc = str, specifies location of measurements
           delta_t = int, sets the required time interval between one observation and the next one. 
                     If the next observation is recorded after this interval it means some data is missing and the plot needs to be fragmented.
                     default: 1, used for daily records
           date_type = str, specifies the record interval. Needed for specifying the time interval between one observation and the next one
                       default: 'D', for daily data. 
                       other option: 'MS', month start (01-xx)
           fill_missing = bool, specifies if missing records are filled or not. 
                          default: False, dataframe contains  missing records. If set to True the title specifies that missing data 
                                   is replaced with daily/monthly average across years 
    
    Output: 
           none, plot of measured data. Lines are fragmented if missing data are not replaced 
    ''' 
    
    start = df['Date (yyyy-mm-dd)'].min().year
    end = df['Date (yyyy-mm-dd)'].max().year

    if variable == 'Q':
        var, unit, color = 'discharge', r'($m^3/s$)', 'red'
        ylim = [0, df.iloc[:,1].max()+5000]
    elif variable == 'WL':
        var, unit, color = 'water level', '(m)', 'navy'
        ylim = [df.iloc[:,1].min()-0.5, df.iloc[:,1].max()+0.5]

    if date_type=='D':
        time_int = 'daily' 
    elif date_type=='MS':
        time_int = 'monthly'

    col_name = fr'Average {time_int} {var} {unit}'
    ylabel = fr'{var} {unit}'.capitalize()
    title = f'Average {time_int} {var} at {loc} between {start}-{end}.'

    if fill_missing == True:
        title = title + f' Missing data are filled with {time_int} average across years'

    fig = plt.figure(figsize=[12,5])

    start_idx = 0
    for i in range(1, len(df['Date (yyyy-mm-dd)'])):
    
    # if next observation is more than 'delta_t' days after previous the plotted line is stopped 
        if df['Date (yyyy-mm-dd)'].iloc[i] - df['Date (yyyy-mm-dd)'].iloc[i-1] > np.timedelta64(delta_t, date_type): 
            plt.plot(df['Date (yyyy-mm-dd)'].iloc[start_idx:i], 
                     df[col_name].iloc[start_idx:i], color=color, linewidth=2.5)
            start_idx = i
    
    plt.plot(df['Date (yyyy-mm-dd)'].iloc[start_idx:], df[col_name].iloc[start_idx:], 
             color=color, label=f'{time_int} {var}', linewidth=2.5)
    
    # plt.title(title)
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.ylim(ylim)
    plt.xlim([df['Date (yyyy-mm-dd)'].min(), df['Date (yyyy-mm-dd)'].max()])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # plt.grid()
    plt.legend(fontsize=13)
    
    if save_img:
        plt.savefig(rf'images\report\10_appendix\river\avg_{time_int}_{var}.png', 
                    bbox_inches='tight', dpi=600)
        plt.show()
        plt.close(fig)
    else:
        plt.show()

    return None

def plot_all_years(df, delta_t=31, date_format='%b', save_img=False):
    '''
    Plot the yearly evolution of different years with overlapping lines with different colors.
    It is supposed to be used with the 'WL_d_Baha_64_94' dataframe.

    Inputs: 
           df = pandas DataFrame, contains recorded data with possibly missing values
           delta_t = int, sets the interval (days) between two ticks in the x-axis.
                     default: 31 (days)
           date_format = str, specifies the date format.
                       default: '%b', prints only the month. 
    
    Output: 
           none, plot with fragmented lines
    ''' 
    # plot each year with a different color
    cmap = plt.get_cmap('viridis')  
    num_colors = len(df.columns) - 1
    colors = [cmap(i /num_colors) for i in range(num_colors)]

    fig = plt.figure(figsize=[12,5])

    # loop over columns to plot each year separately
    for i, column in enumerate(df.columns[1:]):
        plt.plot(df['Date (dd-mm)'], df[column], color=colors[i], linewidth=2.5)

    # plt.title('Daily water level at Bahadurabad between 1964 and 1994')
    plt.xlabel('Time (month)', fontsize=14)
    plt.ylabel('Water level (m)', fontsize=14)
    plt.plot(df['Date (dd-mm)'], np.mean(df, axis=1), linewidth=3, 
             color='red', label='average water level')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlim([df['Date (dd-mm)'].iloc[0], df['Date (dd-mm)'].iloc[-1]])

    # label only months in x-axis
    date_formatter = DateFormatter(date_format)  
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.xticks(ticks=df['Date (dd-mm)'].iloc[0::delta_t], rotation=45, fontsize=13)
    plt.yticks(fontsize=13)

    ticks = np.linspace(float(min(df.columns[1:])), float(max(df.columns[1:])), num=10)
    labels = [str(int(t)) for t in ticks]

    norm = mcolors.Normalize(vmin=min(df.columns[1:].values.flatten()), vmax=max(df.columns[1:].values.flatten()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = plt.axes([0.91, 0.11, 0.05, 0.75])
    plt.colorbar(sm, cax=cax, ticks=ticks)
    cax.set_yticklabels(labels)
    
    if save_img:
        plt.savefig(rf'images\report\10_appendix\river\waterlevel_Bahadurabad.png', 
                    bbox_inches='tight', dpi=1000)
        plt.show()
        plt.close(fig)
    else:
        plt.show()
    return None

def plot(df):
    '''
    Plot the reshaped array 'WL_d_Baha_64_94_reshaped'.

    Input:
          df = pandas DataFrame

    Output:
           none, plot across years of recorded data
    '''

    fig = plt.figure(figsize=[20,7])
    
    plt.plot(df['Date (yyyy-mm-dd)'], df['Average daily water level (m)'], color='blue', label='daily average')
    plt.title('Average daily water level at Bahadurabad between 1964-1994')
    plt.xlabel('Time (years)')
    plt.ylabel('Water level (m)')
    plt.xlim([df['Date (yyyy-mm-dd)'].min(), df['Date (yyyy-mm-dd)'].max()])
    date_formatter = DateFormatter('%Y')  
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.xticks(ticks=df['Date (yyyy-mm-dd)'].iloc[0::365], rotation=45) # need to use the date of previous dataframe for x ticks 
    plt.legend()
    plt.show()

    return None

def plot_avgQ(df_avgs):
    '''
    Plot the average daily discharge computed across all years

    Input: 
          df_avgs = pandas Dataframe
    
    Output: None
    '''
    
    fig = plt.figure(figsize=[12,5])
    plt.plot(df_avgs['Date (dd-mm)'], df_avgs['Average daily water level (m)'], 
             color='blue', label='average daily discharge')
    plt.title(f'Average across years of daily discharge at Bahadurabad between 1964-1994')
    plt.xlabel('Date (-)')
    plt.ylabel(r'Water level ($m$)')
    date_formatter = DateFormatter('%b')  
    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.xticks(ticks=df_avgs['Date (dd-mm)'].iloc[0::31], rotation=45)
    plt.legend()
    plt.show()

    return None

def plot_monthlyQs(df_monthly):
    '''
    Plot the monthly records of discharge (max, min, and average)

    Input: 
          df_monthly = pandas Dataframe
    '''
    fig = plt.figure(figsize=[18,5])
    plt.plot(df_monthly['Date (yyyy-mm-dd)'], df_monthly[r'Monthly max discharge ($m^3/s$)'], 
             color='orange', label='max')
    plt.plot(df_monthly['Date (yyyy-mm-dd)'], df_monthly[r'Monthly min discharge ($m^3/s$)'], 
             color='red', label='min')
    plt.plot(df_monthly['Date (yyyy-mm-dd)'], df_monthly[r'Monthly average discharge ($m^3/s$)'], 
             color='green', label='average')
    plt.title('Monthly discharge at Bahadurabad between 1994-2000')
    plt.xlabel('Time (years)')
    plt.ylabel(r'Discharge ($m^3/s$)')
    plt.ylim(0)
    plt.xlim([df_monthly['Date (yyyy-mm-dd)'].min(), df_monthly['Date (yyyy-mm-dd)'].max()])
    plt.legend(loc='upper left')
    plt.show()

    return None

def plot_WL_9400(df):
    '''
    Plot the daily water level measured at Bahadurabad between 1994-2000

    Input:
          df = pandas Dataframe
    '''
    fig = plt.figure(figsize=[17,5])
    plt.plot(df['Date (yyyy-mm-dd)'], df['Average daily water level (m)'], 
            color='blue', label='daily average')
    plt.title('Average daily water level at Bahadurabad between 1994-2000')
    plt.xlabel('Time (years)')
    plt.ylabel(r'Water level ($m$)')
    plt.xlim([df['Date (yyyy-mm-dd)'].min(), df['Date (yyyy-mm-dd)'].max()])
    plt.legend(loc='upper left')
    plt.show()
    return None

def plot_bathymetry_WL(list_wl, list_q, path=r'data\bathymetry_1992.csv', 
                       save_path=r'images\report\2_literature\bath_clean_overall.png'):
    '''
    Plot the river bathymetry at Bahadurabad with the statistical relevant water levels

    Input:
          list_wl = list, contains statistical relevant water levels
          list_q = list, contains discharge associated to water levels
          path = str, path where bathymetry data is stored
                 default: r'images\report\2_literature\bath_clean.csv'
          save_path = str, path where image is saved
                 default: r'images\report\2_literature\bath_clean_overall.png'
    '''
    
    bath = pd.read_csv(path, header=0)
    
    colors = ['skyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'blue', 'mediumblue', 'navy']
    legend = [fr'min $Q={list_q[0]:.1f}$ $m^3/s$', fr'25% $Q={list_q[1]:.1f}$ $m^3/s$', f'50% $Q={list_q[2]:.1f}$ $m^3/s$', 
            f'mean $Q={list_q[3]:.1f}$ $m^3/s$', f'75% $Q={list_q[4]:.1f}$ $m^3/s$', f'90% $Q={list_q[5]:.1f}$ $m^3/s$', f'max $Q={list_q[6]:.1f}$ $m^3/s$']
    cmap = plt.get_cmap('inferno')
    plt.figure(figsize=(12,6))
    plt.plot(bath['X'], bath['Z'], zorder=2, color='darkgoldenrod', lw=3, label='bathymetry')
    for i in range(len(list_wl)):
        color = cmap(i / len(list_wl))  
        plt.axhline(list_wl[i], color=colors[i], zorder=1, lw=2.5, ls='--', label=legend[i])
    plt.xlim([0, np.max(bath['X'])+30])
    plt.ylim([0, np.max(bath['Z'])+0.5])
    plt.legend(fontsize=15., ncol=2, loc='lower right')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Distance (m)', fontsize=18)
    plt.ylabel('Elevation (m PWD)', fontsize=18)
    plt.tight_layout()
    # plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    return None