# This code  has plotting functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import zscore
import statsmodels.api as sm
from scipy import stats


import os


def set_figure():
    from matplotlib import rcParams
        # set the plotting values
    rcParams['figure.figsize'] = [12, 12]
    rcParams['font.size'] = 12
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    rcParams['axes.spines.right']  = False
    rcParams['axes.spines.top']    = False
    rcParams['axes.spines.left']   = True
    rcParams['axes.spines.bottom'] = True

    params = {'axes.labelsize': 'large',
            'axes.titlesize':'large',
            'xtick.labelsize':'large',
            'ytick.labelsize':'large',
            'legend.fontsize': 'large'}
    
    rcParams.update(params)

def save_figure(name,base_path):
    plt.savefig(os.path.join(base_path, f'{name}.png'), 
                bbox_inches='tight', transparent=False)
   # plt.savefig(os.path.join(base_path, f'{name}.svg'), 
   #             bbox_inches='tight', transparent=True)

def save_figureAll(name,base_path):
    plt.savefig(os.path.join(base_path, f'{name}.png'), 
                bbox_inches='tight', transparent=False)
    plt.savefig(os.path.join(base_path, f'{name}.svg'), 
               bbox_inches='tight', transparent=True)

def lineplot_sessions(dffTrace_mean,analysis_params, colormap,leg_label,
                    zscoreRun, duration='5sec', savefigname=None,
                    savefigpath=None, ax = None):
    from scipy.ndimage import gaussian_filter1d
    color =  sns.color_palette(colormap, len(analysis_params))
    sessionsData ={}

    for indx, params in enumerate(analysis_params) :
        array = dffTrace_mean[params]   
        if np.array_equal(array, np.array(None)):
            sessionsData[indx] = None
        else:
            nCell = array.shape[0]
            analysis_window = array.shape[1]    
            array = np.reshape(array, (nCell, analysis_window))
            if zscoreRun:
                sessionsData[indx]= zscore(array, axis = 1)
                # normalise to first second
                sessionsData[indx]= (array - np.nanmean(array[:,30:59], axis = 1)[:,None]) / np.nanstd(array[:,30:59], axis = 1)[:,None]
              #  baseline = np.nanmean(array[:, 0:60], axis=1)  # (n_cells,)
             #   sessionsData[indx]= sessionsData[indx] = array - baseline[:, None]
            else:
                sessionsData[indx]= array
    
            sortedInd = np.array(np.nanmean(sessionsData[indx][:, 60:(60 + 15)], axis=1)).argsort()[::-1]
            # cut the last 5% of the rows
            sortedInd = sortedInd[:int(len(sortedInd)*(10/12))]
            
    step = 30 # for x ticks
    if int(duration[0]) > 5.1:
        print('Traces are only avaiable for 5 sec after onset defined time')
    else:
        yaxis_length = int(duration[0])*30

    for idx, sessionData in enumerate(sessionsData):
        plot_data = sessionsData[idx]
        plot_data = plot_data[sortedInd]
        if type(plot_data) != type(None):
            
            x_labels = np.linspace(-2, 6, plot_data.shape[1], dtype = int)
            xticks = np.arange(0, len(x_labels), step)
            xticklabels = x_labels[::step]
            df = pd.DataFrame(plot_data).melt()
            # Smooth the data using lowess method from statsmodels
            x=df['variable']
            y=df['value']
            time = (np.arange(-60, 180) / 30)
            #lowess_smoothed = sm.nonparametric.lowess(y, x,frac=0.05, return_sorted=True)
            mean_trace = pd.Series(np.nanmean(plot_data, axis=0))
            smooth_trace = mean_trace.rolling(window=25, center=True, min_periods=1).mean()
            # Create the plot
            ax = sns.lineplot(x=x, y=y, color=color[idx], 
                              label=leg_label, ax = ax)
            ax.plot(time, smooth_trace, color='deeppink', linewidth=2)
            # ax = sns.lineplot(x=lowess_smoothed[:, 0], y=lowess_smoothed[:, 1], 
            #                   color=color[idx], linewidth = 3 , ax = ax)

            ax.axvline(x=60, color='k', linestyle='--')
            ax.set_xticks (ticks = xticks, labels= xticklabels)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_xlim(30,60+yaxis_length)
            ax.set_xlabel('Time (sec)')
            if zscoreRun:
                ax.set_ylabel(r'$\Delta F/F$')
            else:
                ax.set_ylabel(r'$\Delta F/F$')
            ax.legend(loc='upper left', fontsize='small',
                      bbox_to_anchor=(0.6, 1), borderaxespad=0., frameon=False)
            #plt.title(analysis_params[idx])
    if savefigpath != None:
        save_figureAll(savefigname,savefigpath)

def heatmap_sessions(dffTrace_mean,analysis_params, colormap,
                       selectedSession, ymax, duration='5sec', savefigname=None,
                         savefigpath=None, axis = None, cellNumberOn = False):
    ## Parameters
    fRate = 1000/30
    pre_frames    = 2000.0# in ms
    pre_frames    = int(np.ceil(pre_frames/fRate))
    analysisWindowDur = 500 # in ms
    analysisWindowDur = int(np.ceil(analysisWindowDur/fRate))

    sessionsData ={}

    for indx, params in enumerate(analysis_params) :
        array = dffTrace_mean[params]
        if np.array_equal(array, np.array(None)):
            sessionsData[indx] = None
        else:
            nCell = array.shape[0]
            analysis_window = array.shape[1]
            array = np.reshape(array, (nCell, analysis_window))
            array = zscore(array, axis = 1)
            array =  (array - np.nanmean(array[:,30:59], axis = 1)[:,None]) / np.nanstd(array[:,30:59], axis = 1)[:,None]
            sessionsData[indx]= array
        
    ymaxValue = ymax[0]
    yminValue = ymax[1]
    step = 30 # for x ticks
    if int(duration[0]) > 5.1:
        print('Traces are only avaiable for 5 sec after onset defined time')
    else:
        yaxis_length = int(duration[0])*30
    
        ax_im = {} 
        grid_ratio = [1 for _ in range(len(analysis_params))]
        grid_ratio.append(0.05) # for the colorbar
        if axis is None:
            fig, axes = plt.subplots(nrows=1, ncols=len(analysis_params)+1, figsize=((len(analysis_params)+1)*2, 3), 
                                    gridspec_kw={'width_ratios': grid_ratio})
            ax_im = axes
        else:
            ax_im[0] = axis

        for idx, sessionData in enumerate(sessionsData):
            plot_data = sessionsData[idx]
            if type(plot_data) != type(None):
                if selectedSession == 'WithinSession':
                    sortedInd = np.array(np.nanmean(plot_data[:, pre_frames:(pre_frames + analysisWindowDur)], axis=1)).argsort()[::-1]
                else:
                    sortedInd = np.array(np.nanmean(sessionsData[selectedSession][:, pre_frames:(pre_frames + analysisWindowDur)], axis=1)).argsort()[::-1]
                # cut the last 5% of the rows
                sortedInd = sortedInd[:int(len(sortedInd)*(10/12))]
                plot_data = plot_data[sortedInd]
                x_labels = np.linspace(-2, 6, plot_data.shape[1], dtype = int)
                xticks = np.arange(0, len(x_labels), step)
                xticklabels = x_labels[::step]
                
                ax = sns.heatmap(plot_data, vmin = yminValue, vmax = ymaxValue, 
                                 cbar = False, yticklabels = False,cmap = colormap, 
                                 ax = ax_im[idx], center = 0)
                ax.axvline(x=pre_frames, color='w', linewidth = 1)
                ax.set_xticks (ticks = xticks, labels= xticklabels)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.set_xlim(30,pre_frames+yaxis_length)
                ax.set_xlabel('Time (sec)')
                ax.set_title(analysis_params[idx])

                if cellNumberOn:
                    n_cells = plot_data.shape[0]
                    yticks = np.arange(0, n_cells, 25)
                    yticks = yticks[::-1]  # Reverse the order of yticks
                    yticklabels = yticks
                    ax.set_yticks(ticks = yticks, labels= yticklabels)
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


        if axis is None:
            # Create a color bar for all heatmaps next to the last subplot
            # Hide the y-axis label for the dummy heatmap
            axes[-1].set_yticks([])
            # Create a dummy heatmap solely for the color bar
            cax = axes[-1].inset_axes([-0.3, 0.2, 0.3, 0.3])
            sns.heatmap(np.zeros((1, 1)), ax=cax, cbar=True,  cmap=colormap, cbar_kws={'label': 'DFF','shrink': 0.5})
            axes[0].set_ylabel('Cells')
        if savefigname != None:
            save_figure(savefigname,savefigpath) 
        
def session_window_means(session_dicts, keys, pre_frames, fps):
    start = pre_frames+1
    end   = pre_frames + int(2*fps)
    out   = []
    for key in keys:
        vals = []
        for sess in session_dicts:
            arr = sess.get(key)
            if arr is None: 
                continue
            mt = zscore(np.nanmean(arr, axis=0))
            vals.append(np.nanmean(mt[start:end]))
        out.append(vals)
    return out

def mean_and_sem(list_of_lists):
    means = [np.nanmean(v) for v in list_of_lists]
    sems  = [np.nanstd(v, ddof=0)/np.sqrt(len(v)) for v in list_of_lists]
    return np.array(means), np.array(sems)

def plot_trained_vs_naive_bars(ax,
                               analysis_params,
                               trained_vals,
                               naive_vals,
                               palettes=('flare', 'gray_r'),
                               width=0.35,
                               ylabel=r'$\Delta F/F$',
                               title='Average dFF across mice\n(n = 3 mice)',
                               label = ['Naive','Trained'],
                                 ):
    """
    ax:              matplotlib Axes to draw into
    analysis_params: list of condition names (x‐axis groups)
    trained_vals:    list of lists, per‐condition session values for trained group
    naive_vals:      list of lists, per‐condition session values for naive group
    palettes:        tuple of (trained_palette, naive_palette) names for seaborn
    width:           bar width (total separation is 2*width)
    ylabel:          y‐axis label
    title:           multi‐line title (use \\n for newline)
    """
    # clear the Axes first if you want a fresh plot
    ax.clear()

    x = np.arange(len(analysis_params))
    w = width

    # compute means and SEMs
    def mean_sem(vals):
        m = np.array([np.nanmean(v) for v in vals])
        s = np.array([np.nanstd(v, ddof=0)/np.sqrt(len(v)) for v in vals])
        return m, s

    tr_means, tr_sems = mean_sem(trained_vals)
    nv_means, nv_sems = mean_sem(naive_vals)

    # palettes
    pal_tr = sns.color_palette(palettes[0], len(analysis_params))
    pal_nv = sns.color_palette(palettes[1], len(analysis_params))

    # bars
    ax.bar(x + w/2, tr_means, width=w,
           yerr=tr_sems,
           color=pal_tr,
           capsize=5,
           label='Trained',
           alpha=0.7)
    ax.bar(x - w/2, nv_means, width=w,
           yerr=nv_sems,
           color=pal_nv,
           capsize=5,
           label='Naive',
           alpha=0.7)

    # dots with jitter
    for xi, vals in enumerate(trained_vals):
        jitter = (np.random.rand(len(vals)) - 0.5) * w * 0.8
        ax.scatter(np.full_like(vals, xi + w/2) + jitter,
                   vals, color='k', s=30, alpha=0.8)
    for xi, vals in enumerate(naive_vals):
        jitter = (np.random.rand(len(vals)) - 0.5) * w * 0.8
        ax.scatter(np.full_like(vals, xi - w/2) + jitter,
                   vals, color='k', s=30, alpha=0.8)

    # horizontal zero line
    ax.axhline(y=0, color='k', linestyle='--')

    # ticks & labels
    ax.set_xticks([xi - w/4, xi + w ])
    ax.set_xticklabels(label, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # legend without box
    ax.legend(loc='upper left',
              fontsize='medium',
              bbox_to_anchor=(0, 1),
              borderaxespad=0.,
              frameon=False)

    return ax