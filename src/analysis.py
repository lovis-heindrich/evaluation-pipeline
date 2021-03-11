import numpy as np
from math import pi
import matplotlib.pyplot as plt

# Rounds number to two decimal places to next smallest multiple of 0.05
def get_lower_bound(n):
    n = int(n*100)
    r = n % 5
    return float(n - r)/100

# Based on https://python-graph-gallery.com/radar-chart/
def make_radar_plot(colname, df_list, label, title="", num_steps=4, filename=None, max=1, subplot=111, clear=True):
    if clear:
        plt.clf()
    categories = df_list[0].index.values.tolist()
    
    # Check if performance is lower than 0.85 to set lower x-axis bound
    smallest = 0.85
    biggest = max
    for df in df_list:
        values = df.loc[:,colname].values.tolist()
        smallest = np.min((smallest, np.min(values)))
        biggest = np.max((biggest, np.max(values)))
    smallest = get_lower_bound(smallest)
    biggest = get_lower_bound(biggest)

    # number of variable
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    ax = plt.subplot(subplot, polar=True)
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Align x-axis names to avoid overlap
    for name, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            name.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            name.set_horizontalalignment('left')
        else:
            name.set_horizontalalignment('right')
    
    # Draw ylabels
    ax.set_rlabel_position(0)

    # Create list of labels from smallest_value+0.05 to 0.95
    yticks = []
    step_size = (biggest-smallest) / num_steps
    steps = [np.round(smallest + i * step_size, 2) for i in range(num_steps)]
    yticks = steps[1:]
    ylabel = [str(y) for y in yticks]
    plt.yticks(yticks, ylabel, color="grey", size=7)
    plt.ylim(smallest,biggest)
    
    for i, df in enumerate(df_list):
        values = df.loc[:,colname].values.tolist()
        # Repeat first value for circular data structure
        values += values[:1]
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid', label = label[i])
        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(-0.15, 0))
    plt.title(title, y=1.1)
    plt.tight_layout()
    plt.figure(figsize=(20,10))

    if filename:
        plt.savefig(filename, dpi=300)


def make_radar_subplots(colnames, df_list, label, title="", num_steps=4, filename=None, maxval=1):
    plt.clf()
    categories = df_list[0].index.values.tolist()
    my_dpi=300
    plt.figure(figsize=(3000/my_dpi, 2000/my_dpi), dpi=my_dpi)
    measure = colnames[0]
    levels = colnames[1]

    for i, level in enumerate(levels):
        # Check if performance is lower than 0.85 to set lower x-axis bound
        smallest = 0.85
        biggest = maxval
        for df in df_list:
            values = df.loc[:,(measure, level)].values.tolist()
            smallest = np.min((smallest, np.min(values)))
            biggest = np.max((biggest, np.max(values)))
        smallest = get_lower_bound(smallest)
        biggest = get_lower_bound(biggest)

        # number of variable
        N = len(categories)

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Initialise the spider plot
        ax = plt.subplot(int("23"+str(i+1)), polar=True)
        
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='grey', size=8)
        
        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        # Align x-axis names to avoid overlap
        for name, angle in zip(ax.get_xticklabels(), angles):
            if angle in (0, np.pi):
                name.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                name.set_horizontalalignment('left')
            else:
                name.set_horizontalalignment('right')
        
        # Draw ylabels
        ax.set_rlabel_position(0)

        # Create list of labels from smallest_value+0.05 to 0.95
        yticks = []
        step_size = (biggest-smallest) / num_steps
        steps = [np.round(smallest + i * step_size, 2) for i in range(num_steps)]
        yticks = steps[1:]
        ylabel = [str(y) for y in yticks]
        plt.yticks(yticks, ylabel, color="grey", size=7)
        plt.ylim(smallest,biggest)
        
        for i, df in enumerate(df_list):
            values = df.loc[:,(measure, level)].values.tolist()
            # Repeat first value for circular data structure
            values += values[:1]
            # Plot data
            ax.plot(angles, values, linewidth=1, linestyle='solid', label = label[i])
            # Fill area
            ax.fill(angles, values, 'b', alpha=0.1)
        
        
        plt.title(level, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(2.2, 1))
    plt.subplots_adjust(hspace = 0.4, wspace=0.5)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=16)
    #plt.figure(figsize=(30,20))

    if filename:
        fig.savefig(filename, dpi=my_dpi)


# Based on https://python-graph-gallery.com/radar-chart/
def make_score_plot(df, title="", filename=None):
    plt.clf()
    categories = df.index.values.tolist()

    # number of variable
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Align x-axis names to avoid overlap
    for name, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            name.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            name.set_horizontalalignment('left')
        else:
            name.set_horizontalalignment('right')
    
    # Draw ylabels
    ax.set_rlabel_position(0)

    # Create list of labels from smallest_value+0.05 to 0.95
    yticks = [0,1,2,3,4]
    ylabel = [str(y) for y in yticks]
    plt.yticks(yticks, ylabel, color="grey", size=7)
    plt.ylim(-0.5,5)
    
    for col in df.columns.values:
        values = df.loc[:,col].values.tolist()
        # Repeat first value for circular data structure
        values += values[:1]
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid', label = col)
        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(-0.15, 0))
    plt.title(title, y=1.1)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)