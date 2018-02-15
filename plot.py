import sys
import numpy as np
import matplotlib.pyplot as plt




def plot_scatter(ax, prng, nb_samples=100):
    """Scatter plot.
    """
    for mu, sigma, marker in [(-.5, 0.75, 'o'), (0.75, 1., 's')]:
        x, y = prng.normal(loc=mu, scale=sigma, size=(2, nb_samples))
        ax.plot(x, y, ls='none', marker=marker)
    ax.set_xlabel('X-label')
    return ax


def plot_colored_sinusoidal_lines(ax):
    """Plot sinusoidal lines with colors following the style color cycle.
    """
    L = 2 * np.pi
    x = np.linspace(0, L)
    nb_colors = len(plt.rcParams['axes.prop_cycle'])
    shift = np.linspace(0, L, nb_colors, endpoint=False)
    for s in shift:
        ax.plot(x, np.sin(x + s), '-')
    ax.set_xlim([x[0], x[-1]])
    return ax


def plot_bar_graphs(ax, prng, min_value=5, max_value=25, nb_samples=5):
    """Plot two bar graphs side by side, with letters as x-tick labels.
    """
    x = np.arange(nb_samples)
    ya, yb = prng.randint(min_value, max_value, size=(2, nb_samples))
    width = 0.25
    ax.bar(x, ya, width)
    ax.bar(x + width, yb, width, color='C2')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e'])
    return ax


def plot_colored_circles(ax, prng, nb_samples=15):
    """Plot circle patches.

    NB: draws a fixed amount of samples, rather than using the length of
    the color cycle, because different styles may have different numbers
    of colors.
    """
    for sty_dict, j in zip(plt.rcParams['axes.prop_cycle'], range(nb_samples)):
        ax.add_patch(plt.Circle(prng.normal(scale=3, size=2),
                                radius=1.0, color=sty_dict['color']))
    # Force the limits to be the same across the styles (because different
    # styles may have different numbers of available colors).
    ax.set_xlim([-4, 8])
    ax.set_ylim([-5, 6])
    ax.set_aspect('equal', adjustable='box')  # to plot circles as circles
    return ax


def plot_image_and_patch(ax, prng, size=(20, 20)):
    """Plot an image with random values and superimpose a circular patch.
    """
    values = prng.random_sample(size=size)
    ax.imshow(values, interpolation='none')
    c = plt.Circle((5, 5), radius=5, label='patch')
    ax.add_patch(c)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])


def plot_histograms(ax, prng, nb_samples=10000):
    """Plot 4 histograms and a text annotation.
    """
    params = ((10, 10), (4, 12), (50, 12), (6, 55))
    for a, b in params:
        values = prng.beta(a, b, size=nb_samples)
        ax.hist(values, histtype="stepfilled", bins=30, alpha=0.8, normed=True)
    # Add a small annotation.
    ax.annotate('Annotation', xy=(0.25, 4.25), xycoords='data',
                xytext=(0.9, 0.9), textcoords='axes fraction',
                va="top", ha="right",
                bbox=dict(boxstyle="round", alpha=0.2),
                arrowprops=dict(
                          arrowstyle="->",
                          connectionstyle="angle,angleA=-95,angleB=35,rad=10"),
                )
    return ax


def plot_figure(style_label=""):
    """Setup and plot the demonstration figure with a given style.
    """
    # Use a dedicated RandomState instance to draw the same "random" values
    # across the different figures.
    prng = np.random.RandomState(96917002)

    # Tweak the figure size to be better suited for a row of numerous plots:
    # double the width and halve the height. NB: use relative changes because
    # some styles may have a figure size different from the default one.
    (fig_width, fig_height) = plt.rcParams['figure.figsize']
    fig_size = [fig_width * 2, fig_height / 2]

    fig, axes = plt.subplots(ncols=6, nrows=1, num=style_label,
                             figsize=fig_size, squeeze=True)
    axes[0].set_ylabel(style_label)

    plot_scatter(axes[0], prng)
    plot_image_and_patch(axes[1], prng)
    plot_bar_graphs(axes[2], prng)
    plot_colored_circles(axes[3], prng)
    plot_colored_sinusoidal_lines(axes[4])
    plot_histograms(axes[5], prng)

    fig.tight_layout()

    return fig


def load_data(filename=""):
    if filename == "":
	print "Please specify data file"
        sys.exit()
    with open(filename) as f:
        content = f.readlines()
        #print content
        content = [float(x.strip()) for x in content] 

    print sum(content)
    return content



def plot_mem_prof(ax, filename1="", filename2="", filename3="",  filename4=""):
    """Plot two bar graphs side by side, with letters as x-tick labels.
    """
    #y =  load_data("layer_data.log")
    y1 =  load_data(filename1)
    y2 =  load_data(filename2)
    y3 =  load_data(filename3)
    y4 =  load_data(filename4)
    print y4
    x = np.arange(len(y1))
    print x


    width = 0.5


    ax.bar(x, y1, width, label='Output data', color=[0.7, 0.7, 0.7],  edgecolor =[0.7, 0.7, 0.7])
    ax.bar(x, y2, width, bottom=y1, label='Input data', color=[0.5, 0.5, 0.5], edgecolor=[0.5, 0.5, 0.5])
    ax.bar(x, y3, width, bottom=[sum(yy) for yy in zip(y1, y2)], label='Weight', color=[0.9, 0.9, 0.9], edgecolor =[0.5, 0.5, 0.5], hatch='/////')
    ax.bar(x, y4, width, bottom=[sum(yy) for yy in zip(y1, y2, y3)], label='Other', color= [0.1, 0.1, 0.1], edgecolor= [0.1, 0.1, 0.1])

    ax.set_xticks(x)
    layer_names_yolo_608 = ["conv1", "max1", "conv2", "max2", "conv3", "conv4", "conv5", "max3", "conv6", "conv7", "conv8", "max4", "conv9", "conv10", "conv11", "conv12", "conv13", "max5", "conv14", "conv15", "conv16", "conv17", "conv18", "conv19", "conv20", "route1", "conv21", "reorg", "route2", "conv22", "conv23", "region"]
    layer_names_tiny_yolo = ["conv1", "max1", "conv2", "max2", "conv3", "max3", "conv4", "max4", "conv5", "max5", "conv6", "max6", "conv7", "conv8", "conv9", "region"]
    ax.set_xticklabels(layer_names_yolo_608, rotation=30)
    #ax.set_xticklabels(layer_names_tiny_yolo, rotation=30)
    ax.set_xlim([-1,len(x)])
    plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1)

    return ax

def plot_figure_mem_prof(style_label=""):
    """Setup and plot the demonstration figure with a given style.
    """
    # Use a dedicated RandomState instance to draw the same "random" values
    # across the different figures.
    prng = np.random.RandomState(96917002)


    #plt.set_cmap('Greys')
    #plt.rcParams['image.cmap']='Greys'


    # Tweak the figure size to be better suited for a row of numerous plots:
    # double the width and halve the height. NB: use relative changes because
    # some styles may have a figure size different from the default one.
    (fig_width, fig_height) = plt.rcParams['figure.figsize']
    fig_size = [fig_width * 1.8, fig_height / 2]

    fig, axes = plt.subplots(ncols=1, nrows=1, num=style_label, figsize=fig_size, squeeze=True)
    plt.set_cmap('Greys')

    axes.set_ylabel("Memory size (MB)")
    plot_mem_prof(axes, "./profile/yolo608/layer_output.log", "./profile/yolo608/layer_input.log", "./profile/yolo608/layer_weight.log", "./profile/yolo608/layer_other.log")
    #test_plot_bar4(axes, "./profile/tiny_yolo/layer_output.log", "./profile/tiny_yolo/layer_input.log", "./profile/tiny_yolo/layer_weight.log", "./profile/tiny_yolo/layer_other.log")
    fig.tight_layout()

    return fig

def plot_time_prof(ax, filename1="", filename2=""):
    """Plot two bar graphs side by side, with letters as x-tick labels.
    """
    y1 =  load_data(filename1)
    y2 =  load_data(filename2)
    x = np.arange(len(y1))
    print x
    width = 0.5

    ax.bar(x, y1, width, label='Computation', color=[0.7, 0.7, 0.7],  edgecolor =[0.7, 0.7, 0.7])
    ax.bar(x, y2, width, bottom=y1, label='Communication', color=[0.3, 0.3, 0.3], edgecolor=[0.3, 0.3, 0.3])


    ax.set_xticks(x)
    layer_names_yolo_608 = ["conv1", "max1", "conv2", "max2", "conv3", "conv4", "conv5", "max3", "conv6", "conv7", "conv8", "max4", "conv9", "conv10", "conv11", "conv12", "conv13", "max5", "conv14", "conv15", "conv16", "conv17", "conv18", "conv19", "conv20", "route1", "conv21", "reorg", "route2", "conv22", "conv23", "region"]
    layer_names_tiny_yolo = ["conv1", "max1", "conv2", "max2", "conv3", "max3", "conv4", "max4", "conv5", "max5", "conv6", "max6", "conv7", "conv8", "conv9", "region"]
    #ax.set_xticklabels(layer_names_yolo_608, rotation=30)
    ax.set_xticklabels(layer_names_tiny_yolo, rotation=30)
    ax.set_xlim([-1,len(x)])
    plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1)

    return ax


def plot_figure_time_prof(style_label=""):

    prng = np.random.RandomState(96917002)


    #plt.set_cmap('Greys')
    #plt.rcParams['image.cmap']='Greys'


    # Tweak the figure size to be better suited for a row of numerous plots:
    # double the width and halve the height. NB: use relative changes because
    # some styles may have a figure size different from the default one.
    (fig_width, fig_height) = plt.rcParams['figure.figsize']
    fig_size = [fig_width * 1.8, fig_height / 2]

    fig, axes = plt.subplots(ncols=1, nrows=1, num=style_label, figsize=fig_size, squeeze=True)
    plt.set_cmap('Greys')

    axes.set_ylabel("Memory size (MB)")
    #plot_time_prof(axes, "./profile/yolo608/layer_exe_time.log", "./profile/yolo608/layer_comm_time.log")
    plot_time_prof(axes, "./profile/tiny_yolo/layer_exe_time.log", "./profile/tiny_yolo/layer_comm_time.log")
    fig.tight_layout()

    return fig


if __name__ == "__main__":

    # Setup a list of all available styles, in alphabetical order but
    # the `default` and `classic` ones, which will be forced resp. in
    # first and second position.
    style_list = list(plt.style.available)  # *new* list: avoids side effects.
    style_list.remove('classic')  # `classic` is in the list: first remove it.
    style_list.sort()
    #print style_list
    style_list.insert(0, u'default')
    style_list.insert(1, u'classic')





    # Plot a demonstration figure for every available style sheet.
    #for style_label in style_list:
    #    with plt.style.context(style_label):
    #        fig = plot_figure(style_label=style_label)
    #fig = plot_figure_mem_prof()
    fig = plot_figure_time_prof()

    plt.show()
