import sys
import numpy as np
import matplotlib.pyplot as plt

global_font_size = 16

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
    ax.set_xticklabels(layer_names_yolo_608, rotation=30)
    #ax.set_xticklabels(layer_names_tiny_yolo, rotation=30)
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

    axes.set_ylabel("Latency (s)")
    plot_time_prof(axes, "./profile/yolo608/layer_exe_time.log", "./profile/yolo608/layer_comm_time.log")
    #plot_time_prof(axes, "./profile/tiny_yolo/layer_exe_time.log", "./profile/tiny_yolo/layer_comm_time.log")
    fig.tight_layout()

    return fig

def plot_one_input_resource(ax, filename_list=[""]):
    """
	Plot two bar graphs side by side, with letters as x-tick labels.
        latency_dev_num_non_reuse.log
    """


    filename_list = ["./profile/yolo608/mr/latency_dev_num.log",
		"./profile/yolo608/5X5/latency_dev_num_reuse.log",
		"./profile/yolo608/4X4/latency_dev_num_reuse.log",
		"./profile/yolo608/3X3/latency_dev_num_reuse.log"]
    


    y1 =  load_data(filename_list[0])
    y2 =  load_data(filename_list[1])
    y3 =  load_data(filename_list[2])
    y4 =  load_data(filename_list[3])
    x = np.arange(len(y1))
    print x
    width = 0.2

    ax.bar(x-width, y1, width, label='MapReduce', color=[0.7, 0.7, 0.7],  edgecolor =[0.7, 0.7, 0.7])
    ax.bar(x, y2, width, label='WS - 5X5', color=[0.5, 0.5, 0.5],  edgecolor =[0.5, 0.5, 0.5])
    ax.bar(x+width, y3, width, label='WS - 4X4', color=[0.3, 0.3, 0.3],  edgecolor =[0.3, 0.3, 0.3])
    ax.bar(x+width+width, y4, width, label='WS - 3X3', color=[0.1, 0.1, 0.1],  edgecolor =[0.1, 0.1, 0.1])


    ax.set_xticks(x)
    ax.set_xticklabels(['1-dev','2-dev','3-dev','4-dev','5-dev','6-dev'])
    ax.set_xlim([-0.5,len(x)-0.3])
    ax.set_ylim([0, 25])
    plt.tick_params(labelsize=global_font_size)
    plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1, prop={'size': global_font_size})

    return ax

def plot_figure_one_input_resource(style_label=""):

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

    axes.set_ylabel("Latency (s)", fontsize=global_font_size)
    axes.set_xlabel("Number of devices", fontsize=global_font_size)

    plot_one_input_resource(axes, ["./profile/yolo608/latency_dev_num.log"])
    fig.tight_layout()

    return fig

def plot_reuse_benefit(ax, filename_list=[""]):
    """
	Plot two bar graphs side by side, with letters as x-tick labels.
        latency_dev_num_non_reuse.log
    """


    filename_reuse= ["./profile/yolo608/3X3/latency_dev_num_reuse.log",
		"./profile/yolo608/4X4/latency_dev_num_reuse.log",
		"./profile/yolo608/5X5/latency_dev_num_reuse.log"]
    
    filename_non_reuse= ["./profile/yolo608/3X3/latency_dev_num_non_reuse.log",
		"./profile/yolo608/4X4/latency_dev_num_non_reuse.log",
		"./profile/yolo608/5X5/latency_dev_num_non_reuse.log"]

    y2 =  load_data(filename_reuse[0]) + load_data(filename_reuse[1]) + load_data(filename_reuse[2]) 
    y1 =  load_data(filename_non_reuse[0]) + load_data(filename_non_reuse[1]) + load_data(filename_non_reuse[2])

    x = np.array([0, 1, 2, 3, 4, 5,    7, 8, 9, 10, 11, 12,    14, 15, 16, 17, 18, 19  ])
    print x
    
    width = 0.3
    xticks = [0, 1, 2, 2.5, 3, 4, 5,    7, 8, 9, 9.5, 10, 11, 12,    14, 15, 16, 16.5, 17, 18, 19 ]
    xticks_minor = [-1, 6, 13, 20]
    ax.set_xticks( xticks )
    ax.set_xticks( xticks_minor, minor=True )
    ax.grid( 'off', axis='x' )
    ax.grid( 'off', axis='x', which='minor' )

    #ax.set_xticks(x)
    xlbls = ['1-dev','2-dev','3-dev', "WS - 3X3", '4-dev','5-dev','6-dev' ,  
		'1-dev','2-dev','3-dev', "WS - 4X4", '4-dev','5-dev','6-dev' , 
		'1-dev','2-dev','3-dev', "WS - 5X5", '4-dev','5-dev','6-dev'] 
    va =          [ 0, 0, 0, -.14, 0, 0, 0,  0, 0, 0, -.14, 0, 0, 0,  0, 0, 0, -.14, 0, 0, 0]
    rotation =    [ 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0]

    ax.set_xticklabels(xlbls)
    ax.tick_params( axis='x', which='minor', direction='out', length=40 , top='off')
    ax.tick_params( axis='x', which='major', bottom='off', top='off' )

    #va = [ 0, -.05, 0, -.05, -.05, -.05 ]

    for t, y, rt in zip( ax.get_xticklabels( ), va , rotation):
	    t.set_y( y )
	    t.set_rotation( rt )


    ax.bar(x-0.5*width, y1, width, label='W/O Reuse', color=[0.5, 0.5, 0.5],  edgecolor =[0.5, 0.5, 0.5])
    ax.bar(x+0.5*width, y2, width, label='Shuffle+Reuse', color=[0.3, 0.3, 0.3],  edgecolor =[0.3, 0.3, 0.3])

    


    ax.set_xlim([-1,  20])
    #ax.set_ylim([0 ,  25])
    plt.tick_params(labelsize=global_font_size)
    plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1, prop={'size': global_font_size})

    return ax



def plot_figure_reuse_benefit(style_label=""):

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

    axes.set_ylabel("Latency (s)", fontsize=global_font_size)
    #axes.set_xlabel("Number of devices", fontsize=global_font_size)

    plot_reuse_benefit(axes, [])
    #fig.tight_layout()

    return fig



def plot_mem_footprint(ax, filename_list=[""]):
    """Plot two bar graphs side by side, with letters as x-tick labels.
    """
    #y =  load_data("layer_data.log")
    filename_list = ["profile/yolo608/mr/layer_input_1.log", 
			"profile/yolo608/mr/layer_output_1.log", 
			"profile/yolo608/mr/layer_weight.log", 
			"profile/yolo608/mr/layer_other.log"]
    y1 = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1]))
    y2 =  load_data(filename_list[2])
    y3 =  load_data(filename_list[3])


    filename_list = ["profile/yolo608/mr/layer_input_4.log", 
			"profile/yolo608/mr/layer_output_4.log", 
			"profile/yolo608/mr/layer_weight.log", 
			"profile/yolo608/mr/layer_other.log"]
    y4 = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1]))
    y5 =  load_data(filename_list[2])
    y6 =  load_data(filename_list[3])


    filename_list = ["profile/yolo608/5X5/layer_input.log", 
			"profile/yolo608/5X5/layer_output.log", 
			"profile/yolo608/5X5/layer_output_ir.log",
			"profile/yolo608/5X5/layer_weight.log", 
			"profile/yolo608/5X5/layer_other.log"]
    y7 = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1]))
    y8=  load_data(filename_list[3])
    y9 = np.array(load_data(filename_list[4])) + np.array(load_data(filename_list[2]))




    x = np.arange(len(y1))
    print x


    width = 0.3


    ax.bar(x-width, y1, width, label='Data (Original)', color=[0.5, 0.5, 0.5],  edgecolor =[0.5, 0.5, 0.5])
    ax.bar(x,       y4, width, label='Data (BODP-4)', color=[0.7, 0.7, 0.7],  edgecolor =[0.7, 0.7, 0.7])
    ax.bar(x+width, y7, width, label='Data (FG-5X5)', color=[0.9, 0.9, 0.9],  edgecolor =[0.9, 0.9, 0.9])

    ax.bar(x-width, y2, width, bottom=y1, label='Weight', color=[0.9, 0.9, 0.9], edgecolor =[0.5, 0.5, 0.5], hatch='/////')
    ax.bar(x-width, y3, width, bottom=[sum(yy) for yy in zip(y1, y2)], label='Other', color= [0.1, 0.1, 0.1], edgecolor= [0.1, 0.1, 0.1])

    ax.bar(x, y5, width, bottom=y4,  color=[0.9, 0.9, 0.9], edgecolor =[0.5, 0.5, 0.5], hatch='/////')
    ax.bar(x, y6, width, bottom=[sum(yy) for yy in zip(y4, y5)],  color= [0.1, 0.1, 0.1], edgecolor= [0.1, 0.1, 0.1])

    ax.bar(x+width, y8, width, bottom=y7,  color=[0.9, 0.9, 0.9], edgecolor =[0.5, 0.5, 0.5], hatch='/////')
    ax.bar(x+width, y9, width, bottom=[sum(yy) for yy in zip(y7, y8)],  color= [0.1, 0.1, 0.1], edgecolor= [0.1, 0.1, 0.1])

    ax.set_xticks(x)
    layer_names_yolo_608 = ["conv1", "max1", "conv2", "max2",  "conv3", "conv4", "conv5", "max3",  "conv6", "conv7", "conv8", "max4",  "conv9", "conv10", "conv11", "conv12"]
    layer_names_tiny_yolo = ["conv1", "max1", "conv2", "max2", "conv3", "max3", "conv4", "max4", "conv5", "max5", "conv6", "max6", "conv7", "conv8", "conv9", "region"]
    ax.set_xticklabels(layer_names_yolo_608, rotation=30)
    #ax.set_xticklabels(layer_names_tiny_yolo, rotation=30)
    ax.set_xlim([-1,len(x)])
    plt.legend(loc=9, ncol=5, bbox_to_anchor=(0.5, 1.16), framealpha=1)

    return ax

def plot_figure_mem_footprint(style_label=""):
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
    plot_mem_footprint(axes, [""])
    #test_plot_bar4(axes, "./profile/tiny_yolo/layer_output.log", "./profile/tiny_yolo/layer_input.log", "./profile/tiny_yolo/layer_weight.log", "./profile/tiny_yolo/layer_other.log")
    fig.tight_layout()

    return fig


def plot_mem_total(ax, filename_list=[""]):
    """Plot two bar graphs side by side, with letters as x-tick labels.
    """
    #y =  load_data("layer_data.log")
    filename_list = ["profile/yolo608/mr/layer_input_1.log", 
			"profile/yolo608/mr/layer_output_1.log", 
			"profile/yolo608/mr/layer_weight.log", 
			"profile/yolo608/mr/layer_other.log"]
    other = sum(load_data(filename_list[2])) + sum(load_data(filename_list[3]))
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1]))
    y1 = other  + max(y)
    filename_list = ["profile/yolo608/mr/layer_input_2.log", "profile/yolo608/mr/layer_output_2.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1]))
    y2 = other + max(y)
    filename_list = ["profile/yolo608/mr/layer_input_3.log", "profile/yolo608/mr/layer_output_3.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1]))
    y3 = other + max(y)
    filename_list = ["profile/yolo608/mr/layer_input_4.log", "profile/yolo608/mr/layer_output_4.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1]))
    y4 = other + max(y)
    filename_list = ["profile/yolo608/mr/layer_input_5.log", "profile/yolo608/mr/layer_output_5.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1]))
    y5 = other + max(y)
    filename_list = ["profile/yolo608/mr/layer_input_6.log", "profile/yolo608/mr/layer_output_6.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1]))
    y6 = other + max(y)

    filename_list = ["profile/yolo608/5X5/layer_input.log", 
			"profile/yolo608/5X5/layer_output.log", 
			"profile/yolo608/5X5/layer_output_ir.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1])) + np.array(load_data(filename_list[2]))
    y5X5 = other + max(y)
    filename_list = ["profile/yolo608/4X4/layer_input.log", 
			"profile/yolo608/4X4/layer_output.log", 
			"profile/yolo608/4X4/layer_output_ir.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1])) + np.array(load_data(filename_list[2]))
    y4X4 = other + max(y)
    filename_list = ["profile/yolo608/3X3/layer_input.log", 
			"profile/yolo608/3X3/layer_output.log", 
			"profile/yolo608/3X3/layer_output_ir.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1])) + np.array(load_data(filename_list[2]))
    y3X3 = other + max(y)



    x = np.arange(6)
    print x


    width = 0.2
    

    ax.bar(x-1.5*width, [y1, y2, y3, y4, y5, y6], width, label='BODP', color=[0.9, 0.9, 0.9],  edgecolor =[0.9, 0.9, 0.9])
    ax.bar(x - 0.5*width, [y3X3, y3X3, y3X3, y3X3, y3X3, y3X3], width, label='FG-3X3', color=[0.7, 0.7, 0.7],  edgecolor =[0.7, 0.7, 0.7])
    ax.bar(x + 0.5*width, [y4X4, y4X4, y4X4, y4X4, y4X4, y4X4], width, label='FG-4X4', color=[0.5, 0.5, 0.5],  edgecolor =[0.5, 0.5, 0.5])
    ax.bar(x + 1.5*width, [y5X5, y5X5, y5X5, y5X5, y5X5, y5X5], width, label='FG-5X5', color=[0.3, 0.3, 0.3],  edgecolor =[0.3, 0.3, 0.3])


    ax.set_xticks(x)

    ax.set_xticklabels(['1-dev', '2-dev', '3-dev', '4-dev', '5-dev', '6-dev'],  fontsize=global_font_size)
    #ax.set_xticklabels(layer_names_tiny_yolo, rotation=30)
    ax.set_xlim([-.5,len(x)-0.5])
    plt.legend(loc=9, ncol=5, bbox_to_anchor=(0.5, 1.16), framealpha=1 , fontsize=global_font_size)

    return ax

def plot_figure_mem_total(style_label=""):
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
    plt.tick_params(labelsize=global_font_size)
    axes.set_ylabel("Memory size (MB)" , fontsize=global_font_size)
    plot_mem_total(axes, [""])
    #test_plot_bar4(axes, "./profile/tiny_yolo/layer_output.log", "./profile/tiny_yolo/layer_input.log", "./profile/tiny_yolo/layer_weight.log", "./profile/tiny_yolo/layer_other.log")
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
    #fig = plot_figure_time_prof()
    #fig = plot_figure_one_input_resource()
    #fig = plot_figure_reuse_benefit()
    #fig = plot_figure_mem_footprint()
    fig = plot_figure_mem_total()
    plt.show()
