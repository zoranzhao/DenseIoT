import sys
import numpy as np
import matplotlib.pyplot as plt

global_font_size = 20
larger_size = global_font_size + 10
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


def load_data_vector(filename=""):
    if filename == "":
	print "Please specify data file"
        sys.exit()
    print filename
    ret_data = []
    with open(filename) as f:
        content = f.readlines()
	print content
	for line in content:
           print [float(x.strip()) for x in line.split(",")]
           print sum([float(x.strip()) for x in line.split(",")])
	   ret_data.append(sum([float(x.strip()) for x in line.split(",")]))

    print ret_data
    return ret_data




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


    width = 0.7


    out_d = ax.bar(x, y1, width, label='Output data', color=[0.7, 0.7, 0.7],  edgecolor =[0,0,0], linewidth=0.5)
    in_d = ax.bar(x, y2, width, bottom=y1, label='Input data', color=[0.5, 0.5, 0.5], edgecolor =[0,0,0], linewidth=0.5)
    weight = ax.bar(x, y3, width, bottom=[sum(yy) for yy in zip(y1, y2)], label='Weight', color=[0.9, 0.9, 0.9], edgecolor =[0,0,0], hatch='/////', linewidth=0.5)
    other = ax.bar(x, y4, width, bottom=[sum(yy) for yy in zip(y1, y2, y3)], label='Other', color= [0.1, 0.1, 0.1], edgecolor =[0,0,0], linewidth=0.5)



    ax.bar(x, y1, width, label='Output data', color=[0.7, 0.7, 0.7],  edgecolor =[0.7, 0.7, 0.7], linewidth=0.5)
    ax.bar(x, y3, width, bottom=[sum(yy) for yy in zip(y1, y2)], label='Weight', color=[0.9, 0.9, 0.9], edgecolor =[0,0,0], hatch='/////', linewidth=0.5)
    ax.bar(x, y2, width, bottom=y1, label='Input data', color=[0.5, 0.5, 0.5], edgecolor =[0.5, 0.5, 0.5], linewidth=0.5)
    ax.bar(x, y4, width, bottom=[sum(yy) for yy in zip(y1, y2, y3)], label='Other', color= [0.1, 0.1, 0.1], edgecolor =[0,0,0], linewidth=0.5)

    ax.bar(x, [sum(yy) for yy in zip(y1, y2, y3, y4)], width , label='Other', fill=False, edgecolor =[0,0,0], linewidth=0.5)


    plt.figlegend((out_d[0], in_d[0], weight[0],  other[0]), ('Output data','Input data', 'Weight', 'Other'), loc=9, ncol=4, bbox_to_anchor=(0.5, 1.01), framealpha=1)


    ax.set_xticks(x)
    layer_names_yolo_608 = ["conv1", "max1", "conv2", "max2", "conv3", "conv4", "conv5", "max3", "conv6", "conv7", "conv8", "max4", "conv9", "conv10", "conv11", "conv12", "conv13", "max5", "conv14", "conv15", "conv16", "conv17", "conv18", "conv19", "conv20", "route1", "conv21", "reorg", "route2", "conv22", "conv23", "region"]
    layer_names_tiny_yolo = ["conv1", "max1", "conv2", "max2", "conv3", "max3", "conv4", "max4", "conv5", "max5", "conv6", "max6", "conv7", "conv8", "conv9", "region"]
    ax.set_xticklabels(layer_names_yolo_608, rotation=90)
    #ax.set_xticklabels(layer_names_tiny_yolo, rotation=30)
    ax.set_xlim([-1,len(x)])
    #plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1)

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
    width = 0.7

    comp = ax.bar(x, y1, width, label='Computation', color=[0.7, 0.7, 0.7],  edgecolor =[0,0,0], linewidth=0.5)
    commu = ax.bar(x, y2, width, bottom=y1, label='Communication', color=[0.3, 0.3, 0.3], edgecolor=[0,0,0], linewidth=0.5)

    ax.bar(x, y1, width, label='Computation', color=[0.7, 0.7, 0.7],  edgecolor =[0.7, 0.7, 0.7], linewidth=0.5)
    ax.bar(x, y2, width, bottom=y1, label='Communication', color=[0.3, 0.3, 0.3], edgecolor=[0.3, 0.3, 0.3], linewidth=0.5)
    ax.bar(x, [sum(yy) for yy in zip(y1, y2)], width,  fill=False, edgecolor=[0, 0, 0], linewidth=0.5)

    ax.set_xticks(x)
    layer_names_yolo_608 = ["conv1", "max1", "conv2", "max2", "conv3", "conv4", "conv5", "max3", "conv6", "conv7", "conv8", "max4", "conv9", "conv10", "conv11", "conv12", "conv13", "max5", "conv14", "conv15", "conv16", "conv17", "conv18", "conv19", "conv20", "route1", "conv21", "reorg", "route2", "conv22", "conv23", "region"]
    layer_names_tiny_yolo = ["conv1", "max1", "conv2", "max2", "conv3", "max3", "conv4", "max4", "conv5", "max5", "conv6", "max6", "conv7", "conv8", "conv9", "region"]
    ax.set_xticklabels(layer_names_yolo_608, rotation=90)
    #ax.set_xticklabels(layer_names_tiny_yolo, rotation=30)
    ax.set_xlim([-1,len(x)])
    #plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1)

    plt.figlegend((comp[0], commu[0]), ('Computation','Communication'), loc=9, ncol=4, bbox_to_anchor=(0.5, 1.01), framealpha=1)

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

def plot_figure_multiple_input_resource_2(style_label=""):

    prng = np.random.RandomState(96917002)


    #plt.set_cmap('Greys')
    #plt.rcParams['image.cmap']='Greys'


    # Tweak the figure size to be better suited for a row of numerous plots:
    # double the width and halve the height. NB: use relative changes because
    # some styles may have a figure size different from the default one.
    (fig_width, fig_height) = plt.rcParams['figure.figsize']
    fig_size = [fig_width * 1.8, fig_height / 2]

    fig, ax = plt.subplots(ncols=1, nrows=1, num=style_label, figsize=fig_size, squeeze=True)
    plt.set_cmap('Greys')

    ax.set_ylabel("Maximum Latency (s)", fontsize=larger_size)
    ax.set_xlabel("Number of data resources out of 6 devices", fontsize=larger_size)



    
    grid = "3x3"
    config = ["MR-BODPv2", "WST-FGP-R"]
    #np.array(load_data_vector(config[0]+"/" + grid + "/single_resource/commu_size_steal.log"))+
	#np.array(load_data_vector(config[0] + "/" + grid + "/single_resource/commu_size_gateway.log"))

    y1 = load_data_vector(config[0] + "/multiple_resource/latency.log")
    y2 = load_data_vector(config[1] + "/" + grid + "/multiple_resource/latency.log")

    x = np.arange(len(y1))
    print x
    width = 0.2

    latency1 = ax.bar(x-0.5*width, y1, width, label='MR-BODP', color=[0.4, 0.4, 0.4],  edgecolor =[0, 0, 0])
    latency2 = ax.bar(x+0.5*width, y2, width, label='WST-FGP (Shuffle)', color=[0.8, 0.8, 0.8],  edgecolor =[0, 0, 0], hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(['1/6','2/6','3/6','4/6','5/6','6/6'])
    ax.set_xlim([-0.5,len(x)-0.3])
    ax.set_ylim([0, 50])

    plt.tick_params(labelsize=larger_size)


    y1 = np.array(load_data_vector(config[0]+ "/multiple_resource/commu_size.log"))
    y2 = np.array(load_data_vector(config[1]+"/" + grid + "/multiple_resource/commu_size_steal.log"))+np.array(load_data_vector(config[1] + "/" + grid + "/multiple_resource/commu_size_gateway.log"))




    ax2 = ax.twinx()
    comm1 = ax2.plot(x-width, y1, label='MR-BODP', linestyle='-.',  linewidth=4, color=[0.4, 0.4, 0.4],  marker="s", markersize=16)
    comm2 = ax2.plot(x+width, y2, label='WST-FGP (Shuffle)', linestyle='-.',  linewidth=4, color=[0.8, 0.8, 0.8],  marker="<", markersize=16)


    ax2.set_ylabel("Commu. size (MB)", fontsize=larger_size)
    ax2.set_xticklabels(['1/6','2/6','3/6','4/6','5/6','6/6'])

    ax2.set_ylim([-120, 125])
    ax2.set_yticks([0, 60, 120])

    plt.tick_params(labelsize=larger_size)
    #plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1, prop={'size': larger_size})
    plt.figlegend((latency1[0], comm1[0], latency2[0],  comm2[0]), ('MR-BODP',' ', 'WST-FGP ('+grid+' Shuffle)', ' '), loc=9, ncol=2, bbox_to_anchor=(0.5, 1), framealpha=1, prop={'size': larger_size})

    #fig.tight_layout()
    return fig
def plot_figure_multiple_input_resource_throughput_2(style_label=""):

    prng = np.random.RandomState(96917002)


    #plt.set_cmap('Greys')
    #plt.rcParams['image.cmap']='Greys'


    # Tweak the figure size to be better suited for a row of numerous plots:
    # double the width and halve the height. NB: use relative changes because
    # some styles may have a figure size different from the default one.
    (fig_width, fig_height) = plt.rcParams['figure.figsize']
    fig_size = [fig_width * 1.8, fig_height / 2]

    fig, ax = plt.subplots(ncols=1, nrows=1, num=style_label, figsize=fig_size, squeeze=True)
    plt.set_cmap('Greys')

    ax.set_ylabel("Throughput", fontsize=larger_size)
    ax.set_xlabel("Number of data resources out of 6 devices", fontsize=larger_size)



    
    grid = "3x3"
    config = ["MR-BODP", "WST-FGP-R"]
    #np.array(load_data_vector(config[0]+"/" + grid + "/single_resource/commu_size_steal.log"))+
	#np.array(load_data_vector(config[0] + "/" + grid + "/single_resource/commu_size_gateway.log"))

    y1 = load_data_vector(config[0] + "/multiple_resource/throughput.log")
    y2 = load_data_vector(config[1] + "/" + grid + "/multiple_resource/throughput.log")

    x = np.arange(len(y1))
    print x
    width = 0.2

    latency1 = ax.bar(x-0.5*width, y1, width, label='MR-BODP', color=[0.4, 0.4, 0.4],  edgecolor =[0, 0, 0])
    latency2 = ax.bar(x+0.5*width, y2, width, label='WST-FGP (Shuffle)', color=[0.8, 0.8, 0.8],  edgecolor =[0, 0, 0], hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(['1/6','2/6','3/6','4/6','5/6','6/6'])
    ax.set_xlim([-0.5,len(x)-0.3])
    ax.set_ylim([0, 0.4])

    plt.tick_params(labelsize=larger_size)





    plt.tick_params(labelsize=larger_size)
    #plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1, prop={'size': larger_size})
    plt.figlegend((latency1[0], latency2[0]), ('MR-BODP','WST-FGP ('+grid+' Shuffle)'), loc=9, ncol=2, bbox_to_anchor=(0.5, 1), framealpha=1, prop={'size': larger_size})

    #fig.tight_layout()
    return fig



def plot_figure_one_input_resource_2(style_label=""):
    """
	Plot two bar graphs side by side, with letters as x-tick labels.
        latency_dev_num_non_reuse.log
    """
    prng = np.random.RandomState(96917002)


    #plt.set_cmap('Greys')
    #plt.rcParams['image.cmap']='Greys'


    # Tweak the figure size to be better suited for a row of numerous plots:
    # double the width and halve the height. NB: use relative changes because
    # some styles may have a figure size different from the default one.
    (fig_width, fig_height) = plt.rcParams['figure.figsize']
    fig_size = [fig_width * 1.8, fig_height / 2]

    fig, ax = plt.subplots(ncols=1, nrows=1, num=style_label, figsize=fig_size, squeeze=True)
    plt.set_cmap('Greys')

    ax.set_ylabel("Latency (s)", fontsize=larger_size)
    ax.set_xlabel("Number of devices", fontsize=larger_size)



    
    grid = "3x3"
    config = ["MR-BODPv2", "WST-FGP-R"]
    #np.array(load_data_vector(config[0]+"/" + grid + "/single_resource/commu_size_steal.log"))+
	#np.array(load_data_vector(config[0] + "/" + grid + "/single_resource/commu_size_gateway.log"))

    y1 = load_data_vector(config[0] + "/single_resource/latency.log")
    y2 = load_data_vector(config[1] + "/" + grid + "/single_resource/latency.log")

    x = np.arange(len(y1))
    print x
    width = 0.2

    latency1 = ax.bar(x-0.5*width, y1, width, label='MR-BODP', color=[0.4, 0.4, 0.4],  edgecolor =[0, 0, 0])
    latency2 = ax.bar(x+0.5*width, y2, width, label='WST-FGP (Shuffle)', color=[0.8, 0.8, 0.8],  edgecolor =[0, 0, 0], hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(['1','2','3','4','5','6'])
    ax.set_xlim([-0.5,len(x)-0.3])
    ax.set_ylim([0, 30])

    plt.tick_params(labelsize=larger_size)


    y1 = np.array(load_data_vector(config[0]+ "/single_resource/commu_size.log"))
    y2 = np.array(load_data_vector(config[1]+"/" + grid + "/single_resource/commu_size_steal.log"))+np.array(load_data_vector(config[1] + "/" + grid + "/single_resource/commu_size_gateway.log"))




    ax2 = ax.twinx()
    comm1 = ax2.plot(x-width, y1, label='MR-BODP', linestyle='-.',  linewidth=4, color=[0.4, 0.4, 0.4],  marker="s", markersize=16)
    comm2 = ax2.plot(x+width, y2, label='WST-FGP (Shuffle)', linestyle='-.',  linewidth=4, color=[0.8, 0.8, 0.8],  marker="<", markersize=16)


    ax2.set_ylabel("Commu. size (MB)", fontsize=larger_size)
    ax2.set_xticklabels(['1','2','3','4','5','6'])

    ax2.set_ylim([-30, 25])
    ax2.set_yticks([0, 10, 20])

    plt.tick_params(labelsize=larger_size)
    #plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1, prop={'size': larger_size})
    plt.figlegend((latency1[0], comm1[0], latency2[0],  comm2[0]), ('MR-BODP',' ', 'WST-FGP ('+grid+' Shuffle)', ' '), loc=9, ncol=2, bbox_to_anchor=(0.5, 1), framealpha=1, prop={'size': larger_size})

    #fig.tight_layout()
    return fig

def plot_figure_multiple_input_resource(style_label=""):
    """
	Plot two bar graphs side by side, with letters as x-tick labels.
        latency_dev_num_non_reuse.log
    """
    prng = np.random.RandomState(96917002)


    #plt.set_cmap('Greys')
    #plt.rcParams['image.cmap']='Greys'


    # Tweak the figure size to be better suited for a row of numerous plots:
    # double the width and halve the height. NB: use relative changes because
    # some styles may have a figure size different from the default one.
    (fig_width, fig_height) = plt.rcParams['figure.figsize']
    fig_size = [fig_width * 1.8, fig_height / 2]

    fig, ax = plt.subplots(ncols=1, nrows=1, num=style_label, figsize=fig_size, squeeze=True)
    plt.set_cmap('Greys')

    ax.set_ylabel("Maximum Latency (s)", fontsize=global_font_size+10)
    ax.set_xlabel("Number of data resources out of 6 devices", fontsize=global_font_size+10)



    
    grid = "3x3"
    config = ["WSH-FGP-NRv2", "WST-FGP-NR", "WST-FGP-R"]
    #np.array(load_data_vector(config[0]+"/" + grid + "/single_resource/commu_size_steal.log"))+
	#np.array(load_data_vector(config[0] + "/" + grid + "/single_resource/commu_size_gateway.log"))
    y1 = load_data_vector(config[0] + "/" + grid + "/multiple_resource/latency.log")
    y2 = load_data_vector(config[1] + "/" + grid + "/multiple_resource/latency.log")
    y3 = load_data_vector(config[2] + "/" + grid + "/multiple_resource/latency.log")

    x = np.arange(len(y1))
    print x
    width = 0.2

    latency1 = ax.bar(x-width, y1, width, label='WSH-FGP (w/o Shuffle)', color=[0.5, 0.5, 0.5],  edgecolor =[0, 0, 0])
    #ax.bar(x, y2, width, label='WST-FGP (w/o Shuffle)', color=[0.5, 0.5, 0.5],  edgecolor =[0.5, 0.5, 0.5])
    #ax.bar(x+width, y3, width, label='WST-FGP (Shuffle)', color=[0.3, 0.3, 0.3],  edgecolor =[0.3, 0.3, 0.3])
    latency2 = ax.bar(x, y2, width, label='WST-FGP (w/o Shuffle)', color=[0.3, 0.3, 0.3],  edgecolor =[0, 0, 0])
    latency3 = ax.bar(x+width, y3, width, label='WST-FGP (Shuffle)', color=[0.8, 0.8, 0.8],  edgecolor =[0, 0, 0], hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(['1','2','3','4','5','6'])
    ax.set_xlim([-0.5,len(x)-0.3])
    ax.set_ylim([0, 70])

    plt.tick_params(labelsize=global_font_size+10)

    y1 = np.array(load_data_vector(config[0]+"/" + grid + "/multiple_resource/throughput.log"))
    y2 = np.array(load_data_vector(config[1]+"/" + grid + "/multiple_resource/throughput.log"))
    y3 = np.array(load_data_vector(config[2]+"/" + grid + "/multiple_resource/throughput.log"))



    ax2 = ax.twinx()
    comm1 = ax2.plot(x-width, y1, label='WSH-FGP (w/o Shuffle)', linestyle='--',  linewidth=4, color=[0.7, 0.7, 0.7],  marker="o", markersize=16)
    comm2 = ax2.plot(x, y2, label='WST-FGP (w/o Shuffle)', linestyle='--',  linewidth=4,  color=[0.2, 0.2, 0.2],  marker="p", markersize=16)
    comm3 = ax2.plot(x+width, y3, label='WST-FGP (Shuffle)', linestyle='--',  linewidth=4, color=[0.8, 0.8, 0.8],  marker="<", markersize=16)


    ax2.set_ylabel("Throughput", fontsize=global_font_size+10)
    ax2.set_xticklabels(['1/6','2/6','3/6','4/6','5/6','6/6'])

    ax2.set_ylim([-0.5, 0.35])
    #ax2.set_yticks([0, 10, 20])
    ax2.set_yticks([0, 0.1, 0.2, 0.3])


    plt.tick_params(labelsize=global_font_size+10)
    #plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1, prop={'size': global_font_size})
    plt.figlegend((latency1[0], comm1[0], latency2[0],  comm2[0], latency3[0],  comm3[0]), ('WSH-FGP',' ', 'WST-FGP', ' ', 'WST-FGP (Shuffle)',   ' '), loc=9, ncol=3, bbox_to_anchor=(0.5, 1), framealpha=1, prop={'size': global_font_size+10})

    #fig.tight_layout()
    return fig



def plot_figure_one_input_resource(style_label=""):
    """
	Plot two bar graphs side by side, with letters as x-tick labels.
        latency_dev_num_non_reuse.log
    """
    prng = np.random.RandomState(96917002)


    #plt.set_cmap('Greys')
    #plt.rcParams['image.cmap']='Greys'


    # Tweak the figure size to be better suited for a row of numerous plots:
    # double the width and halve the height. NB: use relative changes because
    # some styles may have a figure size different from the default one.
    (fig_width, fig_height) = plt.rcParams['figure.figsize']
    fig_size = [fig_width * 1.8, fig_height / 2]

    fig, ax = plt.subplots(ncols=1, nrows=1, num=style_label, figsize=fig_size, squeeze=True)
    plt.set_cmap('Greys')

    ax.set_ylabel("Latency (s)", fontsize=global_font_size+10)
    ax.set_xlabel("Number of devices", fontsize=global_font_size+10)



    
    grid = "5x5"
    config = ["WSH-FGP-NRv2", "WST-FGP-NR", "WST-FGP-R"]
    #np.array(load_data_vector(config[0]+"/" + grid + "/single_resource/commu_size_steal.log"))+
	#np.array(load_data_vector(config[0] + "/" + grid + "/single_resource/commu_size_gateway.log"))
    y1 = load_data_vector(config[0] + "/" + grid + "/single_resource/latency.log")
    y2 = load_data_vector(config[1] + "/" + grid + "/single_resource/latency.log")
    y3 = load_data_vector(config[2] + "/" + grid + "/single_resource/latency.log")

    x = np.arange(len(y1))
    print x
    width = 0.2

    latency1 = ax.bar(x-width, y1, width, label='WSH-FGP (w/o Shuffle)', color=[0.7, 0.7, 0.7],  edgecolor =[0, 0, 0])
    #ax.bar(x, y2, width, label='WST-FGP (w/o Shuffle)', color=[0.5, 0.5, 0.5],  edgecolor =[0.5, 0.5, 0.5])
    #ax.bar(x+width, y3, width, label='WST-FGP (Shuffle)', color=[0.3, 0.3, 0.3],  edgecolor =[0.3, 0.3, 0.3])
    latency2 = ax.bar(x, y2, width, label='WST-FGP (w/o Shuffle)', color=[0.5, 0.5, 0.5],  edgecolor =[0, 0, 0])
    latency3 = ax.bar(x+width, y3, width, label='WST-FGP (Shuffle)', color=[0.8, 0.8, 0.8],  edgecolor =[0, 0, 0], hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(['1','2','3','4','5','6'])
    ax.set_xlim([-0.5,len(x)-0.3])
    ax.set_ylim([0, 55])

    plt.tick_params(labelsize=global_font_size+10)

    y1 = np.array(load_data_vector(config[0]+"/" + grid + "/single_resource/commu_size.log"))
    y2 = np.array(load_data_vector(config[1]+"/" + grid + "/single_resource/commu_size_steal.log"))+np.array(load_data_vector(config[1] + "/" + grid + "/single_resource/commu_size_gateway.log"))
    y3 = np.array(load_data_vector(config[2]+"/" + grid + "/single_resource/commu_size_steal.log"))+np.array(load_data_vector(config[2] + "/" + grid + "/single_resource/commu_size_gateway.log"))



    ax2 = ax.twinx()
    comm1 = ax2.plot(x-width, y1, label='WSH-FGP (w/o Shuffle)', linestyle='-.',  linewidth=4, color=[0.7, 0.7, 0.7],  marker="o", markersize=16)
    comm2 = ax2.plot(x, y2, label='WST-FGP (w/o Shuffle)', linestyle='-.',  linewidth=4,  color=[0.2, 0.2, 0.2],  marker="p", markersize=16)
    comm3 = ax2.plot(x+width, y3, label='WST-FGP (Shuffle)', linestyle='-.',  linewidth=4, color=[0.8, 0.8, 0.8],  marker="<", markersize=16)


    ax2.set_ylabel("Commu. size (MB)", fontsize=global_font_size+10)
    ax2.set_xticklabels(['1','2','3','4','5','6'])

    ax2.set_ylim([-30, 25])
    ax2.set_yticks([0, 10, 20])

    plt.tick_params(labelsize=global_font_size+10)
    #plt.legend(loc=9, ncol=4, bbox_to_anchor=(0.5, 1.16), framealpha=1, prop={'size': global_font_size})
    plt.figlegend((latency1[0], comm1[0], latency2[0],  comm2[0], latency3[0],  comm3[0]), ('WSH-FGP',' ', 'WST-FGP', ' ', 'WST-FGP (Shuffle)',   ' '), loc=9, ncol=3, bbox_to_anchor=(0.5, 1), framealpha=1, prop={'size': global_font_size+10})

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


    filename_list = ["profile/yolo608/5X5/steal/layer_input.log", 
			"profile/yolo608/5X5/steal/layer_output.log", 
			"profile/yolo608/5X5/steal/layer_output_ir.log",
			"profile/yolo608/5X5/steal/layer_weight.log", 
			"profile/yolo608/5X5/steal/layer_other.log"]
    y7 = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1])) + np.array(load_data(filename_list[2]))
    y8=  load_data(filename_list[3])
    y9 = np.array(load_data(filename_list[4]))




    x = np.arange(len(y1))
    print x


    width = 0.25
    space = 0.30

    weight = ax.bar(x-space, y2, width, bottom=y1, label='Weight', color=[0.9, 0.9, 0.9], edgecolor =[0, 0, 0], hatch='/////', linewidth=0.5)
    ax.bar(x, y5, width, bottom=y4,  color=[0.9, 0.9, 0.9], edgecolor =[0, 0, 0], hatch='/////', linewidth=0.5)
    ax.bar(x+space, y8, width, bottom=y7,  color=[0.9, 0.9, 0.9], edgecolor =[0, 0, 0], hatch='/////', linewidth=0.5)



    d_o = ax.bar(x-space, y1, width, label='Data (Original)', color=[0.5, 0.5, 0.5],  edgecolor =[0, 0, 0], linewidth=0.5)
    d_b = ax.bar(x,       y4, width, label='Data (BODP-4)', color=[0.7, 0.7, 0.7],  edgecolor =[0, 0, 0], linewidth=0.5)
    d_f = ax.bar(x+space, y7, width, label='Data (FGP-5X5)', color=[0.9, 0.9, 0.9],  edgecolor =[0, 0, 0], linewidth=0.5)

    ax.bar(x-space, y1, width, label='Data (Original)', color=[0.5, 0.5, 0.5],  edgecolor =[0.5, 0.5, 0.5], linewidth=0.5)
    ax.bar(x,       y4, width, label='Data (BODP-4)', color=[0.7, 0.7, 0.7],  edgecolor =[0.7, 0.7, 0.7], linewidth=0.5)
    ax.bar(x+space, y7, width, label='Data (FGP-5X5)', color=[0.9, 0.9, 0.9],  edgecolor =[0.9, 0.9, 0.9], linewidth=0.5)


    other = ax.bar(x-space, y3, width, bottom=[sum(yy) for yy in zip(y1, y2)], label='Other', color= [0.1, 0.1, 0.1], edgecolor =[0, 0, 0], linewidth=0.5)
    ax.bar(x, y6, width, bottom=[sum(yy) for yy in zip(y4, y5)],  color= [0.1, 0.1, 0.1], edgecolor =[0, 0, 0], linewidth=0.5)
    ax.bar(x+space, y9, width, bottom=[sum(yy) for yy in zip(y7, y8)],  color= [0.1, 0.1, 0.1], edgecolor =[0, 0, 0], linewidth=0.5)


    ax.bar(x-space, np.array(y1)+np.array(y2)+np.array(y3), width, fill = False,  edgecolor =[0, 0, 0], linewidth=0.5)
    ax.bar(x,       np.array(y4)+np.array(y5)+np.array(y6), width, fill = False,  edgecolor =[0, 0, 0], linewidth=0.5)
    ax.bar(x+space, np.array(y7)+np.array(y8)+np.array(y9), width, fill = False,  edgecolor =[0, 0, 0], linewidth=0.5)


    ax.set_xticks(x)
    layer_names_yolo_608 = ["conv1", "max1", "conv2", "max2",  "conv3", "conv4", "conv5", "max3",  "conv6", "conv7", "conv8", "max4",  "conv9", "conv10", "conv11", "conv12"]
    layer_names_tiny_yolo = ["conv1", "max1", "conv2", "max2", "conv3", "max3", "conv4", "max4", "conv5", "max5", "conv6", "max6", "conv7", "conv8", "conv9", "region"]
    ax.set_xticklabels(layer_names_yolo_608, rotation=0)
    #ax.set_xticklabels(layer_names_tiny_yolo, rotation=30)
    ax.set_xlim([-1,len(x)])
    #plt.legend(loc=9, ncol=5, bbox_to_anchor=(0.5, 1.1), framealpha=1)
    plt.figlegend((d_o[0], d_b[0], d_f[0],  weight[0], other[0]), ('Data (Original)','Data (BODP-4)', 'Data (FGP-5X5)', 'Weight', 'Other'), loc=9, ncol=5, bbox_to_anchor=(0.5, 1.01), framealpha=1)



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

    filename_list = ["profile/yolo608/5X5/steal/layer_input.log", 
			"profile/yolo608/5X5/steal/layer_output.log", 
			"profile/yolo608/5X5/steal/layer_output_ir.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1])) + np.array(load_data(filename_list[2]))
    y5X5 = other + max(y)
    filename_list = ["profile/yolo608/4X4/steal/layer_input.log", 
			"profile/yolo608/4X4/steal/layer_output.log", 
			"profile/yolo608/4X4/steal/layer_output_ir.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1])) + np.array(load_data(filename_list[2]))
    y4X4 = other + max(y)
    filename_list = ["profile/yolo608/3X3/steal/layer_input.log", 
			"profile/yolo608/3X3/steal/layer_output.log", 
			"profile/yolo608/3X3/steal/layer_output_ir.log"]
    y = np.array(load_data(filename_list[0])) + np.array(load_data(filename_list[1])) + np.array(load_data(filename_list[2]))
    y3X3 = other + max(y)



    x = np.arange(6)
    print x


    width = 0.2
    

    ax.bar(x-1.5*width, [y1, y2, y3, y4, y5, y6], width, label='BODP', color=[0.9, 0.9, 0.9],  edgecolor =[0,0,0])
    ax.bar(x - 0.5*width, [y3X3, y3X3, y3X3, y3X3, y3X3, y3X3], width, label='FGP-3X3', color=[0.7, 0.7, 0.7],  edgecolor =[0,0,0])
    ax.bar(x + 0.5*width, [y4X4, y4X4, y4X4, y4X4, y4X4, y4X4], width, label='FGP-4X4', color=[0.5, 0.5, 0.5],  edgecolor =[0,0,0])
    ax.bar(x + 1.5*width, [y5X5, y5X5, y5X5, y5X5, y5X5, y5X5], width, label='FGP-5X5', color=[0.3, 0.3, 0.3],  edgecolor =[0,0,0])


    ax.set_xticks(x)

    ax.set_xticklabels(['1', '2', '3', '4', '5', '6'],  fontsize=global_font_size)
    #ax.set_xticklabels(layer_names_tiny_yolo, rotation=30)
    ax.set_xlim([-.5,len(x)-0.5])
    plt.legend(loc=9, ncol=5, bbox_to_anchor=(0.5, 1.09), framealpha=1 , fontsize=global_font_size)

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
    axes.set_xlabel("Number of devices" , fontsize=global_font_size)
    plot_mem_total(axes, [""])
    #test_plot_bar4(axes, "./profile/tiny_yolo/layer_output.log", "./profile/tiny_yolo/layer_input.log", "./profile/tiny_yolo/layer_weight.log", "./profile/tiny_yolo/layer_other.log")
    fig.tight_layout()

    return fig


if __name__ == "__main__":




    #fig = plot_figure_mem_prof()
    #fig = plot_figure_time_prof()


    #fig = plot_figure_one_input_resource()
    #fig = plot_figure_multiple_input_resource()

    #fig = plot_figure_one_input_resource_2()
    fig = plot_figure_multiple_input_resource_2()
    #fig = plot_figure_multiple_input_resource_throughput_2()
    #fig = plot_figure_mem_footprint()
    #fig = plot_figure_mem_total()
    plt.show()

