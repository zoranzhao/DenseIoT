
#include "darknet.h"

extern "C"{
#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "parser.h"
#include "data.h"
#include "option_list.h"
}

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "riot.h"



#define PORTNO 11111
#define SRV "10.145.85.169"

#define AP "192.168.42.1"

#define PINK0    "192.168.42.16"
#define BLUE0    "192.168.42.14"
#define ORANGE0  "192.168.42.15"

#define PINK1    "192.168.42.11"
#define BLUE1    "192.168.42.12"
#define ORANGE1  "192.168.42.13"

#define DEBUG_DIST 1

void write_layer_test(network *netp, int idx)
{
    network net = *netp;
    layer l = net.layers[idx];
    FILE *p_file;

    char filename[50];
    sprintf(filename, "%s_%d.dat", get_layer_string(net.layers[idx + 1].type), idx + 1);
    p_file = fopen(filename, "wb");
 
    fwrite(l.output, sizeof(float), l.outputs, p_file); 
    fclose(p_file);
}

void read_layer_test(network *netp, int idx)
{
    network net = *netp;
    layer l = net.layers[idx];
    FILE *p_file;

    char filename[50];
    sprintf(filename, "%s_%d.dat", get_layer_string(net.layers[idx].type), idx);
    p_file = fopen(filename, "rb");
    fread(net.input, sizeof(float), l.inputs, p_file); 
    fclose(p_file);

}

inline void forward_network_dist_prof_exe(network *netp)
{
    network net = *netp;
    int i;
    double t0 = 0;
    double t1 = 0;
    FILE *conv11;
    FILE *conv33;
    conv11 = fopen("conv11_data.log", "w");
    conv33 = fopen("conv33_data.log", "w");
    //data_file = fopen("layer_data_byte_num.log", "w");

    for(i = 0; i < net.n; ++i){//Iteratively execute the layers

        t0 = what_time_is_it_now();
        net.index = i;
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }
        net.layers[i].forward(net.layers[i], net);
        net.input = net.layers[i].output;  //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
        t1 = what_time_is_it_now();
        printf("Index %d, Layer %s, input data byte num is: %ld, output data byte num is: %ld\n", 
		i, get_layer_string(net.layers[i].type), net.layers[i].inputs*sizeof(float), net.layers[i].outputs*sizeof(float));

        //fprintf(data_file, "%ld\n", net.layers[i].inputs*sizeof(float) );
        //fprintf(time_file, "%lf\n", t1 - t0 );
	//if(net.layers[i].type==CONVOLUTIONAL){
	   //if(net.layers[i].size==1)
        	//fprintf(conv11, "%ld\n", net.layers[i].inputs*sizeof(float)  );
        	//fprintf(conv11, "%lf\n", t1 - t0 );
	   //if(net.layers[i].size==3)
        	//fprintf(conv33, "%ld\n", net.layers[i].inputs*sizeof(float)  );
        	//fprintf(conv33, "%lf\n", t1 - t0 );
	//}
    }

    fclose(conv11);
    fclose(conv33);
}

inline void forward_network_dist_test(network *netp)
{
    network net = *netp;
    int i;

    for(i = 0; i < net.n; ++i){//Iteratively execute the layers
        net.index = i;
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }

 	put_job(net.input, net.layers[i].inputs*sizeof(float), i);

        net.layers[i].forward(net.layers[i], net);
        net.input = net.layers[i].output;  //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
        printf("Index %d, Layer %s, input data byte num is: %ld, output data byte num is: %ld\n", 
		i, get_layer_string(net.layers[i].type), net.layers[i].inputs*sizeof(float), net.layers[i].outputs*sizeof(float));
    }

}

inline float *network_predict_dist_test(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network_dist_test(net);
    float *out = net->output;
    *net = orig;
    return out;
}


inline float *network_predict_dist_prof_exe(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network_dist_prof_exe(net);
    float *out = net->output;
    *net = orig;
    return out;
}
