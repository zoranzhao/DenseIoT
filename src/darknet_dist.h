
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

#define DEBUG_DIST 0

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

//input(w*h) [dh1, dh2]    copy into ==> output  [0, dh2 - dh1]
//	     [dw1, dw2]			         [0, dw2 - dw1]

float* reshape_input(float* input, int w, int h, int c, int dw1, int dw2, int dh1, int dh2){

   int out_w = dw2 - dw1 + 1;
   int out_h = dh2 - dh1 + 1;
   int i,j,k;
   float* output = (float*) malloc( sizeof(float)*out_w*out_h*c );  
   for(k = 0; k < c; ++k){
     for(j = dh1; j < dh2+1; ++j){
       for(i = dw1; i < dw2+1; ++i){
           int in_index  = i + w*(j + h*k);
           int out_index  = (i - dw1) + out_w*(j - dh1) + out_w*out_h*k;
	   output[out_index] = input[in_index];
       }
     }
   }
   return output;

}


//input [0, dh2 - dh1]    copy into ==> output(w*h)   [dh1, dh2]
//	[0, dw2 - dw1]			              [dw1, dw2]

void reshape_output(float* input, float* output, int w, int h, int c, int dw1, int dw2, int dh1, int dh2){

   int in_w = dw2 - dw1 + 1;
   int in_h = dh2 - dh1 + 1;
   int i,j,k;

   for(k = 0; k < c; ++k){
     for(j = 0; j < in_h; ++j){
       for(i = 0; i < in_w; ++i){
           int in_index  = i + in_w*(j + in_h*k);
           int out_index  = (i + dw1) + w*(j + dh1) + w*h*k;
	   output[out_index] = input[in_index];
       }
     }
   }

}


void print_array(char* filename, float* stage_out, int stage_outs, int line){
    FILE *layerfile;
    char layerdata[30];
    int ii;
    sprintf(layerdata, filename);
    layerfile = fopen(layerdata, "w");  
    for(ii = 0; ii < stage_outs; ii++){
       fprintf(layerfile, "%.1f", stage_out[ii]);
       if((ii+1)%line == 0) fprintf(layerfile, "\n");
    }
    fclose(layerfile);
}


inline void forward_network_dist_prof_exe(network *netp)
{
    network net = *netp;
    int i;
    double t0 = 0;
    double t1 = 0;
    int ii;
    FILE *layerfile;


    //Number of partition
    int partition=2;
    int p;
    //Last layer of the partition
    int upto = 3;
    //A memory space to store the output of the fusion block
    size_t stage_outs =  (net.layers[upto].out_w)*(net.layers[upto].out_h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
    //A memory space to hold the input of the fusion block
    float* stage_in = net.input;
    for(p=0; p < partition; p++){
	for(i = 0; i < (upto+1); ++i){//Iteratively execute the layers
		net.index = i;
		layer l = net.layers[i];
		if(net.layers[i].delta){	       
		    fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
		}

		if(i==0&&p==0){
			//print_array("1.txt", net.input, net.layers[i].inputs, net.layers[i].w);printf("====%d=%d===\n",net.layers[i].inputs, net.layers[i].w);
			net.layers[i].h = 307; net.layers[i].out_h = 307; 
			net.layers[i].outputs = net.layers[i].out_h * l.out_w * l.out_c; 
			net.layers[i].inputs = net.layers[i].h * l.w * l.c; 
			net.input = reshape_input(stage_in, 608, 608, 3, 0, 607, 0, 306);
			//print_array("2.txt", net.input, net.layers[i].inputs, net.layers[i].w);printf("====%d=%d===\n",net.layers[i].inputs, net.layers[i].w);
		}

		if(i==0&&p==1){
			net.layers[i].h = 307; net.layers[i].out_h = 307; 
			net.layers[i].outputs = net.layers[i].out_h * l.out_w * l.out_c; 
			net.layers[i].inputs = net.layers[i].h * l.w * l.c; 
			net.input = reshape_input(stage_in, 608, 608, 3, 0, 607, 301, 607);
		}

		if(i==1){net.layers[i].h = 306; net.layers[i].out_h = 153; 
			 net.layers[i].outputs = net.layers[i].out_h * l.out_w * l.out_c; 
			 net.layers[i].inputs = l.w * net.layers[i].h * l.c; 
			 net.input = reshape_input(net.input, 608, 307, 32, 0, 607, 0+p, 305+p);
		}

		if(i==2){net.layers[i].h = 153; net.layers[i].out_h = 153; 
			 net.layers[i].outputs = net.layers[i].out_h * l.out_w * l.out_c; 
			 net.layers[i].inputs = l.w * net.layers[i].h * l.c;  
		}

		if(i==3){net.layers[i].h = 152; net.layers[i].out_h = 76; 
			 net.layers[i].outputs = net.layers[i].out_h * l.out_w * l.out_c; 
			 net.layers[i].inputs = l.w * net.layers[i].h * l.c;
			 net.input = reshape_input(net.input, 304, 153, 64, 0, 303, 0+p, 151+p);		
		}

		//printf("Index %d, Layer %s, input data byte num is: %ld, output data byte num is: %ld\n", 
		//		i, get_layer_string(net.layers[i].type), net.layers[i].inputs*sizeof(float), net.layers[i].outputs*sizeof(float));

		net.layers[i].forward(net.layers[i], net);
		if(i==0){free(net.input);}
		net.input = net.layers[i].output;  //Layer output

		if(i==3&&p==0){reshape_output(net.layers[i].output, stage_out, 152, 152, 64, 0, 151, 0, 75);}
		if(i==3&&p==1){reshape_output(net.layers[i].output, stage_out, 152, 152, 64, 0, 151, 76, 151);}
		if(net.layers[i].truth) {
		    net.truth = net.layers[i].output;
		}
		//printf("Index %d, Layer %s, input data byte num is: %ld, output data byte num is: %ld\n", 
			//i, get_layer_string(net.layers[i].type), net.layers[i].inputs*sizeof(float), net.layers[i].outputs*sizeof(float));
	}

    }

    net.input = stage_out;

    //print_array("tmp.txt",stage_out, stage_outs, net.layers[upto].out_w);
    for(i = (upto+1); i <  net.n; ++i){//Iteratively execute the layers
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
    }

    free(stage_out);

    for(i = 0; i <  net.n; ++i){
		layer l = net.layers[i];
		if(i==0){
			net.layers[i].h = 608; net.layers[i].out_h = 608; 
			net.layers[i].outputs = net.layers[i].out_h * l.out_w * l.out_c; 
			net.layers[i].inputs = net.layers[i].h * l.w * l.c; 
		}
		if(i==1){net.layers[i].h = 608; net.layers[i].out_h = 304; 
			 net.layers[i].outputs = net.layers[i].out_h * l.out_w * l.out_c; 
			 net.layers[i].inputs = l.w * net.layers[i].h * l.c; 
		}
		if(i==2){net.layers[i].h = 304; net.layers[i].out_h = 304; 
			 net.layers[i].outputs = net.layers[i].out_h * l.out_w * l.out_c; 
			 net.layers[i].inputs = l.w * net.layers[i].h * l.c;  
		}
		if(i==3){net.layers[i].h = 304; net.layers[i].out_h = 152; 
			 net.layers[i].outputs = net.layers[i].out_h * l.out_w * l.out_c; 
			 net.layers[i].inputs = l.w * net.layers[i].h * l.c;
		}
    }

/*
    for(i = 0; i <  net.n; ++i){//Iteratively execute the layers
        t0 = what_time_is_it_now();
        net.index = i;
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }
        net.layers[i].forward(net.layers[i], net);



	if(0){
		layer l = net.layers[i];
		char layerdata[30];
		sprintf(layerdata, "layer%d_output.txt",i);
		layerfile = fopen(layerdata, "w");  
		for(ii = 0; ii < l.outputs; ii++){
		    fprintf(layerfile, "%.1f", l.output[ii] );
		    if((ii+1)%l.out_w == 0) fprintf(layerfile, "\n");
		}
		fclose(layerfile);
	}

        net.input = net.layers[i].output;  //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
        t1 = what_time_is_it_now();
        //printf("Index %d, Layer %s, input data byte num is: %ld, output data byte num is: %ld\n", 
		//i, get_layer_string(net.layers[i].type), net.layers[i].inputs*sizeof(float), net.layers[i].outputs*sizeof(float));

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
    //fclose(conv11);
    //fclose(conv33);
*/


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
