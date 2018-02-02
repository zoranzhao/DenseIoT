
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


//Calculate the input partition range
typedef struct partition_range{
    int w1;
    int w2;
    int h1;
    int h2;
    int h;
    int w;
} sub_index;

//typedef struct input_dimension{
//    int w;
//    int h;
//} input_dim;
//input_dim dims[STAGES];
    //Partition overlap
    //int overlap[STAGES];
    //int output_overlap = 0;
    //overlap[upto] = calculate_overlap(output_overlap, net.layers[upto]);
    //for(i = upto-1; i >= 0; i--){
    //     layer l = net.layers[i];
    //     overlap[i] = calculate_overlap(overlap[i+1], l);
    //}
//int calculate_overlap(int cur_overlap, layer l){
//    int next_overlap;
//    if(l.type == CONVOLUTIONAL){
//	next_overlap = cur_overlap*l.stride + l.size/2; 
//    }else if(l.type == MAXPOOL){
//	next_overlap = cur_overlap*l.stride;
//    }
//    return next_overlap;
//}


void print_subindex(sub_index index){
    printf("[[%d, %d][%d],\n", index.w1, index.w2, (index.w2 - index.w1 + 1));
    printf(" [%d, %d][%d]]\n", index.h1, index.h2, (index.h2 - index.h1 + 1));
}

sub_index calculate_range(sub_index output, layer l){
    sub_index input; 
    if(l.type == CONVOLUTIONAL){
       input.w1 = (output.w1*l.stride - l.size/2)>0   ? (output.w1*l.stride - l.size/2) : 0;
       input.w2 = (output.w2*l.stride + l.size/2)<(l.w-1) ? (output.w2*l.stride + l.size/2) : (l.w-1);
       input.h1 = (output.h1*l.stride - l.size/2)>0   ? (output.h1*l.stride - l.size/2) : 0;
       input.h2 = (output.h2*l.stride + l.size/2)<(l.h-1) ? (output.h2*l.stride + l.size/2) : (l.h-1);
    }else if(l.type == MAXPOOL){
       input.w1 = output.w1*l.stride;
       input.w2 = output.w2*l.stride + l.stride -1;
       input.h1 = output.h1*l.stride;
       input.h2 = output.h2*l.stride + l.stride -1;
    }
    return input;
}





sub_index calculate_layeroutput_range(sub_index input, layer l){
    sub_index output; 
    if(l.type == CONVOLUTIONAL){
	if((input.w1) > 0) {input.w1 = input.w1 + l.size/2;}
	if((input.w2) < l.w-1) {input.w2 = input.w2 - l.size/2;} 
	output.w1 = input.w1/l.stride; 
	output.w2 = input.w2/l.stride;
	if((input.h1) > 0) {input.h1 = input.h1 + l.size/2;}
	if((input.h2) < l.h-1) {input.h2 = input.h2 - l.size/2;} 
	output.h1 = input.h1/l.stride; 
	output.h2 = input.h2/l.stride;
    }else if(l.type == MAXPOOL){
	output.w1 = input.w1/l.stride; 
	output.w2 = input.w2/l.stride;
	output.h1 = input.h1/l.stride; 
	output.h2 = input.h2/l.stride;
    }
    return output;
}


sub_index crop_ranges(sub_index large, sub_index small){

    sub_index output; 
    output.w1 = small.w1 - large.w1 ; 
    output.w2 = small.w1 - large.w1 + (small.w2 - small.w1);
    output.h1 = small.h1 - large.h1; 
    output.h2 = small.h1 - large.h1 + (small.h2 - small.h1);
    
    return output;
}

/*
    0 conv     32  3 x 3 / 1   608 x 608 x   3   ->   608 x 608 x  32
    1 max          2 x 2 / 2   608 x 608 x  32   ->   304 x 304 x  32
    2 conv     64  3 x 3 / 1   304 x 304 x  32   ->   304 x 304 x  64
    3 max          2 x 2 / 2   304 x 304 x  64   ->   152 x 152 x  64
    4 conv    128  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x 128
    5 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64
    6 conv    128  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x 128
    7 max          2 x 2 / 2   152 x 152 x 128   ->    76 x  76 x 128
    8 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256
    9 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128
   10 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256
   11 max          2 x 2 / 2    76 x  76 x 256   ->    38 x  38 x 256
   12 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
   13 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256
   14 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512
   15 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256 
/////////////////////////////////////////////////////////////////////////
   16 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512 ~~
/////////////////////////////////////////////////////////////////////////
   17 max          2 x 2 / 2    38 x  38 x 512   ->    19 x  19 x 512
   18 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
   19 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
   20 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
   21 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512
   22 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024
   23 conv   1024  3 x 3 / 1    19 x  19 x1024   ->    19 x  19 x1024
   24 conv   1024  3 x 3 / 1    19 x  19 x1024   ->    19 x  19 x1024
   25 route  16
   26 conv     64  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x  64
   27 reorg              / 2    38 x  38 x  64   ->    19 x  19 x 256
   28 route  27 24
   29 conv   1024  3 x 3 / 1    19 x  19 x1280   ->    19 x  19 x1024
   30 conv    425  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 425
   31 detection
*/



#define STAGES 4
#define PARTITIONS_W 2
#define PARTITIONS_H 2 
#define PARTITIONS 4
 

typedef struct overlapped_data{
   float *up;
   float *down;
   float *left;
   float *right;
   
} ir_data;

ir_data ir_output[STAGES][PARTITIONS_H][PARTITIONS_W];


//void load_reuse_overlap_range(float *input, float *output, sub_index required_range, layer l, ir_data ir_output[])
//{
	
//}
//void save_reuse_overlap_range(float *input, sub_index required_range, int layer, ir_data ir_output[]);


//A table for partition ID
//A mapping of partition IDs
int part_id[PARTITIONS_H][PARTITIONS_W] = {
   {0, 1},
   {2, 3}
};



sub_index cal_reuse_overlap_range(int p_h, int p_w,  int i, sub_index output_ranges[][STAGES], sub_index required_range) {
    //printf("\nReusing in the output of layer %d... ...: \n", i);
    //printf("Partition %d, left %d, above %d\n", part_id[p_h][p_w], part_id[p_h][(p_w-1)>0?(p_w-1):0], part_id[(p_h-1)>0?(p_h-1):0][p_w]);
    int p_id = part_id[p_h][p_w];
    int p_id_nearby;
    //printf("Required input range from next layer, before cropping\n");
    //print_subindex(required_range);
    sub_index crop_range = required_range;
    //Processing the block on the left
    printf("Existing output whose overlap can be reused in output of layer %d... ...: \n", i);
    if(p_w > 0) {
			p_id_nearby = part_id[p_h][p_w-1]; 
			print_subindex(output_ranges[p_id_nearby][i]);
			crop_range.w1 = output_ranges[p_id_nearby][i].w2 + 1;
			printf("[[%d, %d],[%d, %d]]----\n",required_range.w1, crop_range.w1 - 1, crop_range.h1, crop_range.h2);
    }
    //Processing the block above
    if(p_h > 0) {
			p_id_nearby = part_id[p_h-1][p_w]; 
			print_subindex(output_ranges[p_id_nearby][i]);
			crop_range.h1 = output_ranges[p_id_nearby][i].h2 + 1;
			printf("[[%d, %d],[%d, %d]]----\n",crop_range.w1, crop_range.w2, required_range.h1, crop_range.h1 - 1);
    }
    printf("After cropping in output of layer %d... ...: \n", i);
    print_subindex(crop_range);
    return crop_range;
    //print_subindex(crop_range);
    //printf("Partition %d, %d, %d\n", part_id[p_w][p_h], part_id[(p_w-1)>0?(p_w-1):0][p_h], part_id[p_w][(p_h-1)>0?(p_h-1):0]);
}

inline void forward_network_dist_prof_exe(network *netp)
{
    network net = *netp;//Be careful because we are using a shallow copy here
    int i;
    double t0 = 0;
    double t1 = 0;
    int ii;
    FILE *layerfile;


    //Number of partition assume it is even number
    //Network parameters that are required for processing partition
    int partition_w = PARTITIONS_W;
    int partition_h = PARTITIONS_H;
    int partition = partition_h*partition_w;
    int upto = STAGES-1; //This stage execute upto this layer
     
    int p_w;
    int p_h;
    int p;


    //Network
    sub_index original_ranges[STAGES];//Corrrect output ranges for each layer
    sub_index input_ranges[PARTITIONS][STAGES];//Required input ranges for each layer
    sub_index output_ranges[PARTITIONS][STAGES];//Corrrect output ranges for each layer
    sub_index reuse_input_ranges[PARTITIONS][STAGES];//Cropped output ranges without overlap for each layer
    sub_index reuse_output_ranges[PARTITIONS][STAGES];//Cropped output ranges without overlap for each layer

    sub_index stage_input_range;
    stage_input_range.w1 = 0;
    stage_input_range.w2 = net.layers[0].w-1;
    stage_input_range.h1 = 0;
    stage_input_range.h2 = net.layers[0].h-1;
    sub_index stage_output_range;
    stage_output_range.w1 = 0;
    stage_output_range.w2 = net.layers[upto].out_w-1;
    stage_output_range.h1 = 0;
    stage_output_range.h2 = net.layers[upto].out_h-1;


    //Record the original ranges
    for(i = 0; i < upto+1; i++){
	original_ranges[i].w = net.layers[i].w;
	original_ranges[i].h = net.layers[i].h; 
	original_ranges[i].w1 = 0;
	original_ranges[i].w2 = net.layers[i].w - 1;
	original_ranges[i].h1 = 0; 
	original_ranges[i].h2 = net.layers[i].h - 1;
    }


    //Calculate the ranges after partition     
    layer l = net.layers[upto];

    t0 = what_time_is_it_now();
    //Assumption l.out_w and l.out_h are all even numbers here
    for(p_h = 0; p_h < partition_h; p_h++){
	   for(p_w = 0; p_w < partition_w; p_w++){ 
	      //printf("=========%d=========\n", p_w+p_h*partition_h);
	      sub_index stage_range;
	      stage_range.w1 = p_w*(l.out_w/partition_w);
	      stage_range.w2 = p_w*(l.out_w/partition_w) + l.out_w/partition_w - 1;
	      stage_range.h1 = p_h*(l.out_h/partition_h);
	      stage_range.h2 = p_h*(l.out_h/partition_h) + l.out_h/partition_h - 1;
	      sub_index tmp_range = stage_range;
    	      //print_subindex(stage_range);
	      for(i = upto; i >= 0; i--){
    		  layer l = net.layers[i];
		  tmp_range = calculate_range(tmp_range, l);
		  input_ranges[part_id[p_h][p_w]][i] = tmp_range;
    	          //print_subindex(input_ranges[p_w+p_h*partition_h][i]);
	      }
	      //printf("====================\n");
	   }
    }

    for(p = 0; p < partition; p++){
    	for(i = upto; i >= 0; i--){
	    output_ranges[p][i] = calculate_layeroutput_range(input_ranges[p][i], net.layers[i]);
	}
    }
/*
//--------------------------------------------------------------------------------------------------------------

    printf("After cropping in output of layer %d\n", upto);
    print_subindex(output_ranges[1][upto]);
    printf("Required range at input of layer %d\n", upto);
    print_subindex(input_ranges[1][upto]);

    reuse_output_ranges[part_id[0][1]][upto] = output_ranges[part_id[0][1]][upto];//Cropped output ranges without overlap for each layer
    reuse_input_ranges[part_id[0][1]][upto] = input_ranges[part_id[0][1]][upto];//Cropped output ranges without overlap for each layer


    //print_subindex(cal_reuse_overlap_range(0, 1, upto-1, &output_ranges[0], input_ranges[1][upto]));
    sub_index tmp = cal_reuse_overlap_range(0, 1, upto-1, &output_ranges[0], input_ranges[1][upto]);
    //print_subindex(tmp);
    tmp = calculate_range(tmp, net.layers[upto-1]);
    printf("Required range at input of layer %d\n", upto-1);
    print_subindex(tmp);
    reuse_output_ranges[part_id[0][1]][upto-1]  = tmp; 
    for(i = upto-1; i > 0; i--){
        //print_subindex(input_ranges[1][i]);
	//print_subindex(cal_reuse_overlap_range(0, 1, i-1, &output_ranges[0], output_ranges[part_id[0][1]][i]));
	//print_subindex(cal_reuse_overlap_range(0, 1, i-1, &output_ranges[0], tmp));
	tmp = cal_reuse_overlap_range(0, 1, i-1, &output_ranges[0], tmp);
        tmp = calculate_range(tmp, net.layers[i-1]);
        printf("Required range at input of layer %d\n", i-1);
        print_subindex(tmp);
        reuse_input_ranges[part_id[0][1]][i-1]  = tmp; 
    }
//--------------------------------------------------------------------------------------------------------------
*/
   

    //Prepare the input and output of the current stage;
    size_t stage_outs =  (net.layers[upto].out_w)*(net.layers[upto].out_h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
    float* stage_in = net.input;


    //Reshape the input data dims for partitioned layers
    for(p=0; p < PARTITIONS; p++){
	for(i = 0; i < (upto+1); ++i){
	    net.layers[i].h = (input_ranges[p][i].h2 - input_ranges[p][i].h1 + 1); net.layers[i].out_h = (net.layers[i].h/net.layers[i].stride); 
	    net.layers[i].w = (input_ranges[p][i].w2 - input_ranges[p][i].w1 + 1); net.layers[i].out_w = (net.layers[i].w/net.layers[i].stride); 
	    net.layers[i].outputs = net.layers[i].out_h * net.layers[i].out_w * net.layers[i].out_c; 
	    net.layers[i].inputs = net.layers[i].h * net.layers[i].w * net.layers[i].c; 
	    //layer l = net.layers[i];
	    //printf("conv   %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
	}
    }
    t1 = t1 + what_time_is_it_now() - t0;

    //Execute each layer in the network
    for(p=0; p < PARTITIONS; p++){
	for(i = 0; i < (upto+1); ++i){
            t0 = what_time_is_it_now();
	    //Prepare the input data for each partition   
    	    if(i == 0) { net.input = reshape_input(stage_in, (stage_input_range.w2 - stage_input_range.w1 + 1), (stage_input_range.h2 - stage_input_range.h1 + 1), net.layers[i].c, 
					input_ranges[p][i].w1, input_ranges[p][i].w2, input_ranges[p][i].h1, input_ranges[p][i].h2);
			 //printf("%2d %4d %4d %4d %4d %4d %4d\n", (stage_input_range.w2 - stage_input_range.w1 + 1), (stage_input_range.h2 - stage_input_range.h1 + 1), net.layers[i].c, 
			 //		input_ranges[p][i].w1, input_ranges[p][i].w2, input_ranges[p][i].h1, input_ranges[p][i].h2);
	    }


	    net.layers[i].forward(net.layers[i], net);


	    //Prepare the data for next layer   
	    if(net.layers[i].type == CONVOLUTIONAL){
		layer l = net.layers[i];
		//printf("layer %d\n", i);
	        //print_subindex(input_ranges[p][i]);
	        //print_subindex(output_ranges[p][i]);
		//printf("%2d, %2d, %2d\n", l.out_w, l.out_h, l.out_c);
		//print_subindex(crop_range(input_ranges[p][i], output_ranges[p][i]));   
		sub_index tmp = crop_ranges(input_ranges[p][i], output_ranges[p][i]);              
		net.input = reshape_input(net.layers[i].output, l.out_w, l.out_h, l.out_c,  tmp.w1, tmp.w2, tmp.h1, tmp.h2);
	        //printf("\n\n");

	    } else {net.input = net.layers[i].output;}  
	    t1 = t1 + what_time_is_it_now() - t0;

	    if(net.layers[i].truth) {
		    net.truth = net.layers[i].output;
	    }
	}
	//print_subindex(output_ranges[p][upto]);
        t0 = what_time_is_it_now();
	reshape_output(net.input, stage_out, (stage_output_range.w2-stage_output_range.w1 + 1), 
			(stage_output_range.h2-stage_output_range.h1 + 1), net.layers[upto].out_c, 
			output_ranges[p][upto].w1, output_ranges[p][upto].w2,
			output_ranges[p][upto].h1, output_ranges[p][upto].h2);
			//printf("%2d %4d %4d %4d %4d %4d %4d\n", (stage_output_range.w2-stage_output_range.w1 + 1), 
			//(stage_output_range.h2-stage_output_range.h1 + 1), net.layers[upto].out_c, 
			//output_ranges[p][upto].w1, output_ranges[p][upto].w2,
			//output_ranges[p][upto].h1, output_ranges[p][upto].h2);
	t1 = t1 + what_time_is_it_now() - t0;
    }

    //Recover the network
    for(i = 0; i < upto+1; ++i){
	layer l = net.layers[i];
	net.layers[i].h = original_ranges[i].h; net.layers[i].out_h = original_ranges[i].h/l.stride; 
	net.layers[i].w = original_ranges[i].w; net.layers[i].out_w = original_ranges[i].w/l.stride; 
	net.layers[i].outputs = net.layers[i].out_h * net.layers[i].out_w * l.out_c; 
	net.layers[i].inputs = net.layers[i].h * net.layers[i].w * l.c; 
    }


    net.layers[upto].output = stage_out;
    //print_array("tmp.txt",stage_out, stage_outs, (stage_output_range.w2 - stage_output_range.w1 + 1));
    net.input = stage_out;


    for(i = (upto+1); i <  net.n; ++i){//Iteratively execute the layers
        net.index = i;
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }
        net.layers[i].forward(net.layers[i], net);
        net.input = net.layers[i].output;  //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
    }





    printf("Time overhead is %f\n", t1); 
    



//Example code for profiling
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
    double t0 = what_time_is_it_now();
    double t1 = 0;

    FILE *layer_input;
    FILE *layer_output;
    FILE *layer_weight; 

    layer_input  = fopen("layer_input.log", "w"); 
    layer_output = fopen("layer_output.log", "w");  
    layer_weight = fopen("layer_weight.log", "w");

    for(i = 0; i < net.n; ++i){//Iteratively execute the layers
        net.index = i;
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }


        fprintf(layer_input, "%f\n", (float)(net.layers[i].inputs*sizeof(float))/1024.0/1024.0 );
        fprintf(layer_output, "%f\n", (float)(net.layers[i].outputs*sizeof(float))/1024.0/1024.0 );

        if(net.layers[i].type == CONNECTED)
           fprintf(layer_weight, "%f\n", (float)(net.layers[i].outputs*net.layers[i].inputs*sizeof(float))/1024.0/1024.0 );
	else
           fprintf(layer_weight, "%f\n", (float)(net.layers[i].nweights*sizeof(float))/1024.0/1024.0 );

 	//put_job(net.input, net.layers[i].inputs*sizeof(float), i);

        net.layers[i].forward(net.layers[i], net);
        net.input = net.layers[i].output;  //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
        printf("Index %d, Layer %s, input data byte num is: %ld, output data byte num is: %ld\n", 
		i, get_layer_string(net.layers[i].type), net.layers[i].inputs*sizeof(float), net.layers[i].outputs*sizeof(float));
        if(i==(STAGES-1))    printf("Time overhead is %f\n", (what_time_is_it_now() - t0) );
    }

    fclose(layer_input);
    fclose(layer_weight);
    fclose(layer_output);
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
    //forward_network_dist_test(net);
    float *out = net->output;
    *net = orig;
    return out;
}
