
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
} sub_index;

typedef struct input_dimension{
    int w;
    int h;
} input_dim;


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

/*
    //Partition overlap
    int overlap[STAGES];
    int output_overlap = 0;
    overlap[upto] = calculate_overlap(output_overlap, net.layers[upto]);
    for(i = upto-1; i >= 0; i--){
         layer l = net.layers[i];
         overlap[i] = calculate_overlap(overlap[i+1], l);
    }
*/
int calculate_overlap(int cur_overlap, layer l){
    int next_overlap;
    if(l.type == CONVOLUTIONAL){
	next_overlap = cur_overlap*l.stride + l.size/2; 
    }else if(l.type == MAXPOOL){
	next_overlap = cur_overlap*l.stride;
    }
    return next_overlap;
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


sub_index crop_range(sub_index large, sub_index small){

    sub_index output; 
    output.w1 = small.w1 - large.w1 ; 
    output.w2 = small.w1 - large.w1 + (small.w2 - small.w1);
    output.h1 = small.h1 - large.h1; 
    output.h2 = small.h1 - large.h1 + (small.h2 - small.h1);
    
    return output;
}



#define STAGES 4
#define PARTITIONS_W 2
#define PARTITIONS_H 2 
#define PARTITIONS 4
 
inline void forward_network_dist_prof_exe(network *netp)
{
    network net = *netp;
    int i;
    double t0 = 0;
    double t1 = 0;
    int ii;
    FILE *layerfile;

    //Number of partition assum it is even number

    int partition_w = PARTITIONS_W;
    int partition_h = PARTITIONS_H;

    int p_w;
    int p_h;

    int partition = partition_h*partition_w;
    int p;

    int upto = STAGES-1; //This stage execute upto this layer
     

      
    //input_dim dims[STAGES];
    sub_index input_ranges[PARTITIONS][STAGES];//input ranges for each layer
    sub_index output_ranges[PARTITIONS][STAGES];//input ranges for each layer
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

     
    layer l = net.layers[upto];
    //The required range in layers[upto].output




    for(p_h = 0; p_h < partition_h; p_h++){
	   for(p_w = 0; p_w < partition_w; p_w++){ 
	      sub_index stage_range;
	      stage_range.w1 = p_w*(l.out_w/partition_w);
	      stage_range.w2 = p_w*(l.out_w/partition_w) + l.out_w/partition_w - 1;

	      stage_range.h1 = p_h*(l.out_h/partition_h);
	      stage_range.h2 = p_h*(l.out_h/partition_h) + l.out_h/partition_h - 1;

	      sub_index tmp_range = stage_range;
	      for(i = upto; i >= 0; i--){
    		  layer l = net.layers[i];
		  tmp_range = calculate_range(tmp_range, l);
		  input_ranges[p_w+p_h*partition_h][i] = tmp_range;
    	          //print_subindex(input_ranges[p_w+p_h*partition_h][i]);
	      }

	      //printf("\n\n");

	   }
    }
    for(p = 0; p < partition; p++)
    	for(i = upto; i >= 0; i--)
	    output_ranges[p][i] = calculate_layeroutput_range(input_ranges[p][i], net.layers[i]);
	    

    //Prepare the input and output of the current stage;
    size_t stage_outs =  (net.layers[upto].out_w)*(net.layers[upto].out_h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
    float* stage_in = net.input;

    //Reshape the partitioned layers
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


    
    for(p=0; p < PARTITIONS; p++){
	for(i = 0; i < (upto+1); ++i){
    	    if(i == 0) { net.input = reshape_input(stage_in, (stage_input_range.w2 - stage_input_range.w1 + 1), (stage_input_range.h2 - stage_input_range.h1 + 1), net.layers[i].c, 
					input_ranges[p][i].w1, input_ranges[p][i].w2, input_ranges[p][i].h1, input_ranges[p][i].h2);
			 printf("%2d %4d %4d %4d %4d %4d %4d\n", (stage_input_range.w2 - stage_input_range.w1 + 1), (stage_input_range.h2 - stage_input_range.h1 + 1), net.layers[i].c, 
					input_ranges[p][i].w1, input_ranges[p][i].w2, input_ranges[p][i].h1, input_ranges[p][i].h2);
	    }
	    //Assume the convolutional stride is 1
	    else if(net.layers[i-1].type == CONVOLUTIONAL){
		layer l = net.layers[i-1];
		//printf("layer %d\n", i);
	        //print_subindex(input_ranges[p][i-1]);
	        //print_subindex(output_ranges[p][i-1]);
		//printf("%2d, %2d, %2d\n", l.out_w, l.out_h, l.out_c);
		//print_subindex(crop_range(input_ranges[p][i-1], output_ranges[p][i-1]));   
		sub_index tmp = crop_range(input_ranges[p][i-1], output_ranges[p][i-1]);              
		net.input = reshape_input(net.input, l.out_w, l.out_h, l.out_c,  tmp.w1, tmp.w2, tmp.h1, tmp.h2);
	        //printf("\n\n");

	    }
	    net.layers[i].forward(net.layers[i], net);
	    net.input = net.layers[i].output;  //Layer output

	    if(net.layers[i].truth) {
		    net.truth = net.layers[i].output;
	    }
	}
	//print_subindex(output_ranges[p][upto]);
        if(p==1) print_array("tmp.txt", net.layers[upto].output, net.layers[upto].outputs, net.layers[upto].out_w);
	reshape_output(net.layers[upto].output, stage_out, (stage_output_range.w2-stage_output_range.w1 + 1), 
			(stage_output_range.h2-stage_output_range.h1 + 1), net.layers[upto].out_c, 
			output_ranges[p][upto].w1, output_ranges[p][upto].w2,
			output_ranges[p][upto].h1, output_ranges[p][upto].h2);
    }


    //print_array("tmp.txt",stage_out, stage_outs, (stage_output_range.w2 - stage_output_range.w1 + 1));

/*
    size_t stage_outs =  (net.layers[upto].out_w)*(net.layers[upto].out_h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
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
*/
    
/*
    size_t stage_outs =  (net.layers[upto].out_w)*(net.layers[upto].out_h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
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
*/
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
