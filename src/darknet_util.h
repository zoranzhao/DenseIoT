
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

//#include "distriot.h"
#include <iostream>


#define DEBUG_DIST 0

#ifndef DARKNET_UTIL__H
#define DARKNET_UTIL__H


//Calculate the input partition range
typedef struct partition_range{
    int w1;
    int w2;
    int h1;
    int h2;
    int h;
    int w;
} sub_index;


#define STAGES 8
#define PARTITIONS_W 3
#define PARTITIONS_H 3 
#define PARTITIONS 9
#define THREAD_NUM 1


typedef struct input_dimension{
    int w;
    int h;
} input_dim;

typedef struct overlapped_data{
   float *down;
   float *right;
   float *corner;
   bool cover;
   
   sub_index down_range;
   sub_index right_range;
   sub_index corner_range;
} ir_data;








//A table for partition ID
//A mapping of partition IDs
extern int part_id[PARTITIONS_H][PARTITIONS_W];

//Partitioned DNN parameters 
extern sub_index input_ranges[PARTITIONS][STAGES];//Required input ranges for each layer
extern sub_index output_ranges[PARTITIONS][STAGES];//Corrrect output ranges for each layer
extern sub_index stage_input_range;
extern sub_index stage_output_range;
extern sub_index stage_output_partition_ranges[PARTITIONS];
extern float* part_data[PARTITIONS];







//For data reuse

extern sub_index reuse_input_ranges[PARTITIONS][STAGES];//Cropped output ranges without overlap for each layer
extern sub_index reuse_output_ranges[PARTITIONS][STAGES];//Cropped output ranges without overlap for each layer


extern input_dim dims[STAGES];
//Partition overlap
extern int overlaps[STAGES];
extern int output_overlap;
extern ir_data ir_output[STAGES][PARTITIONS_H][PARTITIONS_W];




#endif





