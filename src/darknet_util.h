
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
#include <iostream>
#include "distriot.h"
#include "config.h"

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


typedef struct input_dimension{
    int w;
    int h;
} input_dim;

typedef struct overlapped_data{
   float *down;
   float *right;
   float *up;
   float *left;
   float *corner;
   float *corner_mr[4];
   
   sub_index down_range;
   sub_index right_range;
   sub_index left_range;
   sub_index up_range;

   sub_index corner_range_mr[4];
   sub_index corner_range;

} ir_data;


typedef struct gateway_overlapped_data{
   float *down[CLI_NUM];
   float *right[CLI_NUM];
   float *up[CLI_NUM];
   float *left[CLI_NUM];
   float *corner[CLI_NUM];
   float *corner_mr[4];
   
   sub_index down_range;
   sub_index right_range;
   sub_index left_range;
   sub_index up_range;

   sub_index corner_range_mr[4];
   sub_index corner_range;

} ir_data_gateway;

extern bool cover[PARTITIONS_H_MAX][PARTITIONS_W_MAX];


//A table for partition ID
//A mapping of partition IDs
extern int part_id[PARTITIONS_H_MAX][PARTITIONS_W_MAX];

//Partitioned DNN parameters 
extern sub_index input_ranges[PARTITIONS_MAX][STAGES];//Required input ranges for each layer
extern sub_index output_ranges[PARTITIONS_MAX][STAGES];//Corrrect output ranges for each layer
extern sub_index stage_input_range;
extern sub_index stage_output_range;
extern sub_index stage_output_partition_ranges[PARTITIONS_MAX];
extern float* part_data[PARTITIONS_MAX];

//For data reuse
extern sub_index reuse_input_ranges[PARTITIONS_MAX][STAGES];//Cropped output ranges without overlap for each layer
extern sub_index reuse_output_ranges[PARTITIONS_MAX][STAGES];//Cropped output ranges without overlap for each layer
extern input_dim dims[STAGES];
extern float* reuse_part_data[PARTITIONS_MAX];

//Partition overlap
extern int overlaps[STAGES];
extern int output_overlap;
extern ir_data ir_output[STAGES][PARTITIONS_H_MAX][PARTITIONS_W_MAX];

//For smart gateway
extern unsigned int recv_counters[IMG_NUM][CLI_NUM];
extern float* recv_data[IMG_NUM][CLI_NUM][PARTITIONS_MAX];
extern unsigned int frame_counters[CLI_NUM][PARTITIONS_MAX];

//For reuse data serialization and deserialization
extern int up[PARTITIONS_MAX][STAGES];
extern int left[PARTITIONS_MAX][STAGES];	
extern int right[PARTITIONS_MAX][STAGES];
extern int down[PARTITIONS_MAX][STAGES];
extern unsigned int ir_data_size[PARTITIONS_MAX];

extern int result_up[PARTITIONS_MAX][STAGES];
extern int result_left[PARTITIONS_MAX][STAGES];	
extern int result_right[PARTITIONS_MAX][STAGES];
extern int result_down[PARTITIONS_MAX][STAGES];
extern unsigned int result_ir_data_size[PARTITIONS_MAX];
extern int need_ir_data[PARTITIONS_MAX];
extern int coverage[PARTITIONS_H_MAX][PARTITIONS_W_MAX];

inline void stage_output_partition(int w1, int w2, int h1, int h2);
sub_index calculate_range(sub_index output, layer l);
sub_index calculate_layeroutput_range(sub_index input, layer l);
sub_index crop_ranges(sub_index large, sub_index small);
float* reshape_input(float* input, int w, int h, int c, int dw1, int dw2, int dh1, int dh2);
void reshape_output(float* input, float* output, int w, int h, int c, int dw1, int dw2, int dh1, int dh2);
void copy_input_to_output(float* input, float* output, int w, int h, int c, int dw1, int dw2, int dh1, int dh2);
void numbering_part_id();
void clear_coverage();
bool is_part_ready(int part_id);
void print_subindex(sub_index index);

//Global variables for MapReduce-like task distribution
extern float* part_data_mr[PARTITIONS_MAX];
extern float* output_part_data_mr[PARTITIONS_MAX];
extern ir_data ir_output_mr[STAGES][PARTITIONS_H_MAX][PARTITIONS_W_MAX];
extern sub_index input_ranges_mr[PARTITIONS_MAX][STAGES];//Required input ranges for each layer
extern sub_index output_ranges_mr[PARTITIONS_MAX][STAGES];//Corrrect output ranges for each layer

extern int up_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern int left_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern int right_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern int down_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern int corners_mr[4][PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];

extern int result_up_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern int result_left_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern int result_right_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern int result_down_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern int result_corners_mr[4][PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];

extern unsigned int result_ir_data_size_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern unsigned int req_ir_data_size_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
extern float* recv_data_mr[IMG_NUM][PARTITIONS_MAX];

//Add some variable for communication and computation profiling
extern double commu_time;
extern double comp_time;
extern double time0;
extern double time1;


//Global variables for task sharing processing platform
extern int assigned_task_num[CLI_NUM];
extern int cur_client_task_num;

//Using gateway to record IR data and send out data
extern int frame_coverage[IMG_NUM][CLI_NUM][PARTITIONS_H_MAX][PARTITIONS_W_MAX];
extern int local_frame_coverage[IMG_NUM][CLI_NUM][PARTITIONS_H_MAX][PARTITIONS_W_MAX];
extern unsigned int frame_ir_res_counters[CLI_NUM][PARTITIONS_MAX];
extern unsigned int frame_ir_req_counters[CLI_NUM][PARTITIONS_MAX];
void clear_coverage_v2();
bool is_part_ready_v2(int part_id, int frame, int resource);
bool* get_local_coverage_v2(int part_id, int frame, int resource);
void set_coverage_v2(int part_id, int frame, int resource);

extern ir_data_gateway ir_output_gateway[STAGES][PARTITIONS_H_MAX][PARTITIONS_W_MAX];
extern unsigned int local_frame_counters[CLI_NUM][PARTITIONS_MAX];
extern unsigned int steal_frame_counters[CLI_NUM][PARTITIONS_MAX];
extern unsigned int remote_frame_counters[CLI_NUM][PARTITIONS_MAX];

#endif



