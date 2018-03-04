#include "darknet_util.h"


//A table for partition ID
//A mapping of partition IDs
//int part_id[PARTITIONS_H][PARTITIONS_W] = {
//   {0,  1,  2},
//   {3,  4,  5},
//   {6,  7,  8} 
//};
int part_id[PARTITIONS_H][PARTITIONS_W];
//Partitioned DNN parameters 
sub_index input_ranges[PARTITIONS][STAGES];//Required input ranges for each layer
sub_index output_ranges[PARTITIONS][STAGES];//Corrrect output ranges for each layer
sub_index stage_input_range;
sub_index stage_output_range;
sub_index stage_output_partition_ranges[PARTITIONS];
float* part_data[PARTITIONS];


//For data reuse
sub_index reuse_input_ranges[PARTITIONS][STAGES];//Cropped output ranges without overlap for each layer
sub_index reuse_output_ranges[PARTITIONS][STAGES];//Cropped output ranges without overlap for each layer


input_dim dims[STAGES];
//Partition overlap
int overlaps[STAGES];
int output_overlap = 0;
ir_data ir_output[STAGES][PARTITIONS_H][PARTITIONS_W];


//For smart gateway
unsigned int recv_counters[IMG_NUM][CLI_NUM];
float* recv_data[IMG_NUM][CLI_NUM][PARTITIONS];
unsigned int frame_counters[CLI_NUM][PARTITIONS];

//For reuse data serialization and deserialization
int up[PARTITIONS][STAGES];
int left[PARTITIONS][STAGES];	
int right[PARTITIONS][STAGES];
int down[PARTITIONS][STAGES];
unsigned int ir_data_size[PARTITIONS];

int result_up[PARTITIONS][STAGES];
int result_left[PARTITIONS][STAGES];	
int result_right[PARTITIONS][STAGES];
int result_down[PARTITIONS][STAGES];
unsigned int result_ir_data_size[PARTITIONS];

float* reuse_part_data[PARTITIONS];
int coverage[PARTITIONS_H][PARTITIONS_W];


//Indicating whether a particular partition require intermediate data or not
int need_ir_data[PARTITIONS];



//Global variables for MapReduce-like task distribution 
ir_data ir_output_mr[STAGES][PARTITIONS_H][PARTITIONS_W];
sub_index input_ranges_mr[PARTITIONS][STAGES];      	  //Required input ranges for each layer
sub_index output_ranges_mr[PARTITIONS][STAGES];     	  //Corrrect output ranges for each layer
float* part_data_mr[PARTITIONS];
float* output_part_data_mr[PARTITIONS];

int up_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
int left_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
int right_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
int down_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
int corners_mr[4][PARTITIONS_H][PARTITIONS_W][STAGES];

int result_up_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
int result_left_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
int result_right_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
int result_down_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
int result_corners_mr[4][PARTITIONS_H][PARTITIONS_W][STAGES];


unsigned int result_ir_data_size_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
unsigned int req_ir_data_size_mr[PARTITIONS_H][PARTITIONS_W][STAGES];
float* recv_data_mr[IMG_NUM][PARTITIONS];


//For work sharing reference
int assigned_task_num[ACT_CLI];
int cur_client_task_num;

//Add some variable for communication and computation profiling
double commu_time = 0.0;
double comp_time = 0.0;


