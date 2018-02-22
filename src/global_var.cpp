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
unsigned int recv_counters[CLI_NUM] = {0,0,0,0,0,0};
float* recv_data[CLI_NUM][PARTITIONS];


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



