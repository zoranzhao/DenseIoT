#include "darknet_util.h"


//A table for partition ID
//A mapping of partition IDs
//int part_id[PARTITIONS_H_MAX][PARTITIONS_W_MAX] = {
//   {0,  1,  2},
//   {3,  4,  5},
//   {6,  7,  8} 
//};
int part_id[PARTITIONS_H_MAX][PARTITIONS_W_MAX];
//Partitioned DNN parameters 
sub_index input_ranges[PARTITIONS_MAX][STAGES];//Required input ranges for each layer
sub_index output_ranges[PARTITIONS_MAX][STAGES];//Corrrect output ranges for each layer
sub_index stage_input_range;
sub_index stage_output_range;
sub_index stage_output_partition_ranges[PARTITIONS_MAX];
float* part_data[PARTITIONS_MAX];


//For data reuse
sub_index reuse_input_ranges[PARTITIONS_MAX][STAGES];//Cropped output ranges without overlap for each layer
sub_index reuse_output_ranges[PARTITIONS_MAX][STAGES];//Cropped output ranges without overlap for each layer


input_dim dims[STAGES];
//Partition overlap
int overlaps[STAGES];
int output_overlap = 0;
ir_data ir_output[STAGES][PARTITIONS_H_MAX][PARTITIONS_W_MAX];
ir_data_gateway ir_output_gateway[STAGES][PARTITIONS_H_MAX][PARTITIONS_W_MAX];

//For smart gateway
unsigned int recv_counters[IMG_NUM][CLI_NUM];
float* recv_data[IMG_NUM][CLI_NUM][PARTITIONS_MAX];
unsigned int frame_counters[CLI_NUM][PARTITIONS_MAX];

//For reuse data serialization and deserialization
int up[PARTITIONS_MAX][STAGES];
int left[PARTITIONS_MAX][STAGES];	
int right[PARTITIONS_MAX][STAGES];
int down[PARTITIONS_MAX][STAGES];
unsigned int ir_data_size[PARTITIONS_MAX];

int result_up[PARTITIONS_MAX][STAGES];
int result_left[PARTITIONS_MAX][STAGES];	
int result_right[PARTITIONS_MAX][STAGES];
int result_down[PARTITIONS_MAX][STAGES];
unsigned int result_ir_data_size[PARTITIONS_MAX];

float* reuse_part_data[PARTITIONS_MAX];
int coverage[PARTITIONS_H_MAX][PARTITIONS_W_MAX];



//Indicating whether a particular partition require intermediate data or not
int need_ir_data[PARTITIONS_MAX];

//Global variables for MapReduce-like task distribution 
ir_data ir_output_mr[STAGES][PARTITIONS_H_MAX][PARTITIONS_W_MAX];
sub_index input_ranges_mr[PARTITIONS_MAX][STAGES];      	  //Required input ranges for each layer
sub_index output_ranges_mr[PARTITIONS_MAX][STAGES];     	  //Corrrect output ranges for each layer
float* part_data_mr[PARTITIONS_MAX];
float* output_part_data_mr[PARTITIONS_MAX];

int up_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
int left_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
int right_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
int down_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
int corners_mr[4][PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];

int result_up_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
int result_left_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
int result_right_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
int result_down_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
int result_corners_mr[4][PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];


unsigned int result_ir_data_size_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
unsigned int req_ir_data_size_mr[PARTITIONS_H_MAX][PARTITIONS_W_MAX][STAGES];
float* recv_data_mr[IMG_NUM][PARTITIONS_MAX];


//For work sharing reference
int assigned_task_num[CLI_NUM];
int cur_client_task_num;

//Add some variable for communication and computation profiling
double commu_time = 0.0;
double comp_time = 0.0;


//Using gateway to record IR data and send out data
int frame_coverage[IMG_NUM][CLI_NUM][PARTITIONS_H_MAX][PARTITIONS_W_MAX];
int local_frame_coverage[IMG_NUM][CLI_NUM][PARTITIONS_H_MAX][PARTITIONS_W_MAX];

unsigned int frame_ir_res_counters[CLI_NUM][PARTITIONS_MAX];
unsigned int frame_ir_req_counters[CLI_NUM][PARTITIONS_MAX];

unsigned int local_frame_counters[CLI_NUM][PARTITIONS_MAX];
unsigned int steal_frame_counters[CLI_NUM][PARTITIONS_MAX];
unsigned int remote_frame_counters[CLI_NUM][PARTITIONS_MAX];

int ACT_CLI;
int CUR_CLI;
int PARTITIONS;
int PARTITIONS_W;
int PARTITIONS_H;


