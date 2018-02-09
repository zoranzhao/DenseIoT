#include "darknet_dist.h"

sub_index reuse_input_ranges[PARTITIONS][STAGES];//Cropped output ranges without overlap for each layer
sub_index reuse_output_ranges[PARTITIONS][STAGES];//Cropped output ranges without overlap for each layer


typedef struct input_dimension{
    int w;
    int h;
} input_dim;
input_dim dims[STAGES];
Partition overlap
int overlap[STAGES];
int output_overlap = 0;
    //overlap[upto] = calculate_overlap(output_overlap, net.layers[upto]);
    //for(i = upto-1; i >= 0; i--){
    //     layer l = net.layers[i];
    //     overlap[i] = calculate_overlap(overlap[i+1], l);
    //}
int calculate_overlap(int cur_overlap, layer l){
   int next_overlap;
   if(l.type == CONVOLUTIONAL){	
	next_overlap = cur_overlap*l.stride + l.size/2;     
   }else if(l.type == MAXPOOL){
	next_overlap = cur_overlap*l.stride;
   }
   return next_overlap;
}




typedef struct overlapped_data{
   float *down;
   float *right;
   
} ir_data;

ir_data ir_output[STAGES][PARTITIONS_H][PARTITIONS_W];


//void load_reuse_overlap_range(float *input, float *output, sub_index required_range, layer l, ir_data ir_output[])
//{
	
//}
//void save_reuse_overlap_range(float *input, sub_index required_range, int layer, ir_data ir_output[]);



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



