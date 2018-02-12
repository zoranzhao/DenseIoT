//#include "darknet_dist.h"
#include "darknet_util.h"



void print_subindex(sub_index index){
    printf("[[%d, %d][%d],\n", index.w1, index.w2, (index.w));
    printf(" [%d, %d][%d]]\n", index.h1, index.h2, (index.h));
}





sub_index cal_new_range(int p_h, int p_w,  int i, sub_index output_ranges[][STAGES], sub_index required_range) {
    int p_id = part_id[p_h][p_w];
    int p_id_nearby;
    //printf("Required input range from next layer, before cropping\n");
    //print_subindex(required_range);
    sub_index crop_range = required_range;
    //Processing the block on the left
    if(p_w > 0) {
			p_id_nearby = part_id[p_h][p_w-1]; 
			crop_range.w1 = output_ranges[p_id_nearby][i].w2 + 1;
    }
    //Processing the block above
    if(p_h > 0) {
			p_id_nearby = part_id[p_h-1][p_w]; 
			crop_range.h1 = output_ranges[p_id_nearby][i].h2 + 1;
    }
    return crop_range;


}


void cal_reuse_overlap_range(int p_h, int p_w,  int i, sub_index output_ranges[][STAGES], sub_index required_range) {
    //printf("\nReusing in the output of layer %d... ...: \n", i);
    //printf("Partition %d, left %d, above %d\n", part_id[p_h][p_w], part_id[p_h][(p_w-1)>0?(p_w-1):0], part_id[(p_h-1)>0?(p_h-1):0][p_w]);
    int p_id = part_id[p_h][p_w];
    int p_id_nearby;


    //Processing the block on the left
    //printf("Existing output whose overlap can be reused in output of layer %d... ...: \n", i);
    if(p_w > 0) {
			p_id_nearby = part_id[p_h][p_w-1]; 
			ir_output[i][p_h][p_w-1].right_range.w1 = required_range.w1;
			ir_output[i][p_h][p_w-1].right_range.w2 = reuse_output_ranges[p_id_nearby][i].w2;
			ir_output[i][p_h][p_w-1].right_range.h1 = reuse_output_ranges[p_id_nearby][i].h1;
			ir_output[i][p_h][p_w-1].right_range.h2 = reuse_output_ranges[p_id_nearby][i].h2;
			ir_output[i][p_h][p_w-1].right_range.w = ir_output[i][p_h][p_w-1].right_range.w2 - ir_output[i][p_h][p_w-1].right_range.w1 + 1;
			ir_output[i][p_h][p_w-1].right_range.h = ir_output[i][p_h][p_w-1].right_range.h2 - ir_output[i][p_h][p_w-1].right_range.h1 + 1;
    }
    //Processing the block above
    if(p_h > 0) {
			p_id_nearby = part_id[p_h-1][p_w]; 
			ir_output[i][p_h-1][p_w].down_range.w1 = reuse_output_ranges[p_id_nearby][i].w1;
			ir_output[i][p_h-1][p_w].down_range.w2 = reuse_output_ranges[p_id_nearby][i].w2;
			ir_output[i][p_h-1][p_w].down_range.h1 = required_range.h1;
			ir_output[i][p_h-1][p_w].down_range.h2 = reuse_output_ranges[p_id_nearby][i].h2;
			ir_output[i][p_h-1][p_w].down_range.w = ir_output[i][p_h-1][p_w].down_range.w2 - ir_output[i][p_h-1][p_w].down_range.w1 + 1;
			ir_output[i][p_h-1][p_w].down_range.h = ir_output[i][p_h-1][p_w].down_range.h2 - ir_output[i][p_h-1][p_w].down_range.h1 + 1;
    }
    //Processing the block above
    if(p_h > 0 && p_w > 0) {
			p_id_nearby = part_id[p_h-1][p_w-1]; 
			ir_output[i][p_h-1][p_w-1].corner_range.w1 = required_range.w1;
			ir_output[i][p_h-1][p_w-1].corner_range.w2 = reuse_output_ranges[p_id_nearby][i].w2;
			ir_output[i][p_h-1][p_w-1].corner_range.h1 = required_range.h1;
			ir_output[i][p_h-1][p_w-1].corner_range.h2 = reuse_output_ranges[p_id_nearby][i].h2;
			ir_output[i][p_h-1][p_w-1].corner_range.w = ir_output[i][p_h-1][p_w-1].corner_range.w2 - ir_output[i][p_h-1][p_w-1].corner_range.w1 + 1;
			ir_output[i][p_h-1][p_w-1].corner_range.h = ir_output[i][p_h-1][p_w-1].corner_range.h2 - ir_output[i][p_h-1][p_w-1].corner_range.h1 + 1;
    }

    //printf("After cropping in output of layer %d... ...: \n", i);
    //print_subindex(crop_range);
    //return crop_range;
    //print_subindex(crop_range);
    //printf("Partition %d, %d, %d\n", part_id[p_w][p_h], part_id[(p_w-1)>0?(p_w-1):0][p_h], part_id[p_w][(p_h-1)>0?(p_h-1):0]);
}







