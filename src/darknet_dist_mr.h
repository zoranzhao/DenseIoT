#include "darknet_dist.h"
#include "serialization_mr.h"

#ifndef DARKNET_DIST_MR__H 
#define DARKNET_DIST_MR__H


void join_output_mr(int part, float* part_result, float* stage_out, int upto, network net){
    reshape_output(part_result, stage_out, (stage_output_range.w2-stage_output_range.w1 + 1), 
			(stage_output_range.h2-stage_output_range.h1 + 1), net.layers[upto].out_c, 
			output_ranges_mr[part][upto].w1, output_ranges_mr[part][upto].w2,
			output_ranges_mr[part][upto].h1, output_ranges_mr[part][upto].h2);
}




inline void layer_output_partition_mr(network net, int i){//partition the output of layer i
    int w = net.layers[i].out_w;
    int h = net.layers[i].out_h;
    int partition_w = PARTITIONS_W;
    int partition_h = PARTITIONS_H;

    int stride_w = ceil(((float)w)/((float)partition_w));    
    int start_w = 0;
    int end_w = stride_w - 1;

    int stride_h = ceil(((float)h)/((float)partition_h));    
    int start_h = 0;
    int end_h = stride_h - 1;

    for(int p_h = 0; p_h < partition_h; p_h++){
       start_w = 0;
       end_w = stride_w - 1;	 
       for(int p_w = 0; p_w < partition_w; p_w++){
	   output_ranges_mr[part_id[p_h][p_w]][i].w1 = start_w;
	   output_ranges_mr[part_id[p_h][p_w]][i].w2 = end_w;
	   output_ranges_mr[part_id[p_h][p_w]][i].h1 = start_h;
	   output_ranges_mr[part_id[p_h][p_w]][i].h2 = end_h;
	   output_ranges_mr[part_id[p_h][p_w]][i].h = end_h - start_h + 1;
	   output_ranges_mr[part_id[p_h][p_w]][i].w = end_w - start_w + 1;
	   start_w = end_w + 1;
	   if(p_w == (partition_w-2))
	       end_w = w - 1;
	   else
	       end_w = end_w + stride_w; 	 
       }
       start_h = end_h + 1;
       if(p_h == (partition_h-2))
	       end_h = h - 1;
       else
	       end_h = end_h + stride_h; 
    }
}


inline void cal_each_layer_partition_mr(network net, int startfrom, int upto){
    //Calculate the output partitions for each layer within the range
    for(int i = startfrom; i < upto+1; i++){
	layer_output_partition_mr(net, i);
    }

    for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
	for(int p_w = 0; p_w < PARTITIONS_W; p_w++){ 
	   for(int i = upto; i >= startfrom; i--){
		//Calculate input_ranges	
		input_ranges_mr[part_id[p_h][p_w]][i] = calculate_range(output_ranges_mr[part_id[p_h][p_w]][i], net.layers[i]);
	   }
	}
    }

}


void cal_reuse_overlap_range_mr(int p_h, int p_w,  int i, sub_index output_ranges_mr[][STAGES], sub_index required_range) {
    //printf("Reusing in the output of layer %d... ...: \n", i);
    //std::cout << "Partition number is: "<< part_id[p_h][p_w] << std::endl;
    //printf("Partition %d, left %d, above %d\n", part_id[p_h][p_w], part_id[p_h][(p_w-1)>0?(p_w-1):0], part_id[(p_h-1)>0?(p_h-1):0][p_w]);
    int p_id = part_id[p_h][p_w];
    int p_id_nearby;

    //Processing the block on the left
    //printf("Existing output whose overlap can be reused in output of layer %d... ...: \n", i);
    if(p_w > 0) {
			p_id_nearby = part_id[p_h][p_w-1]; 
			ir_output_mr[i][p_h][p_w-1].right_range.w1 = required_range.w1;
			ir_output_mr[i][p_h][p_w-1].right_range.w2 = output_ranges_mr[p_id_nearby][i].w2;
			ir_output_mr[i][p_h][p_w-1].right_range.h1 = output_ranges_mr[p_id_nearby][i].h1;
			ir_output_mr[i][p_h][p_w-1].right_range.h2 = output_ranges_mr[p_id_nearby][i].h2;
			ir_output_mr[i][p_h][p_w-1].right_range.w = ir_output_mr[i][p_h][p_w-1].right_range.w2 - ir_output_mr[i][p_h][p_w-1].right_range.w1 + 1;
			ir_output_mr[i][p_h][p_w-1].right_range.h = ir_output_mr[i][p_h][p_w-1].right_range.h2 - ir_output_mr[i][p_h][p_w-1].right_range.h1 + 1;
			//std::cout << "Left block ..." << std::endl;
			//print_subindex(output_ranges_mr[p_id_nearby][i]);
			//std::cout << "Cur block ..." << std::endl;
			//print_subindex(output_ranges_mr[part_id[p_h][p_w]][i]);
			//std::cout << "Required block is ..." << std::endl;
			//print_subindex(required_range);
			//print_subindex(ir_output_mr[i][p_h][p_w-1].right_range);


    }
    //Processing the block on the right
    if((p_w + 1) < PARTITIONS_W ) {
			p_id_nearby = part_id[p_h][p_w+1]; 
			ir_output_mr[i][p_h][p_w+1].left_range.w1 = output_ranges_mr[p_id_nearby][i].w1;
			ir_output_mr[i][p_h][p_w+1].left_range.w2 = required_range.w2;
			ir_output_mr[i][p_h][p_w+1].left_range.h1 = output_ranges_mr[p_id_nearby][i].h1;
			ir_output_mr[i][p_h][p_w+1].left_range.h2 = output_ranges_mr[p_id_nearby][i].h2;
			ir_output_mr[i][p_h][p_w+1].left_range.w = ir_output_mr[i][p_h][p_w+1].left_range.w2 - ir_output_mr[i][p_h][p_w+1].left_range.w1 + 1;
			ir_output_mr[i][p_h][p_w+1].left_range.h = ir_output_mr[i][p_h][p_w+1].left_range.h2 - ir_output_mr[i][p_h][p_w+1].left_range.h1 + 1;
			//std::cout << "Right block ..." << std::endl;
			//print_subindex(ir_output_mr[i][p_h][p_w+1].left_range);
    }

    //Processing the block above
    if(p_h > 0) {
			p_id_nearby = part_id[p_h-1][p_w]; 
			ir_output_mr[i][p_h-1][p_w].down_range.w1 = output_ranges_mr[p_id_nearby][i].w1;
			ir_output_mr[i][p_h-1][p_w].down_range.w2 = output_ranges_mr[p_id_nearby][i].w2;
			ir_output_mr[i][p_h-1][p_w].down_range.h1 = required_range.h1;
			ir_output_mr[i][p_h-1][p_w].down_range.h2 = output_ranges_mr[p_id_nearby][i].h2;
			ir_output_mr[i][p_h-1][p_w].down_range.w = ir_output_mr[i][p_h-1][p_w].down_range.w2 - ir_output_mr[i][p_h-1][p_w].down_range.w1 + 1;
			ir_output_mr[i][p_h-1][p_w].down_range.h = ir_output_mr[i][p_h-1][p_w].down_range.h2 - ir_output_mr[i][p_h-1][p_w].down_range.h1 + 1;
			//std::cout << "Up block ..." << std::endl;
			//print_subindex(ir_output_mr[i][p_h-1][p_w].down_range);
    }
    //Processing the block down
    if((p_h + 1) < PARTITIONS_H ) {
			p_id_nearby = part_id[p_h+1][p_w]; 
			ir_output_mr[i][p_h+1][p_w].up_range.w1 = output_ranges_mr[p_id_nearby][i].w1;
			ir_output_mr[i][p_h+1][p_w].up_range.w2 = output_ranges_mr[p_id_nearby][i].w2;
			ir_output_mr[i][p_h+1][p_w].up_range.h1 = output_ranges_mr[p_id_nearby][i].h1;
			ir_output_mr[i][p_h+1][p_w].up_range.h2 = required_range.h2;
			ir_output_mr[i][p_h+1][p_w].up_range.w = ir_output_mr[i][p_h+1][p_w].up_range.w2 - ir_output_mr[i][p_h+1][p_w].up_range.w1 + 1;
			ir_output_mr[i][p_h+1][p_w].up_range.h = ir_output_mr[i][p_h+1][p_w].up_range.h2 - ir_output_mr[i][p_h+1][p_w].up_range.h1 + 1;
			//std::cout << "Down block ..." << std::endl;
			//print_subindex(ir_output_mr[i][p_h+1][p_w].up_range);
    }

//left up corner
    if(p_w > 0 && p_h > 0) {
	p_id_nearby = part_id[p_h-1][p_w-1]; 
	ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].w1 = required_range.w1;
	ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].h1 = required_range.h1;
	ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].w2 = output_ranges_mr[p_id_nearby][i].w2;
	ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].h2 = output_ranges_mr[p_id_nearby][i].h2;
	ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].w = ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].w2 - ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].w1 + 1;
	ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].h = ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].h2 - ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].h1 + 1;
    }    
//right up corner
    if((p_w + 1) < PARTITIONS_W && p_h > 0) {
	p_id_nearby = part_id[p_h-1][p_w+1]; 
	ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].w1 = output_ranges_mr[p_id_nearby][i].w1;
	ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].h1 = required_range.h1;
	ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].w2 = required_range.w2;
	ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].h2 = output_ranges_mr[p_id_nearby][i].h2;
        ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].w = ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].w2 - ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].w1 + 1;
        ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].h = ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].h2 - ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].h1 + 1;
    }    
//left down corner
    if(p_w > 0 && (p_h + 1) < PARTITIONS_H) {
	p_id_nearby = part_id[p_h+1][p_w-1]; 
	ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].w1 = required_range.w1;
	ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].h1 = output_ranges_mr[p_id_nearby][i].h1;
	ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].w2 = output_ranges_mr[p_id_nearby][i].w2;
	ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].h2 = required_range.h2;
	ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].h = ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].h2 - ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].h1 + 1;
	ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].w = ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].w2 - ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].w1 + 1;
    }    
//right down corner
    if((p_w + 1) < PARTITIONS_W && (p_h + 1) < PARTITIONS_H) {
	p_id_nearby = part_id[p_h+1][p_w+1]; 
	ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].w1 = output_ranges_mr[p_id_nearby][i].w1;
	ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].h1 = output_ranges_mr[p_id_nearby][i].h1;
	ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].w2 = required_range.w2;
	ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].h2 = required_range.h2;
	ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].h = ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].h2 - ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].h1 + 1;
	ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].w = ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].w2 - ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].w1 + 1;
    }    



}


inline network reshape_network_mr(int startfrom, int upto, network net){
    int print_reshape_info = 0;
    numbering_part_id();
    cal_each_layer_partition_mr(net, startfrom, upto);

    if(print_reshape_info == 1){
	    for(int i = upto; i >= startfrom; i--){
		for(int p = 0; p < PARTITIONS; p++){
		    std::cout << "The input at layer: "<< i <<" ... :" << std::endl;
		    print_subindex(input_ranges_mr[p][i]);
		    std::cout << "The output at layer: "<< i <<" ... :" << std::endl;
		    print_subindex(output_ranges_mr[p][i]);
		}
	    }
    }
    for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
	for(int p_w = 0; p_w < PARTITIONS_W; p_w++){
	    for(int i = 0; i < STAGES; i++){
		ir_output_mr[i][p_h][p_w].down_range.w = 0;
		ir_output_mr[i][p_h][p_w].down_range.h = 0;
		ir_output_mr[i][p_h][p_w].right_range.w = 0;
		ir_output_mr[i][p_h][p_w].right_range.h = 0;
		ir_output_mr[i][p_h][p_w].up_range.w = 0;
		ir_output_mr[i][p_h][p_w].up_range.h = 0;
		ir_output_mr[i][p_h][p_w].left_range.w = 0;
		ir_output_mr[i][p_h][p_w].left_range.h = 0;
            }
        }
    }
    //Calculate reuse
    for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
	for(int p_w = 0; p_w < PARTITIONS_W; p_w++){ 
	    for(int i = upto; i > 0; i--){
		cal_reuse_overlap_range_mr(p_h, p_w, i-1, &output_ranges_mr[0], input_ranges_mr[part_id[p_h][p_w]][i] );
	    }
	}
    }

/*
    for(int i = upto-1; i >= startfrom; i--){
	std::cout << "==========================At layer: "<< i <<" ... :" << std::endl;
        for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
          for(int p_w = 0; p_w < PARTITIONS_W; p_w++){
	    std::cout << "Partition: "<< part_id[p_h][p_w] <<" ... :" << std::endl;

	    if(ir_output_mr[i][p_h][p_w].down_range.w>0&&ir_output_mr[i][p_h][p_w].down_range.h>0)
		    {std::cout << "Down" << std::endl;   print_subindex(ir_output_mr[i][p_h][p_w].down_range);}
	    if(ir_output_mr[i][p_h][p_w].right_range.w>0&&ir_output_mr[i][p_h][p_w].right_range.h>0)
		    {std::cout << "Right" << std::endl;  print_subindex(ir_output_mr[i][p_h][p_w].right_range);}
	    if(ir_output_mr[i][p_h][p_w].up_range.w>0&&ir_output_mr[i][p_h][p_w].up_range.h>0)
		    {std::cout << "Up" << std::endl;     print_subindex(ir_output_mr[i][p_h][p_w].up_range);}
	    if(ir_output_mr[i][p_h][p_w].left_range.w>0&&ir_output_mr[i][p_h][p_w].left_range.h>0)
		    {std::cout << "Left" << std::endl;   print_subindex(ir_output_mr[i][p_h][p_w].left_range);}

	    std::cout << "Printing corner ... ... " << std::endl; 
	    for(int ii = 0; ii < 4; ii++){
		if(ir_output_mr[i][p_h][p_w].corner_range_mr[ii].w > 0 && ir_output_mr[i][p_h][p_w].corner_range_mr[ii].h > 0){
			std::cout << "Corner ID is: "<< ii << std::endl;		    
			print_subindex(ir_output_mr[i][p_h][p_w].corner_range_mr[ii]);
		}
	    }
  	  }
	}
    }
*/

    stage_input_range.w1 = 0;
    stage_input_range.w2 = net.layers[startfrom].w-1;
    stage_input_range.w = net.layers[startfrom].w;
    stage_input_range.h1 = 0;
    stage_input_range.h2 = net.layers[startfrom].h-1;
    stage_input_range.h = net.layers[startfrom].h;


    stage_output_range.w1 = 0;
    stage_output_range.w2 = net.layers[upto].out_w-1;
    stage_output_range.w = net.layers[upto].out_w;
    stage_output_range.h1 = 0;
    stage_output_range.h2 = net.layers[upto].out_h-1;
    stage_output_range.h = net.layers[upto].out_h;

    cal_dependency_mr(net, startfrom, upto);
    result_cal_dependency_mr(net, startfrom, upto);

    return net;
}


void fork_input_mr(int startfrom, float* stage_in, network net){

    int part;
    //Prepare the input data for each partition   
    for(part = 0; part < PARTITIONS; part ++) { 
      	part_data_mr[part] = reshape_input(stage_in, stage_input_range.w, stage_input_range.h,  net.layers[startfrom].c, 
					input_ranges_mr[part][startfrom].w1, input_ranges_mr[part][startfrom].w2, 
					input_ranges_mr[part][startfrom].h1, input_ranges_mr[part][startfrom].h2);
	//printf("%2d %4d %4d %4d %4d %4d %4d\n", (stage_input_range.w2 - stage_input_range.w1 + 1), (stage_input_range.h2 - stage_input_range.h1 + 1), net.layers[startfrom].c, 
	  //input_ranges_mr[part][startfrom].w1, input_ranges_mr[part][startfrom].w2, input_ranges_mr[part][startfrom].h1, input_ranges_mr[part][startfrom].h2);
    }


}

//
inline network forward_stage_mr(int p_h, int p_w, float *input, int startfrom, int upto,  network net)
{
    int print_data_to_record = 0;
    int part = part_id[p_h][p_w];
    net.input = input;
    //Reshape first
    for(int i = startfrom; i < (upto+1); ++i){
	net.layers[i].h = (input_ranges_mr[part][i].h2 - input_ranges_mr[part][i].h1 + 1); net.layers[i].out_h = (net.layers[i].h/net.layers[i].stride); 
	net.layers[i].w = (input_ranges_mr[part][i].w2 - input_ranges_mr[part][i].w1 + 1); net.layers[i].out_w = (net.layers[i].w/net.layers[i].stride); 
	net.layers[i].outputs = net.layers[i].out_h * net.layers[i].out_w * net.layers[i].out_c; 
	net.layers[i].inputs = net.layers[i].h * net.layers[i].w * net.layers[i].c; 
    }


    for(int i = startfrom; i < upto+1; i++){
	net.layers[i].forward(net.layers[i], net);

	float * cropped_output;
	if(net.layers[i].type == CONVOLUTIONAL){
		//std::cout<< "We should crop the output of the conv layer first..." <<std::endl;
		layer l = net.layers[i]; 
		sub_index tmp = crop_ranges(input_ranges_mr[part][i], output_ranges_mr[part][i]);   
		cropped_output = reshape_input(net.layers[i].output, l.out_w, l.out_h, l.out_c,  tmp.w1, tmp.w2, tmp.h1, tmp.h2);
	}else{
		cropped_output = net.layers[i].output;
	}


	//What should we record for the current layer?
	if((ir_output_mr[i][p_h][p_w].down_range.w>0)&&(ir_output_mr[i][p_h][p_w].down_range.h>0)){
		sub_index down_index = ir_output_mr[i][p_h][p_w].down_range;
		down_index.w1 -= output_ranges_mr[part][i].w1;
		down_index.w2 -= output_ranges_mr[part][i].w1;
		down_index.h1 -= output_ranges_mr[part][i].h1;
		down_index.h2 -= output_ranges_mr[part][i].h1;
                if(print_data_to_record == 1){
		   std::cout << "Down: " << std::endl;
		   print_subindex(ir_output_mr[i][p_h][p_w].down_range);
		   print_subindex(down_index);
		}
		ir_output_mr[i][p_h][p_w].down = reshape_input(cropped_output, output_ranges_mr[part][i].w, output_ranges_mr[part][i].h, net.layers[i].out_c, 
							down_index.w1, down_index.w2, 
							down_index.h1, down_index.h2);

	}
	if((ir_output_mr[i][p_h][p_w].right_range.w>0)&&(ir_output_mr[i][p_h][p_w].right_range.h>0)){
		sub_index right_index = ir_output_mr[i][p_h][p_w].right_range;
		right_index.w1 -= output_ranges_mr[part][i].w1;
		right_index.w2 -= output_ranges_mr[part][i].w1;
		right_index.h1 -= output_ranges_mr[part][i].h1;
		right_index.h2 -= output_ranges_mr[part][i].h1;
                if(print_data_to_record == 1){
		   std::cout << "Right: " << std::endl;
		   print_subindex(ir_output_mr[i][p_h][p_w].right_range);
	  	   print_subindex(right_index);
		}
		ir_output_mr[i][p_h][p_w].right =  reshape_input(cropped_output, output_ranges_mr[part][i].w, output_ranges_mr[part][i].h, net.layers[i].out_c, 
							right_index.w1, right_index.w2, 
							right_index.h1, right_index.h2);

	}
	//What should we record for the current layer?
	if((ir_output_mr[i][p_h][p_w].up_range.w>0)&&(ir_output_mr[i][p_h][p_w].up_range.h>0)){
		sub_index up_index = ir_output_mr[i][p_h][p_w].up_range;
		up_index.w1 -= output_ranges_mr[part][i].w1;
		up_index.w2 -= output_ranges_mr[part][i].w1;
		up_index.h1 -= output_ranges_mr[part][i].h1;
		up_index.h2 -= output_ranges_mr[part][i].h1;
                if(print_data_to_record == 1){
		   std::cout << "Up: " << std::endl;
		   print_subindex(ir_output_mr[i][p_h][p_w].up_range);
		   print_subindex(up_index);
		}
		ir_output_mr[i][p_h][p_w].up = reshape_input(cropped_output, output_ranges_mr[part][i].w, output_ranges_mr[part][i].h, net.layers[i].out_c, 
							up_index.w1, up_index.w2, 
							up_index.h1, up_index.h2);

	}
	if((ir_output_mr[i][p_h][p_w].left_range.w>0)&&(ir_output_mr[i][p_h][p_w].left_range.h>0)){
		sub_index left_index = ir_output_mr[i][p_h][p_w].left_range;
		left_index.w1 -= output_ranges_mr[part][i].w1;
		left_index.w2 -= output_ranges_mr[part][i].w1;
		left_index.h1 -= output_ranges_mr[part][i].h1;
		left_index.h2 -= output_ranges_mr[part][i].h1;
                if(print_data_to_record == 1){
		   std::cout << "Left: " << std::endl;
		   print_subindex(ir_output_mr[i][p_h][p_w].left_range);
		   print_subindex(left_index);
		}
		ir_output_mr[i][p_h][p_w].left = reshape_input(cropped_output, output_ranges_mr[part][i].w, output_ranges_mr[part][i].h, net.layers[i].out_c, 
							left_index.w1, left_index.w2, 
							left_index.h1, left_index.h2);

	}

        for(int corner_number = 0; corner_number < 4; corner_number ++){
	        if(ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].w > 0 && ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].h > 0 ){
		    sub_index corner_index = ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number];
		    corner_index.w1 -= output_ranges_mr[part][i].w1;
		    corner_index.w2 -= output_ranges_mr[part][i].w1;
		    corner_index.h1 -= output_ranges_mr[part][i].h1;
		    corner_index.h2 -= output_ranges_mr[part][i].h1;
                    if(print_data_to_record == 1){	    
		       std::cout << "Corner: " << corner_number << std::endl;
		       print_subindex(ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number]);
		       print_subindex(corner_index);
		    }
		    ir_output_mr[i][p_h][p_w].corner_mr[corner_number] = 
				reshape_input(cropped_output, output_ranges_mr[part][i].w, output_ranges_mr[part][i].h, net.layers[i].out_c, 
							corner_index.w1, corner_index.w2, 
							corner_index.h1, corner_index.h2);
		}
	}
     
    }


    return net; 
}



void cross_map_overlap_output(network net, int part, int layer_id){//Prepare the input for part_id in layer i, from output of adj partitions in layer i-1
        //part_data_mr[part_id];
	int print_preparin_the_next_layer = 0;
        int p_h = part / PARTITIONS_W; 
        int p_w = part % PARTITIONS_W;
        int cur_layer = layer_id;
        int prev_layer = layer_id-1;

	float * cropped_output;
	if(net.layers[prev_layer].type == CONVOLUTIONAL){
		layer l = net.layers[prev_layer]; 
		sub_index tmp = crop_ranges(input_ranges_mr[part][prev_layer], output_ranges_mr[part][prev_layer]); 
		if(print_preparin_the_next_layer == 1)  {
		       std::cout << "Processing the input data for partition " << part<< " at layer "<< layer_id << std::endl;
			std::cout<< "We should crop the output of the conv layer first..." <<std::endl;
			print_subindex(tmp);
			std::cout << "l.out_w: " << input_ranges_mr[part][prev_layer].w<< ", l.out_h: " 
				<< input_ranges_mr[part][prev_layer].h<< ", l.out_c: " << l.out_c <<std::endl;
		}
		cropped_output = reshape_input(output_part_data_mr[part], input_ranges_mr[part][prev_layer].w, 
						input_ranges_mr[part][prev_layer].h, l.out_c,  tmp.w1, tmp.w2, tmp.h1, tmp.h2);
	}else{
		cropped_output = output_part_data_mr[part];
	}	


	//Here, we should also consider the scenario where output of previous layer doesn't fully fit into the input of current layer
	sub_index alignment;
        if(output_ranges_mr[part_id[p_h][p_w]][prev_layer].w1 < input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1)
		alignment.w1 = input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
	else
		alignment.w1 = output_ranges_mr[part_id[p_h][p_w]][prev_layer].w1;
        if(output_ranges_mr[part_id[p_h][p_w]][prev_layer].h1 < input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1)
		alignment.h1 = input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1;
	else
		alignment.h1 = output_ranges_mr[part_id[p_h][p_w]][prev_layer].h1;

	alignment.w2 = output_ranges_mr[part_id[p_h][p_w]][prev_layer].w2;
	alignment.h2 = output_ranges_mr[part_id[p_h][p_w]][prev_layer].h2;
	alignment.w = alignment.w2 - alignment.w1 + 1; 
	alignment.h = alignment.h2 - alignment.h1 + 1;  


	sub_index tmp = crop_ranges(output_ranges_mr[part_id[p_h][p_w]][prev_layer], alignment); 
	cropped_output = reshape_input(cropped_output, output_ranges_mr[part_id[p_h][p_w]][prev_layer].w, 
						output_ranges_mr[part_id[p_h][p_w]][prev_layer].h, net.layers[prev_layer].out_c,  tmp.w1, tmp.w2, tmp.h1, tmp.h2);


	sub_index main_index;
	main_index.w2 = alignment.w2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
	main_index.h2 = alignment.h2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1;
	main_index.w1 = alignment.w1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1; 
	main_index.h1 = alignment.h1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1;  
	main_index.w = main_index.w2 - main_index.w1 + 1;
	main_index.h = main_index.h2 - main_index.h1 + 1;
	if(print_preparin_the_next_layer == 1)  {
		std::cout << "Main index is ... ... :" << std::endl;
		print_subindex(main_index);
		std::cout << "Output_ranges_mr is ... ... :" << std::endl;
		print_subindex(output_ranges_mr[part_id[p_h][p_w]][prev_layer]);
		print_subindex(alignment);
		std::cout << "Input_ranges_mr is ... ... :" << std::endl;
		print_subindex(input_ranges_mr[part_id[p_h][p_w]][cur_layer]);
	}
	part_data_mr[part]= 
		(float*)malloc(input_ranges_mr[part_id[p_h][p_w]][cur_layer].w*input_ranges_mr[part_id[p_h][p_w]][cur_layer].h*net.layers[prev_layer].out_c*sizeof(float));

	copy_input_to_output(cropped_output, part_data_mr[part], input_ranges_mr[part_id[p_h][p_w]][cur_layer].w, input_ranges_mr[part_id[p_h][p_w]][cur_layer].h, 
							net.layers[prev_layer].out_c, main_index.w1, main_index.w2, main_index.h1, main_index.h2);


	if(print_preparin_the_next_layer == 1)  std::cout<< "Copy padding regions from other partitions" <<std::endl;
	int reuse_h;
	int reuse_w;
	sub_index reuse_index;

	if(up_mr[p_h][p_w][cur_layer] == 1){
			reuse_h = p_h - 1; reuse_w = p_w;		
			reuse_index.w2 = ir_output_mr[prev_layer][reuse_h][reuse_w].down_range.w2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h2 = ir_output_mr[prev_layer][reuse_h][reuse_w].down_range.h2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1;
			reuse_index.w1 = ir_output_mr[prev_layer][reuse_h][reuse_w].down_range.w1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h1 = ir_output_mr[prev_layer][reuse_h][reuse_w].down_range.h1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1; 
			reuse_index.w = reuse_index.w2 - reuse_index.w1 + 1;
			reuse_index.h = reuse_index.h2 - reuse_index.h1 + 1;
			if(print_preparin_the_next_layer == 1)  {
		        	std::cout << "Up index is ... ... :" << std::endl;
				print_subindex(ir_output_mr[prev_layer][reuse_h][reuse_w].down_range);
				print_subindex(reuse_index);
			}
                        copy_input_to_output(ir_output_mr[prev_layer][reuse_h][reuse_w].down, part_data_mr[part], 
						input_ranges_mr[part_id[p_h][p_w]][cur_layer].w, input_ranges_mr[part_id[p_h][p_w]][cur_layer].h, 
						net.layers[prev_layer].out_c, reuse_index.w1, reuse_index.w2, reuse_index.h1, reuse_index.h2);
	}
	if(down_mr[p_h][p_w][cur_layer] == 1){
			reuse_h = p_h + 1; reuse_w = p_w;		
			reuse_index.w2 = ir_output_mr[prev_layer][reuse_h][reuse_w].up_range.w2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h2 = ir_output_mr[prev_layer][reuse_h][reuse_w].up_range.h2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1;
			reuse_index.w1 = ir_output_mr[prev_layer][reuse_h][reuse_w].up_range.w1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h1 = ir_output_mr[prev_layer][reuse_h][reuse_w].up_range.h1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1; 
			reuse_index.w = reuse_index.w2 - reuse_index.w1 + 1;
			reuse_index.h = reuse_index.h2 - reuse_index.h1 + 1;
			if(print_preparin_the_next_layer == 1)  {
				std::cout << "Down index is ... ... :" << std::endl;
				print_subindex(ir_output_mr[prev_layer][reuse_h][reuse_w].up_range);
				print_subindex(reuse_index);
			}
                        copy_input_to_output(ir_output_mr[prev_layer][reuse_h][reuse_w].up, part_data_mr[part], 
						input_ranges_mr[part_id[p_h][p_w]][cur_layer].w, input_ranges_mr[part_id[p_h][p_w]][cur_layer].h, 
						net.layers[prev_layer].out_c, reuse_index.w1, reuse_index.w2, reuse_index.h1, reuse_index.h2);
	}
	if(left_mr[p_h][p_w][cur_layer] == 1){
			reuse_h = p_h; reuse_w = p_w - 1;		
			reuse_index.w2 = ir_output_mr[prev_layer][reuse_h][reuse_w].right_range.w2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h2 = ir_output_mr[prev_layer][reuse_h][reuse_w].right_range.h2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1;
			reuse_index.w1 = ir_output_mr[prev_layer][reuse_h][reuse_w].right_range.w1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h1 = ir_output_mr[prev_layer][reuse_h][reuse_w].right_range.h1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1; 
			reuse_index.w = reuse_index.w2 - reuse_index.w1 + 1;
			reuse_index.h = reuse_index.h2 - reuse_index.h1 + 1;
			if(print_preparin_the_next_layer == 1)  {
				std::cout << "Left index is ... ... :" << std::endl;
				print_subindex(ir_output_mr[prev_layer][reuse_h][reuse_w].right_range);
				print_subindex(reuse_index);
			}
                        copy_input_to_output(ir_output_mr[prev_layer][reuse_h][reuse_w].right, part_data_mr[part], 
						input_ranges_mr[part_id[p_h][p_w]][cur_layer].w, input_ranges_mr[part_id[p_h][p_w]][cur_layer].h, 
						net.layers[prev_layer].out_c, reuse_index.w1, reuse_index.w2, reuse_index.h1, reuse_index.h2);
	}
	if(right_mr[p_h][p_w][cur_layer]  == 1){
			reuse_h = p_h; reuse_w = p_w + 1;		
			reuse_index.w2 = ir_output_mr[prev_layer][reuse_h][reuse_w].left_range.w2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h2 = ir_output_mr[prev_layer][reuse_h][reuse_w].left_range.h2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1;
			reuse_index.w1 = ir_output_mr[prev_layer][reuse_h][reuse_w].left_range.w1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h1 = ir_output_mr[prev_layer][reuse_h][reuse_w].left_range.h1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1; 
			reuse_index.w = reuse_index.w2 - reuse_index.w1 + 1;
			reuse_index.h = reuse_index.h2 - reuse_index.h1 + 1;
			if(print_preparin_the_next_layer == 1)  {
				std::cout << "Right index is ... ... :" << std::endl;
				print_subindex(ir_output_mr[prev_layer][reuse_h][reuse_w].left_range);
				print_subindex(reuse_index);
			}
                        copy_input_to_output(ir_output_mr[prev_layer][reuse_h][reuse_w].left, part_data_mr[part], 
						input_ranges_mr[part_id[p_h][p_w]][cur_layer].w, input_ranges_mr[part_id[p_h][p_w]][cur_layer].h, 
						net.layers[prev_layer].out_c, reuse_index.w1, reuse_index.w2, reuse_index.h1, reuse_index.h2);
	}
        for(int corner_num = 0; corner_num < 4; corner_num++){
		if(corners_mr[corner_num][p_h][p_w][cur_layer] == 1){
			if(corner_num == 0){reuse_h = p_h - 1; reuse_w = p_w - 1;}
			if(corner_num == 1){reuse_h = p_h - 1; reuse_w = p_w + 1;}
			if(corner_num == 2){reuse_h = p_h + 1; reuse_w = p_w - 1;}
			if(corner_num == 3){reuse_h = p_h + 1; reuse_w = p_w + 1;}					
			reuse_index.w2 = ir_output_mr[prev_layer][reuse_h][reuse_w].corner_range_mr[3-corner_num].w2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h2 = ir_output_mr[prev_layer][reuse_h][reuse_w].corner_range_mr[3-corner_num].h2 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1;
			reuse_index.w1 = ir_output_mr[prev_layer][reuse_h][reuse_w].corner_range_mr[3-corner_num].w1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].w1;
			reuse_index.h1 = ir_output_mr[prev_layer][reuse_h][reuse_w].corner_range_mr[3-corner_num].h1 - input_ranges_mr[part_id[p_h][p_w]][cur_layer].h1; 
			reuse_index.w = reuse_index.w2 - reuse_index.w1 + 1;
			reuse_index.h = reuse_index.h2 - reuse_index.h1 + 1;
			if(print_preparin_the_next_layer == 1)  {
				std::cout << "Corner index is ... ... :"<< corner_num << std::endl;
				print_subindex(ir_output_mr[prev_layer][reuse_h][reuse_w].corner_range_mr[3-corner_num]);
				print_subindex(reuse_index);
			}
                        copy_input_to_output(ir_output_mr[prev_layer][reuse_h][reuse_w].corner_mr[3-corner_num], part_data_mr[part], 
						input_ranges_mr[part_id[p_h][p_w]][cur_layer].w, input_ranges_mr[part_id[p_h][p_w]][cur_layer].h, 
						net.layers[prev_layer].out_c, reuse_index.w1, reuse_index.w2, reuse_index.h1, reuse_index.h2);
		}
	}

        //std::cout << "==========Finish=============: " << part_id[p_h][p_w] << std::endl;


}
    

#endif 

