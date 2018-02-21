//#include "darknet_dist.h"
#include "darknet_util.h"
#include "serialization.h"

#ifndef REUSE_DATA__H
#define REUSE_DATA__H

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


sub_index cal_new_range_full(int p_h, int p_w,  int i, sub_index output_ranges[][STAGES], sub_index required_range) {
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
    //Processing the block on the left
    if((p_w + 1) < PARTITIONS_W) {
			p_id_nearby = part_id[p_h][p_w+1]; 
			crop_range.w2 = output_ranges[p_id_nearby][i].w1 - 1;
    }
    //Processing the block below
    if((p_h + 1) < PARTITIONS_H) {
			p_id_nearby = part_id[p_h+1][p_w]; 
			crop_range.h2 = output_ranges[p_id_nearby][i].h1 - 1;
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

void cal_reuse_overlap_range_full(int p_h, int p_w,  int i, sub_index output_ranges[][STAGES], sub_index required_range) {
    //printf("Reusing in the output of layer %d... ...: \n", i);
    //printf("Partition %d, left %d, above %d\n", part_id[p_h][p_w], part_id[p_h][(p_w-1)>0?(p_w-1):0], part_id[(p_h-1)>0?(p_h-1):0][p_w]);
    int p_id = part_id[p_h][p_w];
    int p_id_nearby;

    //Processing the block on the left
    //printf("Existing output whose overlap can be reused in output of layer %d... ...: \n", i);
    if(p_w > 0) {
			p_id_nearby = part_id[p_h][p_w-1]; 
			ir_output[i][p_h][p_w-1].right_range.w1 = required_range.w1;
			ir_output[i][p_h][p_w-1].right_range.w2 = reuse_output_ranges[p_id_nearby][i].w2;
			ir_output[i][p_h][p_w-1].right_range.h1 = required_range.h1;
			ir_output[i][p_h][p_w-1].right_range.h2 = required_range.h2;
			ir_output[i][p_h][p_w-1].right_range.w = ir_output[i][p_h][p_w-1].right_range.w2 - ir_output[i][p_h][p_w-1].right_range.w1 + 1;
			ir_output[i][p_h][p_w-1].right_range.h = ir_output[i][p_h][p_w-1].right_range.h2 - ir_output[i][p_h][p_w-1].right_range.h1 + 1;
			//std::cout << "Left block ..." << std::endl;
			//print_subindex(ir_output[i][p_h][p_w-1].right_range);
    }
    //Processing the block on the right
    if((p_w + 1) < PARTITIONS_W ) {
			p_id_nearby = part_id[p_h][p_w+1]; 
			ir_output[i][p_h][p_w+1].left_range.w1 = reuse_output_ranges[p_id_nearby][i].w1;
			ir_output[i][p_h][p_w+1].left_range.w2 = required_range.w2;
			ir_output[i][p_h][p_w+1].left_range.h1 = required_range.h1;
			ir_output[i][p_h][p_w+1].left_range.h2 = required_range.h2;
			ir_output[i][p_h][p_w+1].left_range.w = ir_output[i][p_h][p_w+1].left_range.w2 - ir_output[i][p_h][p_w+1].left_range.w1 + 1;
			ir_output[i][p_h][p_w+1].left_range.h = ir_output[i][p_h][p_w+1].left_range.h2 - ir_output[i][p_h][p_w+1].left_range.h1 + 1;
			//std::cout << "Right block ..." << std::endl;
			//print_subindex(ir_output[i][p_h][p_w+1].left_range);
    }

    //Processing the block above
    if(p_h > 0) {
			p_id_nearby = part_id[p_h-1][p_w]; 
			ir_output[i][p_h-1][p_w].down_range.w1 = required_range.w1;
			ir_output[i][p_h-1][p_w].down_range.w2 = required_range.w2;
			ir_output[i][p_h-1][p_w].down_range.h1 = required_range.h1;
			ir_output[i][p_h-1][p_w].down_range.h2 = reuse_output_ranges[p_id_nearby][i].h2;
			ir_output[i][p_h-1][p_w].down_range.w = ir_output[i][p_h-1][p_w].down_range.w2 - ir_output[i][p_h-1][p_w].down_range.w1 + 1;
			ir_output[i][p_h-1][p_w].down_range.h = ir_output[i][p_h-1][p_w].down_range.h2 - ir_output[i][p_h-1][p_w].down_range.h1 + 1;
			//std::cout << "Up block ..." << std::endl;
			//print_subindex(ir_output[i][p_h-1][p_w].down_range);
    }
    //Processing the block down
    if((p_h + 1) < PARTITIONS_H ) {
			p_id_nearby = part_id[p_h+1][p_w]; 
			ir_output[i][p_h+1][p_w].up_range.w1 = required_range.w1;
			ir_output[i][p_h+1][p_w].up_range.w2 = required_range.w2;
			ir_output[i][p_h+1][p_w].up_range.h1 = reuse_output_ranges[p_id_nearby][i].h1;
			ir_output[i][p_h+1][p_w].up_range.h2 = required_range.h2;
			ir_output[i][p_h+1][p_w].up_range.w = ir_output[i][p_h+1][p_w].up_range.w2 - ir_output[i][p_h+1][p_w].up_range.w1 + 1;
			ir_output[i][p_h+1][p_w].up_range.h = ir_output[i][p_h+1][p_w].up_range.h2 - ir_output[i][p_h+1][p_w].up_range.h1 + 1;
			//std::cout << "Down block ..." << std::endl;
			//print_subindex(ir_output[i][p_h+1][p_w].up_range);
    }


}

inline network reshape_network_shuffle(int startfrom, int upto, network net){
    //network net = *netp;//Be careful because we are using a shallow copy here
    numbering_part_id();
    int i;
    int ii;

    //Number of partition assume it is even number
    //Network parameters that are required for processing partition
    int partition_w = PARTITIONS_W;
    int partition_h = PARTITIONS_H;
    int partition = partition_h*partition_w;
    //int startfrom = 0;
    //int upto = startfrom + STAGES-1; //This stage execute upto this layer
     
    int p_w;
    int p_h;

    //Calculate the ranges after partition     
    layer l = net.layers[upto];
    stage_output_partition(0, l.out_w-1, 0, l.out_h-1);
    //Assumption l.out_w and l.out_h are all even numbers here
    for(p_h = 0; p_h < partition_h; p_h++){
	   for(p_w = 0; p_w < partition_w; p_w++){ 
	      sub_index tmp_range = stage_output_partition_ranges[part_id[p_h][p_w]];
	      for(i = upto; i >= startfrom; i--){
    		  layer l = net.layers[i];
		  tmp_range = calculate_range(tmp_range, l);
		  input_ranges[part_id[p_h][p_w]][i] = tmp_range;
	      }
	   }
    }

    for(int p = 0; p < partition; p++){
    	for(i = upto; i >= startfrom; i--){
	    output_ranges[p][i] = calculate_layeroutput_range(input_ranges[p][i], net.layers[i]);
	}
    }


    for(p_h = 0; p_h < partition_h; p_h++){
	for(p_w = 0; p_w < partition_w; p_w++){
	    for(int i = 0; i < STAGES; i++){
		ir_output[i][p_h][p_w].down_range.w = 0;
		ir_output[i][p_h][p_w].down_range.h = 0;
		ir_output[i][p_h][p_w].right_range.w = 0;
		ir_output[i][p_h][p_w].right_range.h = 0;
		ir_output[i][p_h][p_w].left_range.w = 0;
		ir_output[i][p_h][p_w].left_range.h = 0;
		ir_output[i][p_h][p_w].up_range.w = 0;
		ir_output[i][p_h][p_w].up_range.h = 0;
            }
        }
    }




//--------------------------------------------------------------------------------------------------------------
    for(p_h = 0; p_h < partition_h; p_h++){
	for(p_w = (p_h + 1) % 2; p_w < partition_w; p_w = p_w + 2){ 
	    reuse_input_ranges[part_id[p_h][p_w]][upto] = input_ranges[part_id[p_h][p_w]][upto];//Cropped output ranges without overlap for each layer
	    reuse_output_ranges[part_id[p_h][p_w]][upto] = output_ranges[part_id[p_h][p_w]][upto];

	    sub_index tmp = cal_new_range_full(p_h, p_w, upto-1, &output_ranges[0], input_ranges[part_id[p_h][p_w]][upto]);
	    tmp = calculate_range(tmp, net.layers[upto-1]);
	    //printf("Required range at input of layer %d\n", upto-1);
	    //print_subindex(tmp);
	    reuse_input_ranges[part_id[p_h][p_w]][upto-1]  = tmp; 
	    //reuse_output_ranges[part_id[p_h][p_w]][upto-1] = calculate_layeroutput_range(tmp, net.layers[upto-1]);

	    for(i = upto-1; i > 0; i--){
		tmp = cal_new_range_full(p_h, p_w, i-1, &output_ranges[0], tmp);
		tmp = calculate_range(tmp, net.layers[i-1]);
		//printf("Required range at input of layer %d\n", i-1);
		//print_subindex(tmp);
		reuse_input_ranges[part_id[p_h][p_w]][i-1]  = tmp; 
	        //reuse_output_ranges[part_id[p_h][p_w]][i-1] = calculate_layeroutput_range(tmp, net.layers[i-1]);
	    }
	 }
    }

    for(i = 0; i < upto+1; i++){
	for(p_h = 0; p_h < partition_h; p_h++){
	    for(p_w = (p_h) % 2; p_w < partition_w; p_w = p_w + 2){ 
		reuse_input_ranges[part_id[p_h][p_w]][i] = input_ranges[part_id[p_h][p_w]][i];//Cropped output ranges without overlap for each layer
		//reuse_output_ranges[part_id[p_h][p_w]][i] = output_ranges[part_id[p_h][p_w]][i];
	    }
	}
    }




    for(int p = 0; p < partition; p++){
    	for(i = upto; i >= startfrom; i--){
	    reuse_output_ranges[p][i] = calculate_layeroutput_range(reuse_input_ranges[p][i], net.layers[i]);
	}
    }

/*
    for(i = 0; i < upto+1; i++){
	std::cout<< " ================= " << "At layer: "<< i << " ================= "<< std::endl;
        for(int p = 0; p < partition; p++){
	    std::cout << "Layer"<< i << "'s input, part id "<< p <<" ... :" << std::endl;
	    print_subindex(reuse_input_ranges[p][i]);
	}
        for(int p = 0; p < partition; p++){
	    std::cout << "Layer"<< i << "'s output, part id "<< p <<" ... :" << std::endl;
	    print_subindex(reuse_output_ranges[p][i]);
	}
    }
*/
//--------------------------------------------------------------------------------------------------------------




//Calculate reuse
    for(p_h = 0; p_h < partition_h; p_h++){
	for(p_w = (p_h + 1) % 2; p_w < partition_w; p_w = p_w + 2){ 
	    for(i = upto; i > 0; i--){
		cal_reuse_overlap_range_full(p_h, p_w, i-1, &reuse_output_ranges[0], reuse_input_ranges[part_id[p_h][p_w]][i] );
	    }
	 }
    }

/*
//Calculate reuse
    for(i = 0; i < upto; i++){
        for(p_h = 0; p_h < partition_h; p_h++){
	    for(p_w = 0; p_w < partition_w; p_w++){ 
		std::cout << "At layer: "<< i << ", part id is: "<< part_id[p_h][p_w] << std::endl;
		if(ir_output[i][p_h][p_w].up_range.w > 0    && ir_output[i][p_h][p_w].up_range.h > 0 ){
			std::cout << "Up up" << std::endl;	
			print_subindex(ir_output[i][p_h][p_w].up_range);
			std::cout << "Current output is:" << std::endl;	
			print_subindex(reuse_output_ranges[part_id[p_h-1][p_w]][i]);
			std::cout << "Required input from next layer"<< std::endl;
			print_subindex(reuse_input_ranges[part_id[p_h-1][p_w]][i+1]);
		}
		if(ir_output[i][p_h][p_w].down_range.w > 0  && ir_output[i][p_h][p_w].down_range.h > 0 ){
			std::cout << "Down down" << std::endl;			
			print_subindex(ir_output[i][p_h][p_w].down_range);
			std::cout << "Current output is:" << std::endl;	
			print_subindex(reuse_output_ranges[part_id[p_h+1][p_w]][i]);
			std::cout << "Required input from next layer"<< std::endl;
			print_subindex(reuse_input_ranges[part_id[p_h+1][p_w]][i+1]);
		}
		if(ir_output[i][p_h][p_w].left_range.w > 0  && ir_output[i][p_h][p_w].left_range.h > 0 ){
			std::cout << "Left left" << std::endl;					
			print_subindex(ir_output[i][p_h][p_w].left_range);
			std::cout << "Current output is:" << std::endl;	
			print_subindex(reuse_output_ranges[part_id[p_h][p_w-1]][i]);
			std::cout << "Required input from next layer"<< std::endl;
			print_subindex(reuse_input_ranges[part_id[p_h][p_w-1]][i+1]);
		}
		if(ir_output[i][p_h][p_w].right_range.w > 0 && ir_output[i][p_h][p_w].right_range.h > 0 ){
			std::cout << "Right right" << std::endl;				
			print_subindex(ir_output[i][p_h][p_w].right_range);
			std::cout << "Current output is:" << std::endl;	
			print_subindex(reuse_output_ranges[part_id[p_h][p_w+1]][i]);
			std::cout << "Required input from next layer"<< std::endl;
			print_subindex(reuse_input_ranges[part_id[p_h][p_w+1]][i+1]);		
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


    ir_data_spatial_dependency(net, startfrom, upto);
    result_ir_data_spatial_dependency(net, startfrom, upto);
    return net;
}







inline network forward_stage_reuse_full(int p_h, int p_w, float *input,int startfrom, int upto,  network net){
	int part = part_id[p_h][p_w];
        //std::cout << "==========Begin=============: " << p_h<<"   "<<p_w<< std::endl;

	//Reshape first
	net.input = input;
	for(int i = startfrom; i < (upto+1); ++i){
	    net.layers[i].h = (reuse_input_ranges[part][i].h2 - reuse_input_ranges[part][i].h1 + 1); net.layers[i].out_h = (net.layers[i].h/net.layers[i].stride); 
	    net.layers[i].w = (reuse_input_ranges[part][i].w2 - reuse_input_ranges[part][i].w1 + 1); net.layers[i].out_w = (net.layers[i].w/net.layers[i].stride); 
	    net.layers[i].outputs = net.layers[i].out_h * net.layers[i].out_w * net.layers[i].out_c; 
	    net.layers[i].inputs = net.layers[i].h * net.layers[i].w * net.layers[i].c; 
	}


    	for(int i = startfrom; i < upto+1; i++){
		//std::cout << "-----------At layer----------: " << i << std::endl;
		//std::cout << "Input is: "<< std::endl;
		//print_subindex(reuse_input_ranges[part][i]);
		//std::cout << "Output is: "<< std::endl;
		//print_subindex(reuse_output_ranges[part][i]);
		net.layers[i].forward(net.layers[i], net);
		float * cropped_output;
	        if(net.layers[i].type == CONVOLUTIONAL){
			//std::cout<< "We should crop the output of the conv layer first..." <<std::endl;
			layer l = net.layers[i]; 
			sub_index tmp = crop_ranges(reuse_input_ranges[part][i], reuse_output_ranges[part][i]);   
			cropped_output = reshape_input(net.layers[i].output, l.out_w, l.out_h, l.out_c,  tmp.w1, tmp.w2, tmp.h1, tmp.h2);
		}else{
			cropped_output = net.layers[i].output;
		}

		int ir_data_size = 0;
		//What should we record for the current layer?
		if((ir_output[i][p_h][p_w].down_range.w>0)&&(ir_output[i][p_h][p_w].down_range.h>0)){
			//std::cout << "Down: " << std::endl;
			//print_subindex(ir_output[i][p_h][p_w].down_range);
			sub_index down_index = ir_output[i][p_h][p_w].down_range;
			down_index.w1 -= reuse_output_ranges[part][i].w1;
			down_index.w2 -= reuse_output_ranges[part][i].w1;
			down_index.h1 -= reuse_output_ranges[part][i].h1;
			down_index.h2 -= reuse_output_ranges[part][i].h1;
			//print_subindex(down_index);
			ir_data_size = ir_data_size + (down_index.w2 - down_index.w1 + 1)*(down_index.h2 - down_index.h1 + 1);  
			ir_output[i][p_h][p_w].down =   reshape_input(cropped_output, reuse_output_ranges[part][i].w, reuse_output_ranges[part][i].h, net.layers[i].out_c, 
							down_index.w1, down_index.w2, 
							down_index.h1, down_index.h2);

		}
		if((ir_output[i][p_h][p_w].right_range.w>0)&&(ir_output[i][p_h][p_w].right_range.h>0)){
			//std::cout << "Right: " << std::endl;
			//print_subindex(ir_output[i][p_h][p_w].right_range);
			sub_index right_index = ir_output[i][p_h][p_w].right_range;
			right_index.w1 -= reuse_output_ranges[part][i].w1;
			right_index.w2 -= reuse_output_ranges[part][i].w1;
			right_index.h1 -= reuse_output_ranges[part][i].h1;
			right_index.h2 -= reuse_output_ranges[part][i].h1;
			//print_subindex(right_index);
			ir_data_size = ir_data_size + (right_index.w2 - right_index.w1 + 1)*(right_index.h2 - right_index.h1 + 1);  
			ir_output[i][p_h][p_w].right =  reshape_input(cropped_output, reuse_output_ranges[part][i].w, reuse_output_ranges[part][i].h, net.layers[i].out_c, 
							right_index.w1, right_index.w2, 
							right_index.h1, right_index.h2);

		}

		//What should we record for the current layer?
		if((ir_output[i][p_h][p_w].up_range.w>0)&&(ir_output[i][p_h][p_w].up_range.h>0)){
			//std::cout << "Up: " << std::endl;
			//print_subindex(ir_output[i][p_h][p_w].up_range);
			sub_index up_index = ir_output[i][p_h][p_w].up_range;
			up_index.w1 -= reuse_output_ranges[part][i].w1;
			up_index.w2 -= reuse_output_ranges[part][i].w1;
			up_index.h1 -= reuse_output_ranges[part][i].h1;
			up_index.h2 -= reuse_output_ranges[part][i].h1;
			//print_subindex(up_index);
			ir_data_size = ir_data_size + (up_index.w2 - up_index.w1 + 1)*(up_index.h2 - up_index.h1 + 1);  
			ir_output[i][p_h][p_w].up =   reshape_input(cropped_output, reuse_output_ranges[part][i].w, reuse_output_ranges[part][i].h, net.layers[i].out_c, 
							up_index.w1, up_index.w2, 
							up_index.h1, up_index.h2);

		}
		if((ir_output[i][p_h][p_w].left_range.w>0)&&(ir_output[i][p_h][p_w].left_range.h>0)){
			//std::cout << "Left: " << std::endl;
			//print_subindex(ir_output[i][p_h][p_w].left_range);
			sub_index left_index = ir_output[i][p_h][p_w].left_range;
			left_index.w1 -= reuse_output_ranges[part][i].w1;
			left_index.w2 -= reuse_output_ranges[part][i].w1;
			left_index.h1 -= reuse_output_ranges[part][i].h1;
			left_index.h2 -= reuse_output_ranges[part][i].h1;
			//print_subindex(left_index);
			ir_data_size = ir_data_size + (left_index.w2 - left_index.w1 + 1)*(left_index.h2 - left_index.h1 + 1);  
			ir_output[i][p_h][p_w].left =  reshape_input(cropped_output, reuse_output_ranges[part][i].w, reuse_output_ranges[part][i].h, net.layers[i].out_c, 
							left_index.w1, left_index.w2, 
							left_index.h1, left_index.h2);

		}

		if(net.layers[i].out_c * ir_data_size * sizeof(float) > 0)
		 std::cout << "The size of overlapped data for part "<< part << " at layer "<< i <<" is:"<< (float)(net.layers[i].out_c * ir_data_size * sizeof(float))/1024.0/1024.0 << std::endl;

		int up = 0;
		int left = 0;	
		int right = 0;
		int down = 0;
		
		if(i < upto){
	          if(net.layers[i+1].type == CONVOLUTIONAL){
			//If next layer is a convlutional layer, then collect the adj parts output
			//std::cout<< "We should gather the output adj parts from this layer..." <<std::endl;
			if((p_h>0)&&(ir_output[i][p_h-1][p_w].down_range.w>0)&&(ir_output[i][p_h-1][p_w].down_range.h>0)){
				//std::cout << "Require input from above part in this layer: " << std::endl;
				up = 1;
				//print_subindex(ir_output[i][p_h-1][p_w].down_range);
			}
			if((p_w>0)&&(ir_output[i][p_h][p_w-1].right_range.w>0)&&(ir_output[i][p_h][p_w-1].right_range.h>0)){
				//std::cout << "Require input from left part in this layer: " << std::endl;
				left = 1;
				//print_subindex(ir_output[i][p_h][p_w-1].right_range);
			}

			if(((p_h+1)<PARTITIONS_H)&&(ir_output[i][p_h+1][p_w].up_range.w>0)&&(ir_output[i][p_h+1][p_w].up_range.h>0)){
				//std::cout << "Require input from below part in this layer: " << std::endl;
				down = 1;
				//print_subindex(ir_output[i][p_h+1][p_w].up_range);
			}
			if(((p_w+1)<PARTITIONS_W)&&(ir_output[i][p_h][p_w+1].left_range.w>0)&&(ir_output[i][p_h][p_w+1].left_range.h>0)){
				//std::cout << "Require input from right part in this layer: " << std::endl;
				right = 1;
				//print_subindex(ir_output[i][p_h][p_w+1].left_range);
			}

		  }
                }

		float* next_input;
		if(i < upto){

			sub_index main_index;
			main_index.w2 = reuse_output_ranges[part_id[p_h][p_w]][i].w2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1;
			main_index.h2 = reuse_output_ranges[part_id[p_h][p_w]][i].h2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1;
			main_index.w1 = reuse_output_ranges[part_id[p_h][p_w]][i].w1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1; 
			main_index.h1 = reuse_output_ranges[part_id[p_h][p_w]][i].h1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1;  
			main_index.w = main_index.w2 - main_index.w1 + 1;
			main_index.h = main_index.h2 - main_index.h1 + 1;
			//std::cout << "Main index is ... ... :" << std::endl;
			//print_subindex(main_index);
			//print_subindex(reuse_output_ranges[part_id[p_h][p_w]][i]);
			//std::cout << "Required index from next layer is ... ... :" << std::endl;
			//print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i+1]);

		        next_input = (float*)malloc(reuse_input_ranges[part_id[p_h][p_w]][i+1].w*reuse_input_ranges[part_id[p_h][p_w]][i+1].h*net.layers[i].out_c*sizeof(float));
		        copy_input_to_output(cropped_output, next_input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
							net.layers[i].out_c, main_index.w1, main_index.w2, main_index.h1, main_index.h2);
		}

		//std::cout<< ".....................OK, let us do it then....................." <<std::endl;
		int reuse_h;
		int reuse_w;
		sub_index reuse_index;

		if(up == 1){
			reuse_h = p_h - 1; reuse_w = p_w;		
			reuse_index.w2 = ir_output[i][reuse_h][reuse_w].down_range.w2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1;
			reuse_index.h2 = ir_output[i][reuse_h][reuse_w].down_range.h2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1;
			reuse_index.w1 = ir_output[i][reuse_h][reuse_w].down_range.w1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1;
			reuse_index.h1 = ir_output[i][reuse_h][reuse_w].down_range.h1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1; 
			reuse_index.w = reuse_index.w2 - reuse_index.w1 + 1;
			reuse_index.h = reuse_index.h2 - reuse_index.h1 + 1;
		        //std::cout << "Up index is ... ... :" << std::endl;
			//print_subindex(ir_output[i][reuse_h][reuse_w].down_range);
			//print_subindex(reuse_index);
                        copy_input_to_output(ir_output[i][reuse_h][reuse_w].down, next_input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, reuse_index.w1, reuse_index.w2, reuse_index.h1, reuse_index.h2);
		}
		if(down == 1){
			reuse_h = p_h + 1; reuse_w = p_w;		
			reuse_index.w2 = ir_output[i][reuse_h][reuse_w].up_range.w2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1;
			reuse_index.h2 = ir_output[i][reuse_h][reuse_w].up_range.h2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1;
			reuse_index.w1 = ir_output[i][reuse_h][reuse_w].up_range.w1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1;
			reuse_index.h1 = ir_output[i][reuse_h][reuse_w].up_range.h1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1; 
			reuse_index.w = reuse_index.w2 - reuse_index.w1 + 1;
			reuse_index.h = reuse_index.h2 - reuse_index.h1 + 1;
		        //std::cout << "Down index is ... ... :" << std::endl;
			//print_subindex(ir_output[i][reuse_h][reuse_w].up_range);
			//print_subindex(reuse_index);
                        copy_input_to_output(ir_output[i][reuse_h][reuse_w].up, next_input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, reuse_index.w1, reuse_index.w2, reuse_index.h1, reuse_index.h2);
		}
		if(left == 1){
			reuse_h = p_h; reuse_w = p_w - 1;		
			reuse_index.w2 = ir_output[i][reuse_h][reuse_w].right_range.w2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1;
			reuse_index.h2 = ir_output[i][reuse_h][reuse_w].right_range.h2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1;
			reuse_index.w1 = ir_output[i][reuse_h][reuse_w].right_range.w1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1;
			reuse_index.h1 = ir_output[i][reuse_h][reuse_w].right_range.h1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1; 
			reuse_index.w = reuse_index.w2 - reuse_index.w1 + 1;
			reuse_index.h = reuse_index.h2 - reuse_index.h1 + 1;
		        //std::cout << "Left index is ... ... :" << std::endl;
			//print_subindex(ir_output[i][reuse_h][reuse_w].right_range);
			//print_subindex(reuse_index);
                        copy_input_to_output(ir_output[i][reuse_h][reuse_w].right, next_input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, reuse_index.w1, reuse_index.w2, reuse_index.h1, reuse_index.h2);
		}
		if(right == 1){
			reuse_h = p_h; reuse_w = p_w + 1;		
			reuse_index.w2 = ir_output[i][reuse_h][reuse_w].left_range.w2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1;
			reuse_index.h2 = ir_output[i][reuse_h][reuse_w].left_range.h2 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1;
			reuse_index.w1 = ir_output[i][reuse_h][reuse_w].left_range.w1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].w1;
			reuse_index.h1 = ir_output[i][reuse_h][reuse_w].left_range.h1 - reuse_input_ranges[part_id[p_h][p_w]][i+1].h1; 
			reuse_index.w = reuse_index.w2 - reuse_index.w1 + 1;
			reuse_index.h = reuse_index.h2 - reuse_index.h1 + 1;
		        //std::cout << "Right index is ... ... :" << std::endl;
			//print_subindex(ir_output[i][reuse_h][reuse_w].left_range);
			//print_subindex(reuse_index);
                        copy_input_to_output(ir_output[i][reuse_h][reuse_w].left, next_input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, reuse_index.w1, reuse_index.w2, reuse_index.h1, reuse_index.h2);
		}

		net.input = next_input;
	}
        //std::cout << "==========Finish=============: " << part_id[p_h][p_w] << std::endl;

	return net;

}


#endif
