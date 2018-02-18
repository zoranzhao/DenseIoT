//#include "darknet_dist.h"
#include "darknet_util.h"


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

/*


inline network reshape_network_debug(int startfrom, int upto, network net){
    //network net = *netp;//Be careful because we are using a shallow copy here
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
		ir_output[i][p_h][p_w].corner_range.w = 0;
		ir_output[i][p_h][p_w].corner_range.h = 0;
            }
        }
    }
//--------------------------------------------------------------------------------------------------------------
    for(p_h = 0; p_h < partition_h; p_h++){
	for(p_w = 0; p_w < partition_w; p_w++){ 
	    reuse_input_ranges[part_id[p_h][p_w]][upto] = input_ranges[part_id[p_h][p_w]][upto];//Cropped output ranges without overlap for each layer
	    reuse_output_ranges[part_id[p_h][p_w]][upto] = output_ranges[part_id[p_h][p_w]][upto];

	    sub_index tmp = cal_new_range(p_h, p_w, upto-1, &output_ranges[0], input_ranges[part_id[p_h][p_w]][upto]);
	    tmp = calculate_range(tmp, net.layers[upto-1]);
	    //printf("Required range at input of layer %d\n", upto-1);
	    //print_subindex(tmp);
	    reuse_input_ranges[part_id[p_h][p_w]][upto-1]  = tmp; 
	    //reuse_output_ranges[part_id[p_h][p_w]][upto-1] = calculate_layeroutput_range(tmp, net.layers[upto-1]);

	    for(i = upto-1; i > 0; i--){
		tmp = cal_new_range(p_h, p_w, i-1, &output_ranges[0], tmp);
		tmp = calculate_range(tmp, net.layers[i-1]);
		//printf("Required range at input of layer %d\n", i-1);
		//print_subindex(tmp);
		reuse_input_ranges[part_id[p_h][p_w]][i-1]  = tmp; 
	        //reuse_output_ranges[part_id[p_h][p_w]][i-1] = calculate_layeroutput_range(tmp, net.layers[i-1]);
	    }
	 }
    }
    for(int p = 0; p < partition; p++){
    	for(i = upto; i >= startfrom; i--){
	    reuse_output_ranges[p][i] = calculate_layeroutput_range(reuse_input_ranges[p][i], net.layers[i]);
	}
    }
//--------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------
    for(p_h = 0; p_h < partition_h; p_h++){
	for(p_w = 0; p_w < partition_w; p_w++){ 
	    for(i = upto; i > 0; i--){
		cal_reuse_overlap_range(p_h, p_w, i-1, &reuse_output_ranges[0], reuse_input_ranges[part_id[p_h][p_w]][i] );
	    }
	 }
    }

//--------------------------------------------------------------------------------------------------------------



    for(i = startfrom; i <= upto; i++){
       std::cout << "-----------At layer----------: " << i << std::endl;
       for(p_h = 0; p_h < partition_h; p_h++){
	   for(p_w = 0; p_w < partition_w; p_w++){ 
	         std::cout << "part: " << part_id[p_h][p_w] << std::endl;
		  std::cout << "Input is: "<< std::endl;
		  print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i]);
		  std::cout << "Output is: "<< std::endl;
		  print_subindex(reuse_output_ranges[part_id[p_h][p_w]][i]);

	          if (i < upto){
		    std::cout << "Next layer input is ..." << std::endl;
   		    print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i+1]);
		    if((p_h>0)&&(ir_output[i][p_h-1][p_w].down_range.w>0)&&(ir_output[i][p_h-1][p_w].down_range.h>0)){
			     std::cout << "Require input from above part in last layer: " << std::endl;
			     print_subindex(ir_output[i][p_h-1][p_w].down_range);

	            }
		    if((p_w>0)&&(ir_output[i][p_h][p_w-1].right_range.w>0)&&(ir_output[i][p_h][p_w-1].right_range.h>0)){
			     std::cout << "Require input from left part in last layer: " << std::endl;
			     print_subindex(ir_output[i][p_h][p_w-1].right_range);

		    }
		    if(p_h > 0 && p_w > 0&&(ir_output[i][p_h-1][p_w-1].corner_range.w>0)&&(ir_output[i][p_h-1][p_w-1].corner_range.h>0)) {
			     std::cout << "Require input from left above part in last layer: " << std::endl;
			     print_subindex(ir_output[i][p_h-1][p_w-1].corner_range);
		    }
		  }


	      std::cout << "==========overlap=============: " << part_id[p_h][p_w] << std::endl;
	      std::cout << "==========overlap=============: " << part_id[p_h][p_w] << std::endl;
	   }
       }
    }

    for(p_h = 0; p_h < partition_h; p_h++){
      for(p_w = 0; p_w < partition_w; p_w++){ 
        std::cout << "==========now we can=============: " << part_id[p_h][p_w] << std::endl;
        std::cout << "==========now we can=============: " << part_id[p_h][p_w] << std::endl;
    	for(i = startfrom; i < upto; i++){
		std::cout << "-----------At layer----------: " << i << std::endl;
		std::cout << "Input is: "<< std::endl;
		print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i]);
		std::cout << "Output is: "<< std::endl;
		print_subindex(reuse_output_ranges[part_id[p_h][p_w]][i]);
		if((ir_output[i][p_h][p_w].down_range.w>0)&&(ir_output[i][p_h][p_w].down_range.h>0)){
			     std::cout << "Down!!!: " << std::endl;
			     print_subindex(ir_output[i][p_h][p_w].down_range);

		}
		if((ir_output[i][p_h][p_w].right_range.w>0)&&(ir_output[i][p_h][p_w].right_range.h>0)){
			     std::cout << "right: " << std::endl;
			     print_subindex(ir_output[i][p_h][p_w].right_range);

		}
		if((ir_output[i][p_h][p_w].corner_range.w>0)&&(ir_output[i][p_h][p_w].corner_range.h>0)) {
			     std::cout << "corner: " << std::endl;
			     print_subindex(ir_output[i][p_h][p_w].corner_range);
		}
	}
        std::cout << "==========now we can=============: " << part_id[p_h][p_w] << std::endl;
        std::cout << "==========now we can=============: " << part_id[p_h][p_w] << std::endl;
      }
    }



    for(p_h = 0; p_h < partition_h; p_h++){
      for(p_w = 0; p_w < partition_w; p_w++){ 
        std::cout << "==========fake=============: " << part_id[p_h][p_w] << std::endl;
        std::cout << "==========fake=============: " << part_id[p_h][p_w] << std::endl;
    	for(i = startfrom; i < upto; i++){

		std::cout << "-----------At layer----------: " << i << std::endl;
		std::cout << "Input is: "<< std::endl;
		print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i]);
		std::cout << "Output is: "<< std::endl;
		print_subindex(reuse_output_ranges[part_id[p_h][p_w]][i]);
		//What should we record for the current layer
		if((ir_output[i][p_h][p_w].down_range.w>0)&&(ir_output[i][p_h][p_w].down_range.h>0)){
			     std::cout << "Down!!!: " << std::endl;
			     print_subindex(ir_output[i][p_h][p_w].down_range);

		}
		if((ir_output[i][p_h][p_w].right_range.w>0)&&(ir_output[i][p_h][p_w].right_range.h>0)){
			     std::cout << "right: " << std::endl;
			     print_subindex(ir_output[i][p_h][p_w].right_range);

		}
		if((ir_output[i][p_h][p_w].corner_range.w>0)&&(ir_output[i][p_h][p_w].corner_range.h>0)) {
			     std::cout << "corner: " << std::endl;
			     print_subindex(ir_output[i][p_h][p_w].corner_range);
		}


          
		if(i > 0){
		   std::cout << "Require input from last layer output: "<< std::endl;
		   print_subindex(reuse_output_ranges[part_id[p_h][p_w]][i-1]);
	           if(net.layers[i].type == CONVOLUTIONAL){
			    std::cout << "Conv! Require input from last layer adj parts output: "<< std::endl;
			    if((p_h>0)&&(ir_output[i-1][p_h-1][p_w].down_range.w>0)&&(ir_output[i-1][p_h-1][p_w].down_range.h>0)){
				     std::cout << "Require input from above part in last layer: " << std::endl;
				     print_subindex(ir_output[i-1][p_h-1][p_w].down_range);

			    }
			    if((p_w>0)&&(ir_output[i-1][p_h][p_w-1].right_range.w>0)&&(ir_output[i-1][p_h][p_w-1].right_range.h>0)){
				     std::cout << "Require input from left part in last layer: " << std::endl;
				     print_subindex(ir_output[i-1][p_h][p_w-1].right_range);

			    }
			    if(p_h > 0 && p_w > 0&&(ir_output[i-1][p_h-1][p_w-1].corner_range.w>0)&&(ir_output[i-1][p_h-1][p_w-1].corner_range.h>0)) {
				     std::cout << "Require input from left above part in last layer: " << std::endl;
				     print_subindex(ir_output[i-1][p_h-1][p_w-1].corner_range);
			    }

	           }
		}



	        if(net.layers[i].type == CONVOLUTIONAL){
			std::cout<< "We should probably crop the output of the conv layer first..." <<std::endl;
		}

		int up = 0;
		int corner = 0;
		int left = 0;		
		if(i < upto){
	          if(net.layers[i+1].type == CONVOLUTIONAL){
			//If next layer is a convlutional layer, then collect the adj parts output
			std::cout<< "We should gather the output adj parts from this layer..." <<std::endl;
			if((p_h>0)&&(ir_output[i][p_h-1][p_w].down_range.w>0)&&(ir_output[i][p_h-1][p_w].down_range.h>0)){
				std::cout << "Require input from above part in this layer: " << std::endl;
				up = 1;
				print_subindex(ir_output[i][p_h-1][p_w].down_range);
			}
			if((p_w>0)&&(ir_output[i][p_h][p_w-1].right_range.w>0)&&(ir_output[i][p_h][p_w-1].right_range.h>0)){
				std::cout << "Require input from left part in this layer: " << std::endl;
				left = 1;
				print_subindex(ir_output[i][p_h][p_w-1].right_range);
			}
			if((p_h>0)&&(p_w>0)&&(ir_output[i][p_h-1][p_w-1].corner_range.w>0)&&(ir_output[i][p_h-1][p_w-1].corner_range.h>0)) {
				std::cout << "Require input from left above part in this layer: " << std::endl;
				corner = 1;
				print_subindex(ir_output[i][p_h-1][p_w-1].corner_range);
			}

		  }
                }

		std::cout<< ".....................OK, let us do it then....................." <<std::endl;
		if(up == 1 && corner == 0){
			std::cout << "Input for next layer is: "<< std::endl;
			print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i+1]);
			sub_index main_index;
			sub_index up_index;

			main_index.w2 = reuse_output_ranges[part_id[p_h][p_w]][i].w2 - reuse_output_ranges[part_id[p_h][p_w]][i].w1;
			main_index.h2 = reuse_output_ranges[part_id[p_h][p_w]][i].h2 - reuse_output_ranges[part_id[p_h][p_w]][i].h1;
			main_index.w1 = 0; main_index.w = main_index.w2 - main_index.w1 + 1;
			main_index.h1 = 0; main_index.h = main_index.h2 - main_index.h1 + 1;
			

			up_index.w2 = ir_output[i][p_h-1][p_w].down_range.w2 - ir_output[i][p_h-1][p_w].down_range.w1;
			up_index.h2 = ir_output[i][p_h-1][p_w].down_range.h2 - ir_output[i][p_h-1][p_w].down_range.h1;
			up_index.w1 = 0; up_index.w = up_index.w2 - up_index.w1 + 1;
			up_index.h1 = 0; up_index.h = up_index.h2 - up_index.h1 + 1;

			main_index.h1 += up_index.h;
			main_index.h2 += up_index.h;
				
			print_subindex(main_index);
			print_subindex(up_index);
			//input [0, dh2 - dh1]    copy into ==> output(w*h)   [dh1, dh2]
			//	[0, dw2 - dw1]			              [dw1, dw2]

                        //copy_input_to_output(ir_output[i][p_h-1][p_w].down, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						//net.layers[i].out_c, up_index.w1, up_index.w2, up_index.h1, up_index.h2);
                        //copy_input_to_output(ir_output[i][p_h][p_w-1].left, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						//net.layers[i].out_c, left_index.w1, left_index.w2, left_index.h1, left_index.h2);
                        //copy_input_to_output(ir_output[i][p_h-1][p_w-1].corner, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						//net.layers[i].out_c, corner_index.w1, corner_index.w2, corner_index.h1, corner_index.h2);
                        //copy_input_to_output(net.layers[i].output, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						//net.layers[i].out_c, main_index.w1, main_index.w2, main_index.h1, main_index.h2);


		}else if(left == 1 && corner == 0){
			std::cout << "Input for next layer is: "<< std::endl;
			print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i+1]);
			sub_index main_index;
			sub_index left_index;

			main_index.w2 = reuse_output_ranges[part_id[p_h][p_w]][i].w2 - reuse_output_ranges[part_id[p_h][p_w]][i].w1;
			main_index.h2 = reuse_output_ranges[part_id[p_h][p_w]][i].h2 - reuse_output_ranges[part_id[p_h][p_w]][i].h1;
			main_index.w1 = 0; main_index.w = main_index.w2 - main_index.w1 + 1;
			main_index.h1 = 0; main_index.h = main_index.h2 - main_index.h1 + 1;
			

			left_index.w2 = ir_output[i][p_h][p_w-1].right_range.w2 - ir_output[i][p_h][p_w-1].right_range.w1;
			left_index.h2 = ir_output[i][p_h][p_w-1].right_range.h2 - ir_output[i][p_h][p_w-1].right_range.h1;
			left_index.w1 = 0; left_index.w = left_index.w2 - left_index.w1 + 1;
			left_index.h1 = 0; left_index.h = left_index.h2 - left_index.h1 + 1;

			main_index.w1 += left_index.w;
			main_index.w2 += left_index.w;
				
			print_subindex(main_index);
			print_subindex(left_index);

		}else if(corner == 1){
			std::cout << "Input for next layer is: "<< std::endl;
			print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i+1]);
			sub_index main_index;
			sub_index left_index;
			sub_index up_index;
			sub_index corner_index;

			main_index.w2 = reuse_output_ranges[part_id[p_h][p_w]][i].w2 - reuse_output_ranges[part_id[p_h][p_w]][i].w1;
			main_index.h2 = reuse_output_ranges[part_id[p_h][p_w]][i].h2 - reuse_output_ranges[part_id[p_h][p_w]][i].h1;
			main_index.w1 = 0; main_index.w = main_index.w2 - main_index.w1 + 1;
			main_index.h1 = 0; main_index.h = main_index.h2 - main_index.h1 + 1;
			

			left_index.w2 = ir_output[i][p_h][p_w-1].right_range.w2 - ir_output[i][p_h][p_w-1].right_range.w1;
			left_index.h2 = ir_output[i][p_h][p_w-1].right_range.h2 - ir_output[i][p_h][p_w-1].right_range.h1;
			left_index.w1 = 0; left_index.w = left_index.w2 - left_index.w1 + 1;
			left_index.h1 = 0; left_index.h = left_index.h2 - left_index.h1 + 1;

			main_index.w1 += left_index.w;
			main_index.w2 += left_index.w;
				

			up_index.w2 = ir_output[i][p_h-1][p_w].down_range.w2 - ir_output[i][p_h-1][p_w].down_range.w1;
			up_index.h2 = ir_output[i][p_h-1][p_w].down_range.h2 - ir_output[i][p_h-1][p_w].down_range.h1;
			up_index.w1 = 0; up_index.w = up_index.w2 - up_index.w1 + 1;
			up_index.h1 = 0; up_index.h = up_index.h2 - up_index.h1 + 1;

			main_index.h1 += up_index.h;
			main_index.h2 += up_index.h;


			corner_index.w2 = ir_output[i][p_h-1][p_w-1].corner_range.w2 - ir_output[i][p_h-1][p_w-1].corner_range.w1;
			corner_index.h2 = ir_output[i][p_h-1][p_w-1].corner_range.h2 - ir_output[i][p_h-1][p_w-1].corner_range.h1;
			corner_index.w1 = 0; corner_index.w = corner_index.w2 - corner_index.w1 + 1;
			corner_index.h1 = 0; corner_index.h = corner_index.h2 - corner_index.h1 + 1;

			left_index.h1 += corner_index.h;
			left_index.h2 += corner_index.h;

			up_index.w1 += corner_index.w;
			up_index.w2 += corner_index.w;


			print_subindex(main_index);
			print_subindex(left_index);
			print_subindex(up_index);
			print_subindex(corner_index);
		}
		


	}
        std::cout << "==========fake=============: " << part_id[p_h][p_w] << std::endl;
        std::cout << "==========fake=============: " << part_id[p_h][p_w] << std::endl;
      }
    }





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

    return net;
}

*/
#endif

