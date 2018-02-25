#include "darknet_util.h"

#ifndef SERIALIZATION_MR__H
#define SERIALIZATION_MR__H
inline float* result_ir_data_serialization_mr(network net, int part, int i);
inline void result_ir_data_deserialization_mr(network net, int part, float* input, int i);

inline float* req_ir_data_serialization_mr(network net, int part, int i);
inline void req_ir_data_deserialization_mr(network net, int part, float* input, int i);

inline void cal_dependency_mr(network net, int startfrom, int upto){

        for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
           for(int p_w = 0; p_w < PARTITIONS_W; p_w++){
             for(int i = startfrom; i < upto+1; i++){
	        req_ir_data_size_mr[p_h][p_w][i] = 0;
		up_mr[p_h][p_w][i] = 0;
		left_mr[p_h][p_w][i] = 0;
		right_mr[p_h][p_w][i] = 0;
		down_mr[p_h][p_w][i] = 0;
		for(int corner_num=0; corner_num<4; corner_num++)
		    corners_mr[corner_num][p_h][p_w][i] = 0;
	     }
	   }
	}

	for(int part_id = 0; part_id < PARTITIONS; part_id++){
            int p_h = part_id / PARTITIONS_W; 
            int p_w = part_id % PARTITIONS_W;
	    for(int i = startfrom; i < upto; i++){
		if(net.layers[i+1].type == CONVOLUTIONAL){
			//If next layer is a convlutional layer, then collect the adj parts output
			//std::cout<< "We should gather the output adj parts from this layer..." <<std::endl;
			if((p_h>0)&&(ir_output_mr[i][p_h-1][p_w].down_range.w>0)&&(ir_output_mr[i][p_h-1][p_w].down_range.h>0)){
				//std::cout << "Require input from above part in this layer: " << std::endl;
				up_mr[p_h][p_w][i+1] = 1;
				//print_subindex(ir_output_mr[i][p_h-1][p_w].down_range);
				req_ir_data_size_mr[p_h][p_w][i+1] += ir_output_mr[i][p_h-1][p_w].down_range.w*ir_output_mr[i][p_h-1][p_w].down_range.h*net.layers[i].out_c; 
			}
			if((p_w>0)&&(ir_output_mr[i][p_h][p_w-1].right_range.w>0)&&(ir_output_mr[i][p_h][p_w-1].right_range.h>0)){
				//std::cout << "Require input from left part in this layer: " << std::endl;
				left_mr[p_h][p_w][i+1] = 1;
				//print_subindex(ir_output_mr[i][p_h][p_w-1].right_range);
				req_ir_data_size_mr[p_h][p_w][i+1] += ir_output_mr[i][p_h][p_w-1].right_range.w*ir_output_mr[i][p_h][p_w-1].right_range.h*net.layers[i].out_c; 
			}

			if(((p_h+1)<PARTITIONS_H)&&(ir_output_mr[i][p_h+1][p_w].up_range.w>0)&&(ir_output_mr[i][p_h+1][p_w].up_range.h>0)){
				//std::cout << "Require input from below part in this layer: " << std::endl;
				down_mr[p_h][p_w][i+1] = 1;
				//print_subindex(ir_output_mr[i][p_h+1][p_w].up_range);
				req_ir_data_size_mr[p_h][p_w][i+1] += ir_output_mr[i][p_h+1][p_w].up_range.w*ir_output_mr[i][p_h+1][p_w].up_range.h*net.layers[i].out_c; 
			}
			if(((p_w+1)<PARTITIONS_W)&&(ir_output_mr[i][p_h][p_w+1].left_range.w>0)&&(ir_output_mr[i][p_h][p_w+1].left_range.h>0)){
				//std::cout << "Require input from right part in this layer: " << std::endl;
				right_mr[p_h][p_w][i+1] = 1;
				//print_subindex(ir_output_mr[i][p_h][p_w+1].left_range);
				req_ir_data_size_mr[p_h][p_w][i+1] += ir_output_mr[i][p_h][p_w+1].left_range.w*ir_output_mr[i][p_h][p_w+1].left_range.h*net.layers[i].out_c; 
			}
			//left up corner
			if(p_w > 0 && p_h > 0) {
				if(ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].w > 0 && ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].h > 0){
				     corners_mr[0][p_h][p_w][i+1] = 1;
				     req_ir_data_size_mr[p_h][p_w][i+1] += 
					ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].w*ir_output_mr[i][p_h-1][p_w-1].corner_range_mr[3].h*net.layers[i].out_c; 
				}
			}    
			//right up corner
			if((p_w + 1) < PARTITIONS_W && p_h > 0) {
				if(ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].w > 0 && ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].h > 0){
				     corners_mr[1][p_h][p_w][i+1] = 1;
				     req_ir_data_size_mr[p_h][p_w][i+1] += 
					ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].w*ir_output_mr[i][p_h-1][p_w+1].corner_range_mr[2].h*net.layers[i].out_c; 
				}
			}    
			//left down corner
			if(p_w > 0 && (p_h + 1) < PARTITIONS_H) {
				if(ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].w > 0 && ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].h > 0){
				     corners_mr[2][p_h][p_w][i+1] = 1;
				     req_ir_data_size_mr[p_h][p_w][i+1] += 
					ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].w *ir_output_mr[i][p_h+1][p_w-1].corner_range_mr[1].h*net.layers[i].out_c; 
				}
			}    
			//right down corner
			if((p_w + 1) < PARTITIONS_W && (p_h + 1) < PARTITIONS_H) {
				if(ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].w > 0 && ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].h > 0){
				     corners_mr[3][p_h][p_w][i+1] = 1;
				     req_ir_data_size_mr[p_h][p_w][i+1] += 
					ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].w *ir_output_mr[i][p_h+1][p_w+1].corner_range_mr[0].h*net.layers[i].out_c; 
				}
			}    
		}

	    }
	}
}


inline void result_cal_dependency_mr(network net, int startfrom, int upto){
        for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
           for(int p_w = 0; p_w < PARTITIONS_W; p_w++){
             for(int i = startfrom; i < upto+1; i++){
	        result_ir_data_size_mr[p_h][p_w][i] = 0;
		result_up_mr[p_h][p_w][i] = 0;
		result_left_mr[p_h][p_w][i] = 0;
		result_right_mr[p_h][p_w][i] = 0;
		result_down_mr[p_h][p_w][i] = 0;
		for(int corner_num=0; corner_num<4; corner_num++)
		    result_corners_mr[corner_num][p_h][p_w][i] = 0;
	     }
	   }
	}

        for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
           for(int p_w = 0; p_w < PARTITIONS_W; p_w++){
             for(int i = startfrom; i < upto+1; i++){

		if((ir_output_mr[i][p_h][p_w].down_range.w>0)&&(ir_output_mr[i][p_h][p_w].down_range.h>0)){
			result_down_mr[p_h][p_w][i] = 1;	
			result_ir_data_size_mr[p_h][p_w][i] += ir_output_mr[i][p_h][p_w].down_range.w*ir_output_mr[i][p_h][p_w].down_range.h*net.layers[i].out_c;  	
		}
		if((ir_output_mr[i][p_h][p_w].right_range.w>0)&&(ir_output_mr[i][p_h][p_w].right_range.h>0)){
			result_right_mr[p_h][p_w][i] = 1;	
			result_ir_data_size_mr[p_h][p_w][i] += ir_output_mr[i][p_h][p_w].right_range.w*ir_output_mr[i][p_h][p_w].right_range.h*net.layers[i].out_c;  	
		}
		//What should we record for the current layer?
		if((ir_output_mr[i][p_h][p_w].up_range.w>0)&&(ir_output_mr[i][p_h][p_w].up_range.h>0)){
			result_up_mr[p_h][p_w][i] = 1;	
			result_ir_data_size_mr[p_h][p_w][i] +=ir_output_mr[i][p_h][p_w].up_range.w*ir_output_mr[i][p_h][p_w].up_range.h*net.layers[i].out_c;  		
		}
		if((ir_output_mr[i][p_h][p_w].left_range.w>0)&&(ir_output_mr[i][p_h][p_w].left_range.h>0)){
			result_left_mr[p_h][p_w][i] = 1;	
			result_ir_data_size_mr[p_h][p_w][i] +=ir_output_mr[i][p_h][p_w].left_range.w*ir_output_mr[i][p_h][p_w].left_range.h*net.layers[i].out_c;  		
		}

		for(int corner_number = 0; corner_number < 4; corner_number ++){
			if(ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].w > 0 && ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].h > 0 ){
				result_corners_mr[corner_number][p_h][p_w][i] = 1;
				result_ir_data_size_mr[p_h][p_w][i] +=
				ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].w*ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].h*net.layers[i].out_c;  		
			}
		}
	     }
	   }
	}

}



inline float* result_ir_data_serialization_mr(network net, int part, int i){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;
      float *output;
      output = (float*)malloc(result_ir_data_size_mr[p_h][p_w][i]*sizeof(float));
      //std::cout  << "The total number of bytes in this partition ... "<< result_ir_data_size_mr[p_h][p_w][i] << std::endl;
      //int profile_num = 0;

      //std::cout << "===================At layer: " << i  << std::endl;
      if(result_up_mr[p_h][p_w][i]==1){
		std::cout << "serialization up, size is: " << std::endl;
		std::cout << ir_output_mr[i][p_h][p_w].up_range.w*ir_output_mr[i][p_h][p_w].up_range.h*net.layers[i].out_c<<std::endl;
		memcpy(output, ir_output_mr[i][p_h][p_w].up, ir_output_mr[i][p_h][p_w].up_range.w*ir_output_mr[i][p_h][p_w].up_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output_mr[i][p_h][p_w].up_range.w*ir_output_mr[i][p_h][p_w].up_range.h*net.layers[i].out_c;
      }
      if(result_left_mr[p_h][p_w][i]==1){
		std::cout << "serialization left, size is: " << std::endl;
		std::cout << ir_output_mr[i][p_h][p_w].left_range.w*ir_output_mr[i][p_h][p_w].left_range.h*net.layers[i].out_c<<std::endl;
		memcpy(output, ir_output_mr[i][p_h][p_w].left, ir_output_mr[i][p_h][p_w].left_range.w*ir_output_mr[i][p_h][p_w].left_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output_mr[i][p_h][p_w].left_range.w*ir_output_mr[i][p_h][p_w].left_range.h*net.layers[i].out_c;
      }
      if(result_down_mr[p_h][p_w][i]==1){
		std::cout << "serialization down, size is: " << std::endl;
		std::cout << ir_output_mr[i][p_h][p_w].down_range.w*ir_output_mr[i][p_h][p_w].down_range.h*net.layers[i].out_c << std::endl;
		memcpy(output, ir_output_mr[i][p_h][p_w].down, ir_output_mr[i][p_h][p_w].down_range.w*ir_output_mr[i][p_h][p_w].down_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output_mr[i][p_h][p_w].down_range.w*ir_output_mr[i][p_h][p_w].down_range.h*net.layers[i].out_c;
      }
      if(result_right_mr[p_h][p_w][i]==1){
		std::cout << "serialization right, size is: " << std::endl;
		std::cout << ir_output_mr[i][p_h][p_w].right_range.w*ir_output_mr[i][p_h][p_w].right_range.h*net.layers[i].out_c << std::endl;
		memcpy(output, ir_output_mr[i][p_h][p_w].right,ir_output_mr[i][p_h][p_w].right_range.w*ir_output_mr[i][p_h][p_w].right_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output_mr[i][p_h][p_w].right_range.w*ir_output_mr[i][p_h][p_w].right_range.h*net.layers[i].out_c;
      }
      //std::cout << "===================At layer: " << i  << std::endl;

      for(int corner_number = 0; corner_number < 4; corner_number ++){
	 if(result_corners_mr[corner_number][p_h][p_w][i] > 0 ){
		std::cout << "serialization corner "<< corner_number <<", size is: " << std::endl;
		std::cout << ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].w*ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].h*net.layers[i].out_c << std::endl;
		memcpy(output, ir_output_mr[i][p_h][p_w].corner_mr[corner_number],
			ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].w*ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].w*ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].h*net.layers[i].out_c;
 	 }
      }

      return (output - result_ir_data_size_mr[p_h][p_w][i]);
}




inline void result_ir_data_deserialization_mr(network net, int part, float* input, int i){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;

      float* input_data = input;


      if(result_up_mr[p_h][p_w][i]==1){
		ir_output_mr[i][p_h][p_w].up = (float*)malloc(ir_output_mr[i][p_h][p_w].up_range.w*ir_output_mr[i][p_h][p_w].up_range.h*net.layers[i].out_c*sizeof(float));
		memcpy(ir_output_mr[i][p_h][p_w].up, input_data, 
			ir_output_mr[i][p_h][p_w].up_range.w*ir_output_mr[i][p_h][p_w].up_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output_mr[i][p_h][p_w].up_range.w*ir_output_mr[i][p_h][p_w].up_range.h*net.layers[i].out_c;
      }
      if(result_left_mr[p_h][p_w][i]==1){
		ir_output_mr[i][p_h][p_w].left = (float*)malloc(ir_output_mr[i][p_h][p_w].left_range.w*ir_output_mr[i][p_h][p_w].left_range.h*net.layers[i].out_c*sizeof(float));
		memcpy(ir_output_mr[i][p_h][p_w].left, input_data, 
			ir_output_mr[i][p_h][p_w].left_range.w*ir_output_mr[i][p_h][p_w].left_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output_mr[i][p_h][p_w].left_range.w*ir_output_mr[i][p_h][p_w].left_range.h*net.layers[i].out_c;
      }
      if(result_down_mr[p_h][p_w][i]==1){
		ir_output_mr[i][p_h][p_w].down = (float*)malloc(ir_output_mr[i][p_h][p_w].down_range.w*ir_output_mr[i][p_h][p_w].down_range.h*net.layers[i].out_c*sizeof(float));
		memcpy(ir_output_mr[i][p_h][p_w].down, input_data, 
			ir_output_mr[i][p_h][p_w].down_range.w*ir_output_mr[i][p_h][p_w].down_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output_mr[i][p_h][p_w].down_range.w*ir_output_mr[i][p_h][p_w].down_range.h*net.layers[i].out_c;
      }
      if(result_right_mr[p_h][p_w][i]==1){
		ir_output_mr[i][p_h][p_w].right = (float*)malloc(ir_output_mr[i][p_h][p_w].right_range.w*ir_output_mr[i][p_h][p_w].right_range.h*net.layers[i].out_c*sizeof(float));
		memcpy(ir_output_mr[i][p_h][p_w].right, input_data, 
			ir_output_mr[i][p_h][p_w].right_range.w*ir_output_mr[i][p_h][p_w].right_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output_mr[i][p_h][p_w].right_range.w*ir_output_mr[i][p_h][p_w].right_range.h*net.layers[i].out_c;
      }

      for(int corner_number = 0; corner_number < 4; corner_number ++){
	 if(result_corners_mr[corner_number][p_h][p_w][i] > 0 ){
		ir_output_mr[i][p_h][p_w].corner_mr[corner_number] = 
	  (float*)malloc(ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].w*ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].h*net.layers[i].out_c*sizeof(float));
		memcpy(ir_output_mr[i][p_h][p_w].corner_mr[corner_number], input_data, 
			ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].w*ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].w*ir_output_mr[i][p_h][p_w].corner_range_mr[corner_number].h*net.layers[i].out_c;
 	 }
      }


}



inline float* req_ir_data_serialization_mr(network net, int part, int i){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;
      float *output;
      output = (float*)malloc(req_ir_data_size_mr[p_h][p_w][i]*sizeof(float));
      //std::cout  << "The total number of bytes in this partition ... "<< req_ir_data_size_mr[p_h][p_w][i] << std::endl;
      //int profile_num = 0;

      int prev = i - 1;
      //std::cout << "===================At layer: " << i  << std::endl;
      if(up_mr[p_h][p_w][i]==1){
		//std::cout << "Serializing the above block of the " << std::endl;
		memcpy(output, ir_output_mr[prev][p_h-1][p_w].down, 
			ir_output_mr[prev][p_h-1][p_w].down_range.w*ir_output_mr[prev][p_h-1][p_w].down_range.h*net.layers[prev].out_c*sizeof(float) ); 
		output = output + ir_output_mr[prev][p_h-1][p_w].down_range.w*ir_output_mr[prev][p_h-1][p_w].down_range.h*net.layers[prev].out_c;
      }
      if(left_mr[p_h][p_w][i]==1){
		//std::cout << "Serializing the left block of the " << std::endl;
		memcpy(output, ir_output_mr[prev][p_h][p_w-1].right, 
			ir_output_mr[prev][p_h][p_w-1].right_range.w*ir_output_mr[prev][p_h][p_w-1].right_range.h*net.layers[prev].out_c*sizeof(float) ); 
		output = output + ir_output_mr[prev][p_h][p_w-1].right_range.w*ir_output_mr[prev][p_h][p_w-1].right_range.h*net.layers[prev].out_c;

      }
      if(down_mr[p_h][p_w][i]==1){
		//std::cout << "Serializing the down block of the " << std::endl;
		//print_subindex(ir_output_mr[prev][p_h+1][p_w].up_range);
		memcpy(output, ir_output_mr[prev][p_h+1][p_w].up, 
			ir_output_mr[prev][p_h+1][p_w].up_range.w*ir_output_mr[prev][p_h+1][p_w].up_range.h*net.layers[prev].out_c*sizeof(float) ); 
		output = output + ir_output_mr[prev][p_h+1][p_w].up_range.w*ir_output_mr[prev][p_h+1][p_w].up_range.h*net.layers[prev].out_c;
      }
      if(right_mr[p_h][p_w][i]==1){
		//std::cout << "Serializing the right block of the " << std::endl;
		//print_subindex(ir_output_mr[prev][p_h][p_w+1].left_range);
		memcpy(output, ir_output_mr[prev][p_h][p_w+1].left, 
			ir_output_mr[prev][p_h][p_w+1].left_range.w*ir_output_mr[prev][p_h][p_w+1].left_range.h*net.layers[prev].out_c*sizeof(float) ); 
		output = output + ir_output_mr[prev][p_h][p_w+1].left_range.w*ir_output_mr[prev][p_h][p_w+1].left_range.h*net.layers[prev].out_c;

      }
      for(int corner_number = 0; corner_number < 4; corner_number ++){
	 if(corners_mr[corner_number][p_h][p_w][i] > 0 ){
		int reuse_h;
		int reuse_w;
		if(corner_number == 0){reuse_h = p_h - 1; reuse_w = p_w - 1;}
		if(corner_number == 1){reuse_h = p_h - 1; reuse_w = p_w + 1;}
		if(corner_number == 2){reuse_h = p_h + 1; reuse_w = p_w - 1;}
		if(corner_number == 3){reuse_h = p_h + 1; reuse_w = p_w + 1;}
		unsigned int copy_size = ir_output_mr[prev][reuse_h][reuse_w].corner_range_mr[3-corner_number].w*ir_output_mr[prev][reuse_h][reuse_w].corner_range_mr[3-corner_number].h*net.layers[prev].out_c ;
		memcpy(output, ir_output_mr[prev][reuse_h][reuse_w].corner_mr[3-corner_number], copy_size*sizeof(float) ); 
		output = output + copy_size;
 	 }
      }

      return (output - req_ir_data_size_mr[p_h][p_w][i]);
}





inline void req_ir_data_deserialization_mr(network net, int part, float* input, int i){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;
      float* input_data = input;

      int prev =  i -1;
      unsigned int copy_size;
      if(up_mr[p_h][p_w][i]==1){
	copy_size = ir_output_mr[prev][p_h-1][p_w].down_range.w*ir_output_mr[prev][p_h-1][p_w].down_range.h*net.layers[prev].out_c;
	ir_output_mr[prev][p_h-1][p_w].down = (float*)malloc(copy_size*sizeof(float));
	memcpy(ir_output_mr[prev][p_h-1][p_w].down, input_data, copy_size*sizeof(float) ); 
	input_data = input_data + copy_size;
      }
      if(left_mr[p_h][p_w][i]==1){
	copy_size = ir_output_mr[prev][p_h][p_w-1].right_range.w*ir_output_mr[prev][p_h][p_w-1].right_range.h*net.layers[prev].out_c; 
	ir_output_mr[prev][p_h][p_w-1].right = (float*)malloc(copy_size*sizeof(float));
	memcpy(ir_output_mr[prev][p_h][p_w-1].right, input_data, copy_size*sizeof(float) ); 
	input_data = input_data + copy_size;
      }
      if(down_mr[p_h][p_w][i]==1){
	copy_size = ir_output_mr[prev][p_h+1][p_w].up_range.w*ir_output_mr[prev][p_h+1][p_w].up_range.h*net.layers[prev].out_c;
	ir_output_mr[prev][p_h+1][p_w].up = (float*)malloc(copy_size*sizeof(float));
	memcpy(ir_output_mr[prev][p_h+1][p_w].up, input_data, copy_size*sizeof(float) ); 
	input_data = input_data + copy_size;
      }
      if(right_mr[p_h][p_w][i]==1){
	copy_size = ir_output_mr[prev][p_h][p_w+1].left_range.w*ir_output_mr[prev][p_h][p_w+1].left_range.h*net.layers[prev].out_c;
	ir_output_mr[prev][p_h][p_w+1].left = (float*)malloc(copy_size*sizeof(float));
	memcpy(ir_output_mr[prev][p_h][p_w+1].left, input_data, copy_size*sizeof(float) ); 
	input_data = input_data + copy_size;
      }
      for(int corner_number = 0; corner_number < 4; corner_number ++){
	 if(corners_mr[corner_number][p_h][p_w][i] > 0 ){
	   int reuse_h;
	   int reuse_w;
	   if(corner_number == 0){reuse_h = p_h - 1; reuse_w = p_w - 1;}
	   if(corner_number == 1){reuse_h = p_h - 1; reuse_w = p_w + 1;}
	   if(corner_number == 2){reuse_h = p_h + 1; reuse_w = p_w - 1;}
	   if(corner_number == 3){reuse_h = p_h + 1; reuse_w = p_w + 1;}
	   copy_size = 
	ir_output_mr[prev][reuse_h][reuse_w].corner_range_mr[3-corner_number].w*ir_output_mr[prev][reuse_h][reuse_w].corner_range_mr[3-corner_number].h*net.layers[prev].out_c ;
	   memcpy(ir_output_mr[prev][reuse_h][reuse_w].corner_mr[3-corner_number], input_data, copy_size*sizeof(float) ); 
	   input_data = input_data + copy_size;
 	 }
      }


}





#endif
