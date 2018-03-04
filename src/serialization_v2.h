#include "darknet_util.h"

#ifndef SERIALIZATION_V2__H
#define SERIALIZATION_V2__H

inline float* req_ir_data_serialization_v2(network net, int part, int startfrom, int upto, bool* req, unsigned int *size){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;
      float *output;
      output = (float*)malloc(ir_data_size[part]*sizeof(float));
      //std::cout  << "The total number of bytes in this partition ... "<< ir_data_size[part] << std::endl;
      //int profile_num = 0;
      *size = 0;

      for(int i = startfrom; i < upto; i ++){
	//std::cout << "===================At layer: " << i  << std::endl;
	if(up[part][i]==1&&req[0]){
		//std::cout << "Serializing the above block of the " << std::endl;
		//print_subindex(ir_output[i][p_h-1][p_w].down_range);
		memcpy(output, ir_output[i][p_h-1][p_w].down, ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c;
		//profile_num += ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c;
		//std::cout  << "Shift number is ... "<< profile_num << std::endl;
		*size = *size + ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c;
	}
	if(left[part][i]==1&&req[1]){
		//std::cout << "Serializing the left block of the " << std::endl;
		//print_subindex(ir_output[i][p_h][p_w-1].right_range);
		memcpy(output, ir_output[i][p_h][p_w-1].right, ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c;
		//profile_num +=  ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c;
		//std::cout  << "Shift number is ... "<< profile_num << std::endl;
		*size = *size + ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c;
	}
	if(down[part][i]==1&&req[2]){
		//std::cout << "Serializing the down block of the " << std::endl;
		//print_subindex(ir_output[i][p_h+1][p_w].up_range);
		memcpy(output, ir_output[i][p_h+1][p_w].up, ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c;
		//profile_num +=  ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c;
		//std::cout  << "Shift number is ... "<< profile_num << std::endl;
		*size = *size + ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c;
	}
	if(right[part][i]==1&&req[3]){
		//std::cout << "Serializing the right block of the " << std::endl;
		//print_subindex(ir_output[i][p_h][p_w+1].left_range);
		memcpy(output, ir_output[i][p_h][p_w+1].left, ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c;
		//profile_num +=  ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c;
		//std::cout  << "Shift number is ... "<< profile_num << std::endl;
		*size = *size + ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c;
	}
	//std::cout << "===================At layer: " << i  << std::endl;
      }
      
      unsigned int offset = *size;

      *size = (*size) * sizeof(float);
      std::cout << "The size of IR data required to be transmitted is: " << *size << std::endl;
      return (output - offset);
}


inline void req_ir_data_deserialization_v2(network net, int part, float* input, int startfrom, int upto, bool* req){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;

      float* input_data = input;
      for(int i = startfrom; i < upto; ++i){
	if(up[part][i]==1&&req[0]){
	   ir_output[i][p_h-1][p_w].down = (float*)malloc(   ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c*sizeof(float));
	   memcpy(ir_output[i][p_h-1][p_w].down, input_data, ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c*sizeof(float) ); 
	   input_data = input_data + ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c;
	}
	if(left[part][i]==1&&req[1]){
	   ir_output[i][p_h][p_w-1].right = (float*)malloc(   ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c*sizeof(float));
	   memcpy(ir_output[i][p_h][p_w-1].right, input_data, ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c*sizeof(float) ); 
	   input_data = input_data + ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c;
	}
	if(down[part][i]==1&&req[2]){
	   ir_output[i][p_h+1][p_w].up = (float*)malloc(   ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c*sizeof(float));
	   memcpy(ir_output[i][p_h+1][p_w].up, input_data, ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c*sizeof(float) ); 
	   input_data = input_data + ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c;
	}
	if(right[part][i]==1&&req[3]){
	   ir_output[i][p_h][p_w+1].left = (float*)malloc(   ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c*sizeof(float));
	   memcpy(ir_output[i][p_h][p_w+1].left, input_data, ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c*sizeof(float) ); 
	   input_data = input_data + ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c;
	}
      }


}



#endif
