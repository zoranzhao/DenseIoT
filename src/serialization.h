#include "darknet_util.h"

#ifndef SERIALIZATION__H
#define SERIALIZATION__H



inline void ir_data_spatial_dependency(network net, int startfrom, int upto){
      for(int part = 0; part < PARTITIONS; part ++){
         ir_data_size[part] = 0;
         for(int i = startfrom; i < upto; i ++){
		up[part][i] = 0;
		left[part][i] = 0;
		right[part][i] = 0;
		down[part][i] = 0;
         }
      }

      for(int part = 0; part < PARTITIONS; part ++){
        int p_h = part / PARTITIONS_W; 
        int p_w = part % PARTITIONS_W;
        for(int i = startfrom; i < upto; i ++){
	   if(net.layers[i+1].type == CONVOLUTIONAL){
	      if((p_h>0)&&(ir_output[i][p_h-1][p_w].down_range.w>0)&&(ir_output[i][p_h-1][p_w].down_range.h>0)){
		up[part][i] = 1;
	      }
	      if((p_w>0)&&(ir_output[i][p_h][p_w-1].right_range.w>0)&&(ir_output[i][p_h][p_w-1].right_range.h>0)){
		left[part][i] = 1;
	      }
	      if(((p_h+1)<PARTITIONS_H)&&(ir_output[i][p_h+1][p_w].up_range.w>0)&&(ir_output[i][p_h+1][p_w].up_range.h>0)){
		down[part][i] = 1;
	      }
	      if(((p_w+1)<PARTITIONS_W)&&(ir_output[i][p_h][p_w+1].left_range.w>0)&&(ir_output[i][p_h][p_w+1].left_range.h>0)){
		right[part][i] = 1;
	      }
	   }
        }
      }

      for(int part = 0; part < PARTITIONS; part ++){
        int p_h = part / PARTITIONS_W; 
        int p_w = part % PARTITIONS_W;
        for(int i = startfrom; i < upto; i ++){
		if(up[part][i]==1){
			ir_data_size[part] = ir_data_size[part] + (ir_output[i][p_h-1][p_w].down_range.w)*(ir_output[i][p_h-1][p_w].down_range.h)*net.layers[i].out_c;  
		}
		if(left[part][i]==1){
			ir_data_size[part] = ir_data_size[part] + (ir_output[i][p_h][p_w-1].right_range.w)*(ir_output[i][p_h][p_w-1].right_range.h)*net.layers[i].out_c;  
		}
		if(down[part][i]==1){
			ir_data_size[part] = ir_data_size[part] + (ir_output[i][p_h+1][p_w].up_range.w)*(ir_output[i][p_h+1][p_w].up_range.h)*net.layers[i].out_c;  
		}
		if(right[part][i]==1){
			ir_data_size[part] = ir_data_size[part] + (ir_output[i][p_h][p_w+1].left_range.w)*(ir_output[i][p_h][p_w+1].left_range.h)*net.layers[i].out_c;  
		}
        }
      }

}



inline void result_ir_data_spatial_dependency(network net, int startfrom, int upto){
      for(int part = 0; part < PARTITIONS; part ++){
         result_ir_data_size[part] = 0;
         for(int i = startfrom; i < upto; i ++){
		result_up[part][i] = 0;
		result_left[part][i] = 0;
		result_right[part][i] = 0;
		result_down[part][i] = 0;
         }
      }

      for(int part = 0; part < PARTITIONS; part ++){
        int p_h = part / PARTITIONS_W; 
        int p_w = part % PARTITIONS_W;
        for(int i = startfrom; i < upto; i ++){
	   if(net.layers[i+1].type == CONVOLUTIONAL){
	      if((ir_output[i][p_h][p_w].down_range.w>0)&&(ir_output[i][p_h][p_w].down_range.h>0)){
		result_down[part][i] = 1;
	      }
	      if((ir_output[i][p_h][p_w].right_range.w>0)&&(ir_output[i][p_h][p_w].right_range.h>0)){
		result_right[part][i] = 1;
	      }
	      if((ir_output[i][p_h][p_w].up_range.w>0)&&(ir_output[i][p_h][p_w].up_range.h>0)){
		result_up[part][i] = 1;
	      }
	      if((ir_output[i][p_h][p_w].left_range.w>0)&&(ir_output[i][p_h][p_w].left_range.h>0)){
		result_left[part][i] = 1;
	      }
	   }
        }
      }

      for(int part = 0; part < PARTITIONS; part ++){
        int p_h = part / PARTITIONS_W; 
        int p_w = part % PARTITIONS_W;
        for(int i = startfrom; i < upto; i ++){
		if(result_up[part][i]==1){
			result_ir_data_size[part] = result_ir_data_size[part] + (ir_output[i][p_h][p_w].up_range.w)*(ir_output[i][p_h][p_w].up_range.h)*net.layers[i].out_c;  
		}
		if(result_left[part][i]==1){
			result_ir_data_size[part] = result_ir_data_size[part] + (ir_output[i][p_h][p_w].left_range.w)*(ir_output[i][p_h][p_w].left_range.h)*net.layers[i].out_c;  
		}
		if(result_down[part][i]==1){
			result_ir_data_size[part] = result_ir_data_size[part] + (ir_output[i][p_h][p_w].down_range.w)*(ir_output[i][p_h][p_w].down_range.h)*net.layers[i].out_c;  
		}
		if(result_right[part][i]==1){
			result_ir_data_size[part] = result_ir_data_size[part] + (ir_output[i][p_h][p_w].right_range.w)*(ir_output[i][p_h][p_w].right_range.h)*net.layers[i].out_c;  
		}
        }
      }

}
inline float* req_ir_data_serialization(network net, int part, int startfrom, int upto){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;
      float *output;
      output = (float*)malloc(ir_data_size[part]*sizeof(float));
      std::cout  << "The total number of bytes in this partition ... "<< ir_data_size[part] << std::endl;
      //int profile_num = 0;

      for(int i = startfrom; i < upto; i ++){
	//std::cout << "===================At layer: " << i  << std::endl;
	if(up[part][i]==1){
		//std::cout << "Serializing the above block of the " << std::endl;
		//print_subindex(ir_output[i][p_h-1][p_w].down_range);
		memcpy(output, ir_output[i][p_h-1][p_w].down, ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c;
		//profile_num += ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c;
		//std::cout  << "Shift number is ... "<< profile_num << std::endl;
	}
	if(left[part][i]==1){
		//std::cout << "Serializing the left block of the " << std::endl;
		//print_subindex(ir_output[i][p_h][p_w-1].right_range);
		memcpy(output, ir_output[i][p_h][p_w-1].right, ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c;
		//profile_num +=  ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c;
		//std::cout  << "Shift number is ... "<< profile_num << std::endl;
	}
	if(down[part][i]==1){
		//std::cout << "Serializing the down block of the " << std::endl;
		//print_subindex(ir_output[i][p_h+1][p_w].up_range);
		memcpy(output, ir_output[i][p_h+1][p_w].up, ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c;
		//profile_num +=  ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c;
		//std::cout  << "Shift number is ... "<< profile_num << std::endl;
	}
	if(right[part][i]==1){
		//std::cout << "Serializing the right block of the " << std::endl;
		//print_subindex(ir_output[i][p_h][p_w+1].left_range);
		memcpy(output, ir_output[i][p_h][p_w+1].left, ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c;
		//profile_num +=  ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c;
		//std::cout  << "Shift number is ... "<< profile_num << std::endl;
	}
	//std::cout << "===================At layer: " << i  << std::endl;
      }

      return (output - ir_data_size[part]);
}


inline void req_ir_data_deserialization(network net, int part, float* input, int startfrom, int upto){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;

      float* input_data = input;



      for(int i = startfrom; i < upto; ++i){
	if(up[part][i]==1){
		memcpy(ir_output[i][p_h-1][p_w].down, input_data, ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output[i][p_h-1][p_w].down_range.w*ir_output[i][p_h-1][p_w].down_range.h*net.layers[i].out_c;
	}
	if(left[part][i]==1){
		memcpy(ir_output[i][p_h][p_w-1].right, input_data, ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output[i][p_h][p_w-1].right_range.w*ir_output[i][p_h][p_w-1].right_range.h*net.layers[i].out_c;
	}
	if(down[part][i]==1){
		memcpy(ir_output[i][p_h+1][p_w].up, input_data, ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output[i][p_h+1][p_w].up_range.w*ir_output[i][p_h+1][p_w].up_range.h*net.layers[i].out_c;
	}
	if(right[part][i]==1){
		memcpy(ir_output[i][p_h][p_w+1].left, input_data, ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output[i][p_h][p_w+1].left_range.w*ir_output[i][p_h][p_w+1].left_range.h*net.layers[i].out_c;
	}
      }


}


inline float* result_ir_data_serialization(network net, int part, int startfrom, int upto){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;
      float *output;
      output = (float*)malloc(result_ir_data_size[part]*sizeof(float));
      std::cout  << "The total number of bytes in this partition ... "<< result_ir_data_size[part] << std::endl;
      //int profile_num = 0;

      for(int i = startfrom; i < upto; i ++){
	//std::cout << "===================At layer: " << i  << std::endl;
	if(result_up[part][i]==1){
		memcpy(output, ir_output[i][p_h][p_w].up, ir_output[i][p_h][p_w].up_range.w*ir_output[i][p_h][p_w].up_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h][p_w].up_range.w*ir_output[i][p_h][p_w].up_range.h*net.layers[i].out_c;
	}
	if(result_left[part][i]==1){
		memcpy(output, ir_output[i][p_h][p_w].left, ir_output[i][p_h][p_w].left_range.w*ir_output[i][p_h][p_w].left_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h][p_w].left_range.w*ir_output[i][p_h][p_w].left_range.h*net.layers[i].out_c;
	}
	if(result_down[part][i]==1){
		memcpy(output, ir_output[i][p_h][p_w].down, ir_output[i][p_h][p_w].down_range.w*ir_output[i][p_h][p_w].down_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h][p_w].down_range.w*ir_output[i][p_h][p_w].down_range.h*net.layers[i].out_c;
	}
	if(result_right[part][i]==1){
		memcpy(output, ir_output[i][p_h][p_w].right, ir_output[i][p_h][p_w].right_range.w*ir_output[i][p_h][p_w].right_range.h*net.layers[i].out_c*sizeof(float) ); 
		output = output + ir_output[i][p_h][p_w].right_range.w*ir_output[i][p_h][p_w].right_range.h*net.layers[i].out_c;
	}
	//std::cout << "===================At layer: " << i  << std::endl;
      }

      return (output - result_ir_data_size[part]);
}


inline void result_ir_data_deserialization(network net, int part, float* input, int startfrom, int upto){
      int p_h = part / PARTITIONS_W; 
      int p_w = part % PARTITIONS_W;

      float* input_data = input;

      for(int i = startfrom; i < upto; ++i){
	if(result_up[part][i]==1){
		memcpy(ir_output[i][p_h][p_w].up, input_data, ir_output[i][p_h][p_w].up_range.w*ir_output[i][p_h][p_w].up_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output[i][p_h][p_w].up_range.w*ir_output[i][p_h][p_w].up_range.h*net.layers[i].out_c;
	}
	if(result_left[part][i]==1){
		memcpy(ir_output[i][p_h][p_w].left, input_data, ir_output[i][p_h][p_w].left_range.w*ir_output[i][p_h][p_w].left_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output[i][p_h][p_w].left_range.w*ir_output[i][p_h][p_w].left_range.h*net.layers[i].out_c;
	}
	if(result_down[part][i]==1){
		memcpy(ir_output[i][p_h][p_w].down, input_data, ir_output[i][p_h][p_w].down_range.w*ir_output[i][p_h][p_w].down_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output[i][p_h][p_w].down_range.w*ir_output[i][p_h][p_w].down_range.h*net.layers[i].out_c;
	}
	if(result_right[part][i]==1){
		memcpy(ir_output[i][p_h][p_w].right, input_data, ir_output[i][p_h][p_w].right_range.w*ir_output[i][p_h][p_w].right_range.h*net.layers[i].out_c*sizeof(float) ); 
		input_data = input_data + ir_output[i][p_h][p_w].right_range.w*ir_output[i][p_h][p_w].right_range.h*net.layers[i].out_c;
	}
      }


}


#endif
