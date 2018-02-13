#include "darknet_util.h"
#include "reuse_data.h"
#include "distriot.h"
void write_layer_test(network *netp, int idx)
{
    network net = *netp;
    layer l = net.layers[idx];
    FILE *p_file;

    char filename[50];
    sprintf(filename, "%s_%d.dat", get_layer_string(net.layers[idx + 1].type), idx + 1);
    p_file = fopen(filename, "wb");
 
    fwrite(l.output, sizeof(float), l.outputs, p_file); 
    fclose(p_file);
}

void read_layer_test(network *netp, int idx)
{
    network net = *netp;
    layer l = net.layers[idx];
    FILE *p_file;

    char filename[50];
    sprintf(filename, "%s_%d.dat", get_layer_string(net.layers[idx].type), idx);
    p_file = fopen(filename, "rb");
    fread(net.input, sizeof(float), l.inputs, p_file); 
    fclose(p_file);

}

//input(w*h) [dh1, dh2]    copy into ==> output  [0, dh2 - dh1]
//	     [dw1, dw2]			         [0, dw2 - dw1]

float* reshape_input(float* input, int w, int h, int c, int dw1, int dw2, int dh1, int dh2){

   int out_w = dw2 - dw1 + 1;
   int out_h = dh2 - dh1 + 1;
   int i,j,k;
   float* output = (float*) malloc( sizeof(float)*out_w*out_h*c );  
   for(k = 0; k < c; ++k){
     for(j = dh1; j < dh2+1; ++j){
       for(i = dw1; i < dw2+1; ++i){
           int in_index  = i + w*(j + h*k);
           int out_index  = (i - dw1) + out_w*(j - dh1) + out_w*out_h*k;
	   output[out_index] = input[in_index];
       }
     }
   }
   return output;

}

//input [0, dh2 - dh1]    copy into ==> output(w*h)   [dh1, dh2]
//	[0, dw2 - dw1]			              [dw1, dw2]

void reshape_output(float* input, float* output, int w, int h, int c, int dw1, int dw2, int dh1, int dh2){

   int in_w = dw2 - dw1 + 1;
   int in_h = dh2 - dh1 + 1;
   int i,j,k;

   for(k = 0; k < c; ++k){
     for(j = 0; j < in_h; ++j){
       for(i = 0; i < in_w; ++i){
           int in_index  = i + in_w*(j + in_h*k);
           int out_index  = (i + dw1) + w*(j + dh1) + w*h*k;
	   output[out_index] = input[in_index];
       }
     }
   }

}
//
//input [0, dh2 - dh1]    copy into ==> output(w*h)   [dh1, dh2]
//	[0, dw2 - dw1]			              [dw1, dw2]

void copy_input_to_output(float* input, float* output, int w, int h, int c, int dw1, int dw2, int dh1, int dh2){

   int in_w = dw2 - dw1 + 1;
   int in_h = dh2 - dh1 + 1;
   int i,j,k;

   for(k = 0; k < c; ++k){
     for(j = 0; j < in_h; ++j){
       for(i = 0; i < in_w; ++i){
           int in_index  = i + in_w*(j + in_h*k);
           int out_index  = (i + dw1) + w*(j + dh1) + w*h*k;
	   output[out_index] = input[in_index];
       }
     }
   }

}


void print_array(char* filename, float* stage_out, int stage_outs, int line){
    FILE *layerfile;
    char layerdata[30];
    int ii;
    sprintf(layerdata, filename);
    layerfile = fopen(layerdata, "w");  
    for(ii = 0; ii < stage_outs; ii++){
       fprintf(layerfile, "%.1f", stage_out[ii]);
       if((ii+1)%line == 0) fprintf(layerfile, "\n");
    }
    fclose(layerfile);
}







sub_index calculate_range(sub_index output, layer l){
    sub_index input; 
    if(l.type == CONVOLUTIONAL){
       input.w1 = (output.w1*l.stride - l.size/2)>0   ? (output.w1*l.stride - l.size/2) : 0;
       input.w2 = (output.w2*l.stride + l.size/2)<(l.w-1) ? (output.w2*l.stride + l.size/2) : (l.w-1);
       input.h1 = (output.h1*l.stride - l.size/2)>0   ? (output.h1*l.stride - l.size/2) : 0;
       input.h2 = (output.h2*l.stride + l.size/2)<(l.h-1) ? (output.h2*l.stride + l.size/2) : (l.h-1);
    }else if(l.type == MAXPOOL){
       input.w1 = output.w1*l.stride;
       input.w2 = output.w2*l.stride + l.stride -1;
       input.h1 = output.h1*l.stride;
       input.h2 = output.h2*l.stride + l.stride -1;
    }

    input.w = input.w2 -input.w1 + 1;
    input.h = input.h2 -input.h1 + 1;

    return input;
}





sub_index calculate_layeroutput_range(sub_index input, layer l){
    sub_index output; 
    if(l.type == CONVOLUTIONAL){
	if((input.w1) > 0) {input.w1 = input.w1 + l.size/2;}
	if((input.w2) < l.w-1) {input.w2 = input.w2 - l.size/2;} 
	output.w1 = input.w1/l.stride; 
	output.w2 = input.w2/l.stride;
	if((input.h1) > 0) {input.h1 = input.h1 + l.size/2;}
	if((input.h2) < l.h-1) {input.h2 = input.h2 - l.size/2;} 
	output.h1 = input.h1/l.stride; 
	output.h2 = input.h2/l.stride;
    }else if(l.type == MAXPOOL){
	output.w1 = input.w1/l.stride; 
	output.w2 = input.w2/l.stride;
	output.h1 = input.h1/l.stride; 
	output.h2 = input.h2/l.stride;
    }

    output.w = output.w2 -output.w1 + 1;
    output.h = output.h2 -output.h1 + 1;

    return output;
}


sub_index crop_ranges(sub_index large, sub_index small){

    sub_index output; 
    output.w1 = small.w1 - large.w1 ; 
    output.w2 = small.w1 - large.w1 + (small.w2 - small.w1);
    output.h1 = small.h1 - large.h1; 
    output.h2 = small.h1 - large.h1 + (small.h2 - small.h1);
    

    output.w = output.w2 -output.w1 + 1;
    output.h = output.h2 -output.h1 + 1;
    return output;
}


inline void stage_output_partition(int w1, int w2, int h1, int h2){
    int w = w2 - w1 + 1;
    int h = h2 - h1 + 1;
    int partition_w = PARTITIONS_W;
    int partition_h = PARTITIONS_H;

    int stride_w = ceil(((float)w)/((float)partition_w));    
    int start_w = 0;
    int end_w = stride_w - 1;

    int stride_h = ceil(((float)h)/((float)partition_h));    
    int start_h = 0;
    int end_h = stride_h - 1;

    for(int i = 0; i < partition_h; i++){
       start_w = 0;
       end_w = stride_w - 1;	 
       for(int j = 0; j < partition_w; j++){
	   stage_output_partition_ranges[part_id[i][j]].w1 = start_w;
	   stage_output_partition_ranges[part_id[i][j]].w2 = end_w;
	   stage_output_partition_ranges[part_id[i][j]].h1 = start_h;
	   stage_output_partition_ranges[part_id[i][j]].h2 = end_h;
	   stage_output_partition_ranges[part_id[i][j]].h = end_h - start_h + 1;
	   stage_output_partition_ranges[part_id[i][j]].w = end_w - start_w + 1;
	   start_w = end_w + 1;
	   if(j == (partition_w-2))
	       end_w = w - 1;
	   else
	       end_w = end_w + stride_w; 	 
       }
       start_h = end_h + 1;
       if(i == (partition_h-2))
	       end_h = h - 1;
       else
	       end_h = end_h + stride_h; 
    }
}


inline network reshape_network(int startfrom, int upto, network net){
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


//Calculate reuse
    for(p_h = 0; p_h < partition_h; p_h++){
	for(p_w = 0; p_w < partition_w; p_w++){ 
	    for(i = upto; i > 0; i--){
		cal_reuse_overlap_range(p_h, p_w, i-1, &reuse_output_ranges[0], reuse_input_ranges[part_id[p_h][p_w]][i] );
	    }
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


inline network forward_stage_reuse(int p_h, int p_w, float *input,int startfrom, int upto,  network net){
	int part = part_id[p_h][p_w];
        std::cout << "==========Begin=============: " << p_h<<"   "<<p_w<< std::endl;

	//Reshape first
	net.input = input;
	for(int i = startfrom; i < (upto+1); ++i){
	    net.layers[i].h = (reuse_input_ranges[part][i].h2 - reuse_input_ranges[part][i].h1 + 1); net.layers[i].out_h = (net.layers[i].h/net.layers[i].stride); 
	    net.layers[i].w = (reuse_input_ranges[part][i].w2 - reuse_input_ranges[part][i].w1 + 1); net.layers[i].out_w = (net.layers[i].w/net.layers[i].stride); 
	    net.layers[i].outputs = net.layers[i].out_h * net.layers[i].out_w * net.layers[i].out_c; 
	    net.layers[i].inputs = net.layers[i].h * net.layers[i].w * net.layers[i].c; 
	}


    	for(int i = startfrom; i < upto+1; i++){
		std::cout << "-----------At layer----------: " << i << std::endl;
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




		//What should we record for the current layer?
		if((ir_output[i][p_h][p_w].down_range.w>0)&&(ir_output[i][p_h][p_w].down_range.h>0)){
			//std::cout << "Down: " << std::endl;
			//print_subindex(ir_output[i][p_h][p_w].down_range);
			sub_index down_index = ir_output[i][p_h][p_w].down_range;
			down_index.w1 -= reuse_output_ranges[part][i].w1;
			down_index.w2 -= reuse_output_ranges[part][i].w1;
			down_index.h1 -= reuse_output_ranges[part][i].h1;
			down_index.h2 -= reuse_output_ranges[part][i].h1;
			print_subindex(down_index);
			ir_output[i][p_h][p_w].down = reshape_input(cropped_output, reuse_output_ranges[part][i].w, reuse_output_ranges[part][i].h, net.layers[i].out_c, 
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
			print_subindex(right_index);
			ir_output[i][p_h][p_w].right = reshape_input(cropped_output, reuse_output_ranges[part][i].w, reuse_output_ranges[part][i].h, net.layers[i].out_c, 
							right_index.w1, right_index.w2, 
							right_index.h1, right_index.h2);

		}
		if((ir_output[i][p_h][p_w].corner_range.w>0)&&(ir_output[i][p_h][p_w].corner_range.h>0)) {
			//std::cout << "Corner: " << std::endl;
			//print_subindex(ir_output[i][p_h][p_w].corner_range);

			sub_index corner_index = ir_output[i][p_h][p_w].corner_range;
			corner_index.w1 -= reuse_output_ranges[part][i].w1;
			corner_index.w2 -= reuse_output_ranges[part][i].w1;
			corner_index.h1 -= reuse_output_ranges[part][i].h1;
			corner_index.h2 -= reuse_output_ranges[part][i].h1;
			print_subindex(corner_index);
			ir_output[i][p_h][p_w].corner = reshape_input(cropped_output, reuse_output_ranges[part][i].w, reuse_output_ranges[part][i].h, net.layers[i].out_c, 
							corner_index.w1, corner_index.w2, 
							corner_index.h1, corner_index.h2);

		}




		int up = 0;
		int corner = 0;
		int left = 0;		
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
			if((p_h>0)&&(p_w>0)&&(ir_output[i][p_h-1][p_w-1].corner_range.w>0)&&(ir_output[i][p_h-1][p_w-1].corner_range.h>0)) {
				//std::cout << "Require input from left above part in this layer: " << std::endl;
				corner = 1;
				//print_subindex(ir_output[i][p_h-1][p_w-1].corner_range);
			}

		  }
                }

		std::cout<< ".....................OK, let us do it then....................." <<std::endl;
		if(up == 1 && corner == 0){
			//std::cout << "Input for next layer is: "<< std::endl;
			//print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i+1]);
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
				
			//print_subindex(main_index);
			//print_subindex(up_index);
                        copy_input_to_output(ir_output[i][p_h-1][p_w].down, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, up_index.w1, up_index.w2, up_index.h1, up_index.h2);
                        copy_input_to_output(cropped_output, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, main_index.w1, main_index.w2, main_index.h1, main_index.h2);


		}else if(left == 1 && corner == 0){
			//std::cout << "Input for next layer is: "<< std::endl;
			//print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i+1]);
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
				
			//print_subindex(main_index);
			//print_subindex(left_index);
                        copy_input_to_output(ir_output[i][p_h][p_w-1].right, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, left_index.w1, left_index.w2, left_index.h1, left_index.h2);
                        copy_input_to_output(cropped_output, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, main_index.w1, main_index.w2, main_index.h1, main_index.h2);

          


		}else if(corner == 1){
			//std::cout << "Input for next layer is: "<< std::endl;
			//print_subindex(reuse_input_ranges[part_id[p_h][p_w]][i+1]);
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


			//print_subindex(main_index);
			//print_subindex(left_index);
			//print_subindex(up_index);
			//print_subindex(corner_index);

                        copy_input_to_output(ir_output[i][p_h-1][p_w-1].corner, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, corner_index.w1, corner_index.w2, corner_index.h1, corner_index.h2);
                        copy_input_to_output(ir_output[i][p_h][p_w-1].right, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, left_index.w1, left_index.w2, left_index.h1, left_index.h2);
                        copy_input_to_output(ir_output[i][p_h-1][p_w].down, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, up_index.w1, up_index.w2, up_index.h1, up_index.h2);
                        copy_input_to_output(cropped_output, net.input, reuse_input_ranges[part_id[p_h][p_w]][i+1].w, reuse_input_ranges[part_id[p_h][p_w]][i+1].h, 
						net.layers[i].out_c, main_index.w1, main_index.w2, main_index.h1, main_index.h2);

		}else {net.input = cropped_output;}

	}
        std::cout << "==========Finish=============: " << part_id[p_h][p_w] << std::endl;

	return net;

}

void fork_input(int startfrom, float* stage_in, network net){

    int part;
    //Prepare the input data for each partition   
    for(part = 0; part < PARTITIONS; part ++) { 
      part_data[part] = reshape_input(stage_in, stage_input_range.w, stage_input_range.h, net.layers[startfrom].c, 
					input_ranges[part][startfrom].w1, input_ranges[part][startfrom].w2, 
					input_ranges[part][startfrom].h1, input_ranges[part][startfrom].h2);
			 //printf("%2d %4d %4d %4d %4d %4d %4d\n", (stage_input_range.w2 - stage_input_range.w1 + 1), (stage_input_range.h2 - stage_input_range.h1 + 1), net.layers[i].c, 
			 //		input_ranges[p][i].w1, input_ranges[p][i].w2, input_ranges[p][i].h1, input_ranges[p][i].h2);
    }

}


void fork_input_reuse(int startfrom, float* stage_in, network net){

    int part;
    //Prepare the input data for each partition   
    for(part = 0; part < PARTITIONS; part ++) { 
      part_data[part] = reshape_input(stage_in, stage_input_range.w, stage_input_range.h, net.layers[startfrom].c, 
					reuse_input_ranges[part][startfrom].w1, reuse_input_ranges[part][startfrom].w2, 
					reuse_input_ranges[part][startfrom].h1, reuse_input_ranges[part][startfrom].h2);
			 //printf("%2d %4d %4d %4d %4d %4d %4d\n", (stage_input_range.w2 - stage_input_range.w1 + 1), (stage_input_range.h2 - stage_input_range.h1 + 1), net.layers[i].c, 
			 //		input_ranges[p][i].w1, input_ranges[p][i].w2, input_ranges[p][i].h1, input_ranges[p][i].h2);
    }

}

void join_output(int part, float* part_data, float* stage_out, int upto, network net){
    reshape_output(part_data, stage_out, (stage_output_range.w2-stage_output_range.w1 + 1), 
			(stage_output_range.h2-stage_output_range.h1 + 1), net.layers[upto].out_c, 
			output_ranges[part][upto].w1, output_ranges[part][upto].w2,
			output_ranges[part][upto].h1, output_ranges[part][upto].h2);
}


inline network forward_stage(int part, float *input,int startfrom, int upto,  network net)
{
    net.input = input;


    //Reshape first
    for(int i = startfrom; i < (upto+1); ++i){
	    net.layers[i].h = (input_ranges[part][i].h2 - input_ranges[part][i].h1 + 1); net.layers[i].out_h = (net.layers[i].h/net.layers[i].stride); 
	    net.layers[i].w = (input_ranges[part][i].w2 - input_ranges[part][i].w1 + 1); net.layers[i].out_w = (net.layers[i].w/net.layers[i].stride); 
	    net.layers[i].outputs = net.layers[i].out_h * net.layers[i].out_w * net.layers[i].out_c; 
	    net.layers[i].inputs = net.layers[i].h * net.layers[i].w * net.layers[i].c; 
    }

    int to_free = 0;
    float * cropped_output;

    for(int i = startfrom; i < (upto+1); ++i){	    
	    net.layers[i].forward(net.layers[i], net);
	    if (to_free == 1) {
		free(cropped_output); 
		to_free = 0; //Free the memory allocated by the reshape_input function call;
	    }
	    if(net.layers[i].type == CONVOLUTIONAL){
		layer l = net.layers[i];
	        //print_subindex(input_ranges[part][i]);
	        //print_subindex(output_ranges[part][i]);
		//printf("%2d, %2d, %2d\n", l.out_w, l.out_h, l.out_c);
		//print_subindex(crop_ranges(input_ranges[part][i], output_ranges[part][i]));   
		sub_index tmp = crop_ranges(input_ranges[part][i], output_ranges[part][i]);   
		cropped_output = reshape_input(net.layers[i].output, l.out_w, l.out_h, l.out_c,  tmp.w1, tmp.w2, tmp.h1, tmp.h2);
		to_free = 1;
	    } else {cropped_output = net.layers[i].output;}  

	    net.input = cropped_output;

	    //if(net.layers[i].truth) {
		//    net.truth = net.layers[i].output;
	    //}
    }
    if (to_free == 1) free(net.input);
    //

    return net; 
}

inline void forward_network_dist(network *netp, network orig)
{
    int part;
    network net = *netp;

    int startfrom = 0;
    int upto = STAGES-1;

    size_t stage_outs =  (stage_output_range.w)*(stage_output_range.h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
    float* stage_in = net.input; 

    fork_input_reuse(startfrom, stage_in, net);

    for(part = 0; part < PARTITIONS; part ++){
      printf("Putting jobs %d\n", part);
      put_job(part_data[part], input_ranges[part][startfrom].w*input_ranges[part][startfrom].h*net.layers[startfrom].c*sizeof(float), part);
    }

    float* data;
    int part_id;
    unsigned int size;

    double t1 = 0.0;
    double t0 = get_real_time_now();

    for(part = 0; 1; part ++){
       try_get_job((void**)&data, &size, &part_id);
       if(data == NULL) {printf("%d parts out of the %d are processes locally\n", part, PARTITIONS); break;}
       std::cout << "=======================: " << part_id << std::endl;
       //net = forward_stage( part_id, data, startfrom, upto, net);
       if(part_id == 0) net = forward_stage_reuse( 0, 0, data, startfrom, upto, net);
       if(part_id == 1) net = forward_stage_reuse( 0, 1, data, startfrom, upto, net);
       if(part_id == 2) net = forward_stage_reuse( 1, 0, data, startfrom, upto, net);
       if(part_id == 3) net = forward_stage_reuse( 1, 1, data, startfrom, upto, net);
       join_output(part_id, net.layers[upto].output,  stage_out, upto, net);
       free(data);
    }

    for(part = part; part < PARTITIONS; part ++){
       get_result((void**)&data, &size, &part_id);
       printf("Getting result %d from other stealers\n", part_id);
       join_output(part_id, data,  stage_out, upto, net);
       free(data);
    }
    t1 = t1 + get_real_time_now() - t0;
    std::cout << "Processing overhead is: " << t1 << std::endl;


    net.input = stage_out;

    for(int i = (upto + 1); i < net.n; ++i){ //Iteratively execute the layers
        net.index = i;
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }
        net.layers[i].forward(net.layers[i], net);
        net.input = net.layers[i].output; //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
    }
    free(stage_out);
}

inline void forward_network_dist_gateway(network *netp, network orig)
{
    int part;
    network net = *netp;

    int startfrom = 0;
    int upto = 7;

    size_t stage_outs =  (stage_output_range.w)*(stage_output_range.h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
    float* stage_in = net.input; 

    fork_input(startfrom, stage_in, net);
    char reg[10] = "register";



    for(part = 0; part < PARTITIONS; part ++){
      printf("Putting jobs %d\n", part);
      put_job(part_data[part], input_ranges[part][startfrom].w*input_ranges[part][startfrom].h*net.layers[startfrom].c*sizeof(float), part);
    }
    ask_gateway(reg, AP, SMART_GATEWAY); //register number of tasks


    float* data;
    int part_id;
    unsigned int size;

    for(part = 0; 1; part ++){
       try_get_job((void**)&data, &size, &part_id);
       if(data == NULL) {
	   printf("%d parts out of the %d are processes locally, yeeha!\n", part, PARTITIONS); 
	   ask_gateway(reg, AP, SMART_GATEWAY); //remove the registration when we are running out of tasks
	   break;
       }
       net = forward_stage( part_id, data, startfrom, upto, net);
       join_output(part_id, net.layers[upto].output,  stage_out, upto, net);
       free(data);
    }

    for(part = part; part < PARTITIONS; part ++){
       get_result((void**)&data, &size, &part_id);
       //printf("Getting result %d from other stealers\n", part_id);
       join_output(part_id, data,  stage_out, upto, net);
       free(data);
    }

    g_t1  =  what_time_is_it_now() - g_t0;
    //std::cout << "At time " << g_t1 << ", get task " << part_id << " from res_queue" <<std::endl;  
    std::cout << "At time " << g_t1 << " stage1 is finished" <<std::endl;  

    net.input = stage_out;

    for(int i = (upto + 1); i < net.n; ++i){ //Iteratively execute the layers
        net.index = i;
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }
        net.layers[i].forward(net.layers[i], net);
        net.input = net.layers[i].output; //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
    }
    g_t1  =  what_time_is_it_now() - g_t0;
    std::cout << "At time " << g_t1 << " stage2 is finished" <<std::endl;  

    free(stage_out);

}


inline void forward_network_dist_prof(network *netp)
{
    network net = *netp;
    int i;
    double t0 = what_time_is_it_now();
    double t1 = 0;

//  FILE *layer_input;
//  FILE *layer_output;
//  FILE *layer_weight; 
//
//  layer_input  = fopen("layer_input.log", "w"); 
//  layer_output = fopen("layer_output.log", "w");  
//  layer_weight = fopen("layer_weight.log", "w");

    for(i = 0; i < net.n; ++i){//Iteratively execute the layers
        net.index = i;
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }

        //fprintf(layer_input, "%f\n", (float)(net.layers[i].inputs*sizeof(float))/1024.0/1024.0 );
        //fprintf(layer_output, "%f\n", (float)(net.layers[i].outputs*sizeof(float))/1024.0/1024.0 );

        //if(net.layers[i].type == CONNECTED)
          // fprintf(layer_weight, "%f\n", (float)(net.layers[i].outputs*net.layers[i].inputs*sizeof(float))/1024.0/1024.0 );
	//else
          // fprintf(layer_weight, "%f\n", (float)(net.layers[i].nweights*sizeof(float))/1024.0/1024.0 );


        net.layers[i].forward(net.layers[i], net);
        net.input = net.layers[i].output;  //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
        printf("Index %d, Layer %s, input data byte num is: %ld, output data byte num is: %ld\n", 
		i, get_layer_string(net.layers[i].type), net.layers[i].inputs*sizeof(float), net.layers[i].outputs*sizeof(float));
    }

//    fclose(layer_input);
//    fclose(layer_weight);
//    fclose(layer_output);
}

inline float *network_predict_dist_prof(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    //forward_network_dist_prof(net);
    forward_network_dist(net, orig);
    //forward_network_dist_gateway(net, orig);
    float *out = net->output;
    *net = orig;
    return out;
}

inline float *network_predict_dist(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    //forward_network_dist_prof(net);
    //forward_network_dist(net, orig);
    forward_network_dist_gateway(net, orig);
    float *out = net->output;
    *net = orig;
    return out;
}
