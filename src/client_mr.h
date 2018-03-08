#include "darknet_dist_mr.h"
void send_result_mr(dataBlob* blob, const char *dest_ip, int portno);

inline void forward_network_dist_local_mr(network *netp)
{
    network net = *netp;

    int startfrom = 0;
    int upto = STAGES-1;

  
    fork_input_mr(startfrom, net.input, net);

    size_t stage_outs =  (stage_output_range.w)*(stage_output_range.h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  


    float* data;
    int part_id;
    unsigned int size;

    double t1 = 0.0;
    double t0 = get_real_time_now();
    float* tmp[PARTITIONS];
    for(int ii = startfrom; ii < (upto + 1); ii++){
	std::cout << "At layer ... " << ii << std::endl;        
	for(part_id = 0; part_id<PARTITIONS; part_id ++){
	   std::cout << "==========Processing=============: " << part_id << std::endl;
           output_part_data_mr[part_id] = (float*)malloc(
						(input_ranges_mr[part_id][ii].w/net.layers[ii].stride)*
						(input_ranges_mr[part_id][ii].h/net.layers[ii].stride)*net.layers[ii].out_c*sizeof(float));

	   net = forward_stage_mr( part_id/PARTITIONS_W, part_id%PARTITIONS_W, part_data_mr[part_id], ii, ii, net); 
	   memcpy(output_part_data_mr[part_id], net.layers[ii].output, net.layers[ii].out_w*net.layers[ii].out_h*net.layers[ii].out_c*sizeof(float));
	}

	//for(part_id = 0; part_id<PARTITIONS; part_id ++){
	  //   tmp[part_id] = result_ir_data_serialization_mr(net, part_id, ii);
          //   result_ir_data_deserialization_mr(net, part_id, tmp[part_id], ii);
	  //   free(tmp[part_id]);
	//}

	if(ii < upto){
	   //for(part_id = 0; part_id<PARTITIONS; part_id ++){
	      //tmp[part_id] = req_ir_data_serialization_mr(net, part_id, ii+1);
	      //req_ir_data_deserialization_mr(net, part_id, tmp[part_id], ii+1);
	      //free(tmp[part_id]);
	   //}
	   std::cout << "==========Preparing the input for next layer=============: " << std::endl;
	   for(part_id = 0; part_id<PARTITIONS; part_id ++){
               free(part_data_mr[part_id]);
	       cross_map_overlap_output(net, part_id, ii+1);
	       free(output_part_data_mr[part_id]);
	   }    
	}  
    }

    for(part_id = 0; part_id<PARTITIONS; part_id ++){
        join_output_mr(part_id, output_part_data_mr[part_id],  stage_out, upto, net);
	free(output_part_data_mr[part_id]);
        free(part_data_mr[part_id]);
    }
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


inline float *network_predict_dist_mr(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network_dist_local_mr(net);
    float *out = net->output;
    *net = orig;
    return out;
}

void client_compute_local_mr(network *netp, unsigned int number_of_jobs, std::string thread_name)
{

    network *net = netp;
    srand(2222222);
#ifdef NNPACK
    nnp_initialize();
    net->threadpool = pthreadpool_create(THREAD_NUM);
#endif
    int j;
    int id = 0;//5000 > id > 0
    unsigned int cnt = 0;//5000 > id > 0
    for(cnt = 0; cnt < number_of_jobs; cnt ++){
        image sized;
	sized.w = net->w; sized.h = net->h; sized.c = net->c;

	id = cnt;
        load_image_by_number(&sized, id);
        float *X = sized.data;

	network_predict_dist_mr(net, X);


        free_image(sized);
    }
#ifdef NNPACK
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}

inline int bind_port_client_mr(int portno){
    int sockfd;
    struct sockaddr_in serv_addr;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
	sock_error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) 
	sock_error("ERROR on binding");
    listen(sockfd, 10);//back_log numbers 
    return sockfd;
}

inline void forward_network_dist_mr(network *netp, int sockfd, int frame)
{
    int newsockfd;
    socklen_t clilen;
    network net = *netp;
    struct sockaddr_in cli_addr;
    clilen = sizeof(cli_addr);
    unsigned int bytes_length;
    char* blob_buffer;
    int job_id;

    double time0 = 0.0;
    double time1 = 0.0;
    int upto = STAGES-1;
/*
    if(netp -> input != NULL ){

      char request_type[10];
      newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
      read_sock(newsockfd, request_type, 10);


      dataBlob* blob = new dataBlob(netp -> input, (stage_input_range.w)*(stage_input_range.h)*(net.layers[0].c)*sizeof(float), frame); 
      std::cout << "Sending the entire input to gateway ..." << std::endl;
      send_result_mr(blob, AP, PORTNO);
      //free(netp -> input);
      delete blob;
    }
*/
    newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
    if (newsockfd < 0) sock_error("ERROR on accept");
    read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
    read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
    blob_buffer = (char*)malloc(bytes_length);
    read_sock(newsockfd, blob_buffer, bytes_length);
    part_data_mr[job_id] = (float*) blob_buffer; //Get the top-level input data for this network partition  
    
    for(int ii = 0; ii < STAGES; ii++){
        output_part_data_mr[job_id] = (float*)malloc(
						(input_ranges_mr[job_id][ii].w/net.layers[ii].stride)*
						(input_ranges_mr[job_id][ii].h/net.layers[ii].stride)*net.layers[ii].out_c*sizeof(float));
        time0 = what_time_is_it_now();
	net = forward_stage_mr( job_id/PARTITIONS_W, job_id%PARTITIONS_W, part_data_mr[job_id], ii, ii, net); 
        time1 = what_time_is_it_now();
	comp_time = comp_time +  time1 - time0;
	memcpy(output_part_data_mr[job_id], net.layers[ii].output, net.layers[ii].out_w*net.layers[ii].out_h*net.layers[ii].out_c*sizeof(float));
	if(ii < STAGES - 1){
	     //Send current results
	     float* part_result = result_ir_data_serialization_mr(net, job_id, ii);	
	     unsigned int result_size = result_ir_data_size_mr[job_id/PARTITIONS_W][job_id%PARTITIONS_W][ii]*sizeof(float);
	     dataBlob* blob = new dataBlob(part_result, result_size, job_id); 
             std::cout << "Sending IR result at layer"<<(ii)<<" to AP" << " part "<< job_id << std::endl;
             send_result_mr(blob, AP, PORTNO);
	     free(part_result);
	     delete blob;
	     //Collect IR data from other partitions in other clients
	     newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	     if (newsockfd < 0) sock_error("ERROR on accept");
             read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
	     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     blob_buffer = (char*)malloc(bytes_length);
             std::cout << "Receiving IR result for layer"<<(ii+1)<<" from AP" << " part "<< job_id << std::endl;
	     read_sock(newsockfd, blob_buffer, bytes_length);

	     req_ir_data_deserialization_mr(net, job_id, (float*)blob_buffer, ii+1);
             free(part_data_mr[job_id]);
	     cross_map_overlap_output(net, job_id, ii+1);//Reallocate part_data_mr[job_id]
	     free(output_part_data_mr[job_id]);
	}
    }
    dataBlob* blob = new dataBlob(output_part_data_mr[job_id], net.layers[upto].out_w*net.layers[upto].out_h*net.layers[upto].out_c*sizeof(float), job_id);  
    std::cout << "Sending the part result to AP" << " part "<< job_id << std::endl;
    send_result_mr(blob, AP, PORTNO);  
    free(part_data_mr[job_id]);
    free(output_part_data_mr[job_id]);

}

/*
void client_with_image_input_mr(network *netp, unsigned int number_of_jobs, std::string thread_name)
{
    network *net = netp;
    srand(2222222);
#ifdef NNPACK
    nnp_initialize();
    net->threadpool = pthreadpool_create(THREAD_NUM);
#endif

    int id = 0;//5000 > id > 0
    unsigned int cnt = 0;//5000 > id > 0


    int sockfd = bind_port_client_mr(PORTNO);


    for(cnt = 0; cnt < number_of_jobs; cnt ++){
        image sized;
	sized.w = net->w; sized.h = net->h; sized.c = net->c;
	id = cnt;
        load_image_by_number(&sized, id);
        net->input  = sized.data;
        net->truth = 0;
        net->train = 0;
        net->delta = 0;
        forward_network_dist_mr(net, sockfd, cnt);
	if((cnt+1) == IMG_NUM) {
		std::cout << "Computation time is: " << comp_time/IMG_NUM << std::endl;
	}
        free_image(sized);
    }
#ifdef NNPACK
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}
*/


void client_without_image_input_mr(network *netp, unsigned int number_of_jobs, std::string thread_name)
{
    network *net = netp;
    srand(2222222);
#ifdef NNPACK
    nnp_initialize();
    net->threadpool = pthreadpool_create(THREAD_NUM);
#endif

    int sockfd = bind_port_client_mr(PORTNO);


    for(int cnt = 0; cnt < number_of_jobs; cnt ++){
        net->input  = NULL;
        net->truth = 0;
        net->train = 0;
        net->delta = 0;
        forward_network_dist_mr(net, sockfd, cnt);
	if((cnt+1) == IMG_NUM*DATA_CLI) {
		std::cout << "Computation time is: " << comp_time/IMG_NUM << std::endl;
	}
    }
#ifdef NNPACK
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}


void victim_client_local_mr(){
    unsigned int number_of_images = 4;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_mr(0, STAGES-1, *netp);
    std::thread t1(client_compute_local_mr, &net, number_of_images, "client_compute_local_mr");
    t1.join();
}

void send_all_input_to_gateway(network *netp, unsigned int number_of_jobs, int sockfd, std::string thread_name);

void busy_client_mr(){
    unsigned int number_of_jobs = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_mr(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    int sockfd_syn = bind_port_client_mr(SMART_GATEWAY);
    g_t1 = 0;
    g_t0 = what_time_is_it_now();
    std::thread t1(send_all_input_to_gateway, &net, number_of_jobs, sockfd_syn, "send_all_input_to_gateway");
    std::thread t2(client_without_image_input_mr, &net, number_of_jobs*DATA_CLI, "client_without_image_input_mr");
    t1.join();
    t2.join();
}


void idle_client_mr(){
    unsigned int number_of_jobs = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_mr(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    g_t1 = 0;
    g_t0 = what_time_is_it_now();
    std::thread t1(client_without_image_input_mr, &net, number_of_jobs*DATA_CLI, "client_without_image_input_mr");
    t1.join();

}




