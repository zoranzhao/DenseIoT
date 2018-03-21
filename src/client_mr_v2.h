#include "darknet_dist_mr.h"
void send_result_mr(dataBlob* blob, const char *dest_ip, int portno);
inline void forward_network_dist_local_mr(network *netp);
inline float *network_predict_dist_mr(network *net, float *input);
void client_compute_local_mr(network *netp, unsigned int number_of_jobs, std::string thread_name);
inline int bind_port_client_mr(int portno);

inline void forward_network_dist_mr_v2(network *netp, int sockfd, int frame)
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


    if(bytes_length != 0){
	part_data_mr[job_id] = (float*) blob_buffer; //Get the top-level input data for this network partition  
    }else{
	std::cout << "Reuse local data for partitnion:" << job_id << std::endl;
    }
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
             //std::cout << "Sending IR result at layer"<<(ii)<<" to AP" << " part "<< job_id << std::endl;
             send_result_mr(blob, AP, PORTNO);
	     free(part_result);
	     delete blob;
	     //Collect IR data from other partitions in other clients
	     newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	     if (newsockfd < 0) sock_error("ERROR on accept");
             read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
	     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     blob_buffer = (char*)malloc(bytes_length);
             //std::cout << "Receiving IR result for layer"<<(ii+1)<<" from AP" << " part "<< job_id << std::endl;
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


void client_without_image_input_mr_v2(network *netp, unsigned int number_of_jobs, std::string thread_name)
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
        forward_network_dist_mr_v2(net, sockfd, cnt);
	if((cnt+1) == IMG_NUM*DATA_CLI) {
		std::cout << "Computation time is: " << comp_time/IMG_NUM << std::endl;
	}
    }
#ifdef NNPACK
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}


void victim_client_local_mr();



inline void send_yolo_input_mr_v2(network *netp, int sockfd, int frame, float* data)
{
    bool print_client = false;
    int newsockfd;
    socklen_t clilen;
    struct sockaddr_in cli_addr;
    clilen = sizeof(cli_addr);
    network net = *netp;
    if(netp -> input != NULL ){
      char request_type[10];
      newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
      read_sock(newsockfd, request_type, 10);
      dataBlob* blob = new dataBlob((void*)data, (stage_input_range.w)*(stage_input_range.h)*(net.layers[0].c)*sizeof(float), frame); 
      if(print_client) std::cout << "Sending the entire input to gateway ..." << std::endl;
      fork_input_mr(0, data, *netp);

      send_result_share(blob, AP, PORTNO);
      //free(netp -> input);
      delete blob;
    }

}

void send_all_input_to_gateway_mr_v2(network *netp, unsigned int number_of_jobs, int sockfd, std::string thread_name)
{
    network *net = netp;
    srand(2222222);
    int id = 0;//5000 > id > 0
    unsigned int cnt = 0;//5000 > id > 0

    for(cnt = 0; cnt < number_of_jobs; cnt ++){
        image sized;
	sized.w = net->w; sized.h = net->h; sized.c = net->c;
	id = cnt;
        load_image_by_number(&sized, id);
        net->input  = sized.data;
        net->truth = 0;
        net->train = 0;
        net->delta = 0;
        send_yolo_input_mr_v2(net, sockfd, cnt, sized.data);
        free_image(sized);
    }

}



void busy_client_mr_v2(){
    unsigned int number_of_jobs = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_mr(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    int sockfd_syn = bind_port_client_mr(SMART_GATEWAY);
    g_t1 = 0;
    g_t0 = what_time_is_it_now();
    std::thread t1(send_all_input_to_gateway_mr_v2, &net, number_of_jobs, sockfd_syn, "send_all_input_to_gateway_mr_v2");
    std::thread t2(client_without_image_input_mr_v2, &net, number_of_jobs*DATA_CLI, "client_without_image_input_mr");
    t1.join();
    t2.join();
}


void idle_client_mr_v2(){
    unsigned int number_of_jobs = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_mr(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    g_t1 = 0;
    g_t0 = what_time_is_it_now();
    std::thread t1(client_without_image_input_mr_v2, &net, number_of_jobs*DATA_CLI, "client_without_image_input_mr");
    t1.join();

}




