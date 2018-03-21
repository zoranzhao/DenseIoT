#include "darknet_dist_mr.h"
void send_result_share(dataBlob* blob, const char *dest_ip, int portno);
inline int bind_port_client_share(int portno);
void get_data_and_send_result_to_gateway(unsigned int number_of_jobs, int sockfd, std::string thread_name);

void get_data_and_send_result_to_gateway_v2(network *netp, unsigned int number_of_jobs, int sockfd, std::string thread_name){
    network net = *netp; 
    bool print_client = true;
    int newsockfd;
    socklen_t clilen;
    struct sockaddr_in cli_addr;
    clilen = sizeof(cli_addr);

    for(int frame = 0; frame < number_of_jobs; frame++){
	    unsigned int total_part_num = 0;
            unsigned int part = 0;
	    newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	    if (newsockfd < 0) sock_error("ERROR on accept");
	    read_sock(newsockfd, (char*)&total_part_num, sizeof(total_part_num));

   	    if(total_part_num == CUR_CLI){
		    read_sock(newsockfd, (char*)&part, sizeof(part));
		    read_sock(newsockfd, (char*)&total_part_num, sizeof(total_part_num));

		    if(print_client) std::cout << "Recved task number is: "<< total_part_num << std::endl;
		    if(print_client) std::cout << "Starting task ID is: "<< part << std::endl;
		    cur_client_task_num = total_part_num;
		    int job_id; 
		    unsigned int bytes_length;  
		    char* blob_buffer;
	     	    for(int i = 0; i < total_part_num; i ++ ){
		        if(print_client) std::cout << "put_job task ID is: "<< part << std::endl;
			put_job(part_data[part], input_ranges[part][0].w*input_ranges[part][0].h*net.layers[0].c*sizeof(float), part);
			part ++;
	   	    }
	   }else{
	            read_sock(newsockfd, (char*)&total_part_num, sizeof(total_part_num));
		    if(print_client) std::cout << "Recved task number is: "<< total_part_num << std::endl;
		    cur_client_task_num = total_part_num;
		    //close(newsockfd);


		    int job_id; 
		    unsigned int bytes_length;  
		    char* blob_buffer;
		    for(int i = 0; i < total_part_num; i++){
		       //newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
		       read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
		       read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
		       blob_buffer = (char*)malloc(bytes_length);
		       read_sock(newsockfd, blob_buffer, bytes_length);
		       if(print_client) std::cout << "Recved task : "<< job_id << " Size is: "<< bytes_length << std::endl;
		       put_job(blob_buffer, bytes_length, job_id);

		    }
		    close(newsockfd);

	    }




	    for(int i = 0; i < total_part_num; i++){
		dataBlob* blob = result_queue.Dequeue();
		if(print_client) std::cout <<"Sending results size is: "<< blob->getSize() << std::endl;
		if(print_client) std::cout <<"Sending results ID is: "<< blob->getID() << std::endl;  
		send_result_share(blob, AP, PORTNO);
	    }
    }
}


inline void send_yolo_input_v2(network *netp, int sockfd, int frame, float* data)
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
      fork_input(0, data, *netp);
      send_result_share(blob, AP, PORTNO);
      //free(netp -> input);
      delete blob;
    }

}
inline void forward_network_dist_share(network *netp, int sockfd, int frame);


void send_all_input_to_gateway_and_fork_local(network *netp, unsigned int number_of_jobs, int sockfd, std::string thread_name)
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
        send_yolo_input_v2(net, sockfd, cnt, sized.data);
        free_image(sized);
    }

}




void client_without_image_input_share(network *netp, unsigned int number_of_jobs, int sockfd, std::string thread_name);




void busy_client_share_v2(){
    unsigned int number_of_jobs = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    int sockfd = bind_port_client_share(PORTNO);
    int sockfd_syn = bind_port_client_share(SMART_GATEWAY);
    g_t1 = 0;
    g_t0 = what_time_is_it_now();
    std::thread t1(send_all_input_to_gateway_and_fork_local, &net, number_of_jobs, sockfd_syn, "send_all_input_to_gateway_and_fork_local");
    std::thread t2(client_without_image_input_share, &net, number_of_jobs*DATA_CLI, sockfd_syn, "client_without_image_input_share");
    std::thread t3(get_data_and_send_result_to_gateway_v2, &net, number_of_jobs*DATA_CLI, sockfd, "get_data_and_send_result_to_gateway_v2");
    t1.join();
    t2.join();
    t3.join();
}


void idle_client_share_v2(){
    unsigned int number_of_jobs = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    int sockfd = bind_port_client_share(PORTNO);
    g_t1 = 0;
    g_t0 = what_time_is_it_now();
    std::thread t1(client_without_image_input_share, &net, number_of_jobs*DATA_CLI, sockfd, "client_without_image_input_share");
    std::thread t2(get_data_and_send_result_to_gateway_v2, &net, number_of_jobs*DATA_CLI, sockfd, "get_data_and_send_result_to_gateway_v2");
    t1.join();
    t2.join();
}




