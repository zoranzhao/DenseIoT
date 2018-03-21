#include "darknet_dist_mr.h"
//ACT_CLI

void send_result_mr(dataBlob* blob, const char *dest_ip, int portno);
void gateway_require_data(char* request_type, const char *cli_ip, int portno);

void data_map_reduce_v2(network net, int number_of_images, int portno)
{  
   bool print_gateway = false;
   int sockfd, newsockfd;
   socklen_t clilen;
   struct sockaddr_in serv_addr, cli_addr;
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
   clilen = sizeof(cli_addr);
   int job_id;
   unsigned int bytes_length;
   char *blob_buffer;
   std::list< std::string > cli_list;
   std::list< int > job_id_list;
   double time0 = 0.0;
   double time1 = 0.0;
   int frame;
   for(int id = 0; id < number_of_images; id++){
	for(int cli_id = 0; cli_id < DATA_CLI; cli_id++){	
	     //Receive the data from a single client;
	     gateway_require_data("start", addr_list[cli_id], SMART_GATEWAY);
	     newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	     g_t0 = what_time_is_it_now();

	     time0 = what_time_is_it_now();
	     read_sock(newsockfd, (char*)&frame, sizeof(frame));
	     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     blob_buffer = (char*)malloc(bytes_length);
	     read_sock(newsockfd, blob_buffer, bytes_length);
	     close(newsockfd);
	     time1 = what_time_is_it_now();
	     commu_data_amount = commu_data_amount + sizeof(frame) + sizeof(bytes_length) + bytes_length; 
	     commu_time = commu_time + time1 - time0;
	     if(print_gateway)
		std::cout << "Receiving the entire input data to be distributed from client" << inet_ntoa(cli_addr.sin_addr) << std::endl;

	     time0 = what_time_is_it_now();
	     //Distribute the input data 
	     fork_input_mr(0, (float*)blob_buffer, net);
	     for(int cli_cnt = 0; cli_cnt < ACT_CLI; cli_cnt ++ ){
                if(cli_id == cli_cnt) {
		     bytes_length = 0;
		     dataBlob* blob = new dataBlob(part_data_mr[cli_cnt], bytes_length, cli_cnt); 
		     send_result_mr(blob, addr_list[cli_cnt], portno);
	             commu_data_amount = commu_data_amount + sizeof(frame) + sizeof(bytes_length) + bytes_length; 
	       	     free(part_data_mr[cli_cnt]);
		     delete blob;
		     continue;	
		}

		if(print_gateway)
		  std::cout << "Sending the partition "<< cli_cnt << " to client" << addr_list[cli_cnt] << std::endl;
		bytes_length = input_ranges_mr[cli_cnt][0].w*input_ranges_mr[cli_cnt][0].h*net.layers[0].c*sizeof(float);
		dataBlob* blob = new dataBlob(part_data_mr[cli_cnt], bytes_length, cli_cnt); 
		send_result_mr(blob, addr_list[cli_cnt], portno);
	        commu_data_amount = commu_data_amount + sizeof(frame) + sizeof(bytes_length) + bytes_length; 
	       	free(part_data_mr[cli_cnt]);
		delete blob;
	     }
	     time1 = what_time_is_it_now();
	     commu_time = commu_time + time1 - time0;

	     for(int ii = 0; ii < STAGES - 1; ii ++){
		for(int cli_cnt = 0; cli_cnt < ACT_CLI; cli_cnt ++ ){
	     	  newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
		  time0 = what_time_is_it_now();
		  read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
		  read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
		  blob_buffer = (char*)malloc(bytes_length);
		  read_sock(newsockfd, blob_buffer, bytes_length);
		  time1 = what_time_is_it_now();
	          commu_data_amount = commu_data_amount + sizeof(frame) + sizeof(bytes_length) + bytes_length; 
		  commu_time = commu_time + time1 - time0;
		  if(print_gateway)
		     std::cout << "Receiving IR result at layer"<<ii<<" from client" << inet_ntoa(cli_addr.sin_addr)<< " part "<< job_id << std::endl;
		  result_ir_data_deserialization_mr(net, job_id, (float*)blob_buffer, ii);
		  free(blob_buffer);
	     	  close(newsockfd);
	  	  cli_list.push_back( std::string(inet_ntoa(cli_addr.sin_addr)) );
		  job_id_list.push_back(job_id);
		}
		time0 = what_time_is_it_now();
		for(int cli_cnt = 0; cli_cnt < ACT_CLI; cli_cnt ++ ){
		  std::string cur_addr = cli_list.front();
		  cli_list.pop_front();
		  int cur_id = job_id_list.front();
		  job_id_list.pop_front();
		  blob_buffer = (char*) req_ir_data_serialization_mr(net, cur_id, ii+1);
		  bytes_length = req_ir_data_size_mr[cur_id/PARTITIONS_W][cur_id%PARTITIONS_W][ii+1]* sizeof(float);
		  dataBlob* blob = new dataBlob(blob_buffer, bytes_length, cur_id); 
		  if(print_gateway)
		    std::cout << "Sending IR result at layer"<<(ii+1)<<" to client" << cur_addr << " part "<< cur_id << std::endl;
		  send_result_mr(blob, cur_addr.c_str(), portno);
	          commu_data_amount = commu_data_amount + sizeof(frame) + sizeof(bytes_length) + bytes_length; 
		  free(blob_buffer);
		  delete blob;
		}
		time1 = what_time_is_it_now();
		commu_time = commu_time + time1 - time0;
	     }
	     for(int cli_cnt = 0; cli_cnt < ACT_CLI; cli_cnt ++ ){
		  newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
		  time0 = what_time_is_it_now();
		  read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
		  read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
		  blob_buffer = (char*)malloc(bytes_length);
		  read_sock(newsockfd, blob_buffer, bytes_length);
	          commu_data_amount = commu_data_amount + sizeof(frame) + sizeof(bytes_length) + bytes_length; 
		  time1 = what_time_is_it_now();
		  commu_time = commu_time + time1 - time0;
	     	  close(newsockfd);
		  if(print_gateway)
		    std::cout << "Receiving IR result at layer from client" << inet_ntoa(cli_addr.sin_addr)<< " part "<< job_id << std::endl;
		  recv_data[id][cli_id][job_id]=(float*)blob_buffer;
	     }
	     g_t1 = g_t1 + what_time_is_it_now() - g_t0;
	     std::cout << "Global time is: " << g_t1 <<std::endl;
	     std::cout << "Total latency for client "<< cli_id << " is: " << g_t1/((float)(id + 1)) << std::endl;
	     std::cout << "The entire throughput of is: " << ((float)((id)*DATA_CLI + cli_id + 1))/g_t1 << std::endl;
	     std::cout << "Data from client " << cli_id << " has been fully collected and begin to compute ..."<< std::endl;
	     if( ((id + 1) == IMG_NUM) && (cli_id == DATA_CLI-1) ) {
		std::cout << "Communication/synchronization overhead time is: " << commu_time/(IMG_NUM)  << std::endl;
		std::cout << "Communication data amount: " << commu_data_amount/(IMG_NUM*1024*1024) << std::endl;
	     }
	     int all = merge_v2(cli_id, id, 0);
	     ready_queue.Enqueue(all);
	}
   }

   close(sockfd);
}



void gateway_service_mr(network net, int number_of_images, std::string thread_name);

void gateway_sync_mr_v2(network net, int number_of_images, std::string thread_name){
    data_map_reduce_v2(net, number_of_images, PORTNO);
}


void smart_gateway_mr_v2(){
    int number_of_images = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_mr(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    std::thread t1(gateway_sync_mr_v2, net, number_of_images, "gateway_sync_mr");
    std::thread t2(gateway_service_mr, net, number_of_images*DATA_CLI, "gateway_service_mr");

    g_t0 = what_time_is_it_now();
    g_t1 = 0;
    t1.join();
    t2.join();
}

