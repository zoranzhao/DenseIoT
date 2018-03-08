#include "darknet_dist.h"
#include "serialization_v2.h"
//A function to compress cli id and part id into a interger
inline int merge(int cli, int part);
inline int merge_v2(int cli, int frame, int part);
inline int get_cli_v2(int all);
inline int get_part_v2(int all);
inline int get_frame_v2(int all);


void get_ir_data_from_gateway(network net, int all){
     int part_id = get_part_v2(all);
     int frame = get_frame_v2(all);
     int cli_id = get_cli_v2(all);
     int gateway_sock;      
     char *reuse_data;
     unsigned int reuse_data_length;

     bool* req = get_local_coverage_v2(part_id, frame, cli_id);
     if((!req[0]) && (!req[1]) && (!req[2]) && (!req[3])){
	free(req);	
	return;
     }
     struct sockaddr_in serv_addr;
     gateway_sock = socket(AF_INET, SOCK_STREAM, 0);
     if (gateway_sock < 0) 
	sock_error("ERROR opening socket");
     bzero((char *) &serv_addr, sizeof(serv_addr));
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = inet_addr(AP) ;
     serv_addr.sin_port = htons(SMART_GATEWAY);
     if (connect(gateway_sock,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
		sock_error("ERROR connecting");
     write_sock(gateway_sock, "ir_data_r", 10);
     int cli_frame_part = all;	
     write_sock(gateway_sock, (char*)&cli_frame_part, sizeof(cli_frame_part));
     write_sock(gateway_sock, (char*)req, 4*sizeof(bool));

     read_sock(gateway_sock, (char*)&reuse_data_length, sizeof(reuse_data_length));
     reuse_data = (char*)malloc(reuse_data_length);
     read_sock(gateway_sock, reuse_data, reuse_data_length);
     //std::cout << "Stealing reuse data for partition number, size is: "<< reuse_data_length << std::endl;
     req_ir_data_deserialization_v2(net, part_id, (float*)reuse_data, 0, STAGES-1, req);
     //std::cout << "Stealing reuse data for partition number, size is: "<< reuse_data_length << std::endl;
     free(reuse_data);
     free(req);		
     close(gateway_sock);
}


void send_ir_data_to_gateway(network net, int all){
     int part_id = get_part_v2(all);
     int gateway_sock;      
     char *reuse_data;
     unsigned int reuse_data_length;

     struct sockaddr_in serv_addr;
     gateway_sock = socket(AF_INET, SOCK_STREAM, 0);
     if (gateway_sock < 0) 
	sock_error("ERROR opening socket");
     bzero((char *) &serv_addr, sizeof(serv_addr));
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = inet_addr(AP) ;
     serv_addr.sin_port = htons(SMART_GATEWAY);
     if (connect(gateway_sock,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
		sock_error("ERROR connecting");
     write_sock(gateway_sock, "ir_data", 10);

     reuse_data_length = result_ir_data_size[part_id]*sizeof(float);
     reuse_data = (char*) result_ir_data_serialization(net, part_id, 0, STAGES-1);
     int cli_frame_part = all;	
     write_sock(gateway_sock, (char*)&cli_frame_part, sizeof(cli_frame_part));
     write_sock(gateway_sock, (char*)&reuse_data_length, sizeof(reuse_data_length));
     write_sock(gateway_sock, reuse_data, reuse_data_length);

     free(reuse_data);	
     close(gateway_sock);
}


inline int forward_network_dist_gateway_shuffle_v2(network *netp, network orig, int frame)
{
    int workload_amount;
    int part;
    network net = *netp;

    int startfrom = 0;
    int upto = STAGES-1;

    float* stage_in = net.input; 
    int cli_id = (CUR_CLI);
    //Partition and shuffle the input data for the processing stage
    double time0 = 0.0;
    double time1 = 0.0;

    fork_input_reuse(startfrom, stage_in, net);
    for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
	for(int p_w = (p_h) % 2; p_w < PARTITIONS_W; p_w = p_w + 2){ 
		need_ir_data[part_id[p_h][p_w]]=0;
		printf("Putting jobs %d\n", part_id[p_h][p_w]);
		put_job(reuse_part_data[part_id[p_h][p_w]], 
			reuse_input_ranges[part_id[p_h][p_w]][startfrom].w*reuse_input_ranges[part_id[p_h][p_w]][startfrom].h*net.layers[startfrom].c*sizeof(float), 
			merge_v2(cli_id, frame, part_id[p_h][p_w]));
	}
    }
    for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
	for(int p_w = (p_h+1) % 2; p_w < PARTITIONS_W; p_w = p_w + 2){ 
		need_ir_data[part_id[p_h][p_w]]=1;
		printf("Putting jobs %d\n", part_id[p_h][p_w]);
		put_job(reuse_part_data[part_id[p_h][p_w]], 
			reuse_input_ranges[part_id[p_h][p_w]][startfrom].w*reuse_input_ranges[part_id[p_h][p_w]][startfrom].h*net.layers[startfrom].c*sizeof(float), 
			merge_v2(cli_id, frame, part_id[p_h][p_w]));
	}
    }
    char reg[10] = "register";
    ask_gateway(reg, AP, SMART_GATEWAY); //register number of tasks


    float* data;
    int part_id;
    unsigned int size;

  
    for(part = 0; 1; part ++){
       int all;
       try_get_job((void**)&data, &size, &all);
       part_id = get_part_v2(all);
       if(data == NULL) {
	   ask_gateway(reg, AP, SMART_GATEWAY); //remove the registration when we are running out of tasks
	   printf("%d parts out of the %d are processes locally, yeeha!\n", part, PARTITIONS); 
	   workload_amount = part;
	   break;
       }
       //std::cout<< "Processing task "<< part_id <<std::endl;
       if( is_part_ready_v2(part_id, frame, cli_id) != 1 && need_ir_data[part_id]==1){
		time0 = what_time_is_it_now();
		net = forward_stage( part_id/PARTITIONS_W, part_id%PARTITIONS_W, part_data[part_id], startfrom, upto, net);
		time1 = what_time_is_it_now();
		comp_time = comp_time + (time1 - time0);
       }else{
		if(need_ir_data[part_id]==1){
			//std::cout << "[ir_data_r] .... Is able to reuse data for processing in frame: " << frame << ", part: "<< part_id << std::endl;
			get_ir_data_from_gateway(net, all);
		}
		time0 = what_time_is_it_now();
		net = forward_stage_reuse_full( part_id/PARTITIONS_W, part_id%PARTITIONS_W, data, startfrom, upto, net);
		time1 = what_time_is_it_now();
		comp_time = comp_time + (time1 - time0);

       }
       if(need_ir_data[part_id]==0){
		//std::cout << "[ir_data] ... Set coverage for local task..." << std::endl;
		set_global_and_local_coverage_v2(part_id, frame, cli_id);
		send_ir_data_to_gateway(net, all);
       }
       //std::cout<< "Processed task "<< part_id <<std::endl;

       put_result(net.layers[upto].output, net.layers[upto].outputs* sizeof(float), all);
       free(data);
    }

    return workload_amount;

}



inline int network_predict_dist_shuffle_v2(network *net, float *input, int frame)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    int workload_amount = forward_network_dist_gateway_shuffle_v2(net, orig, frame);
    float *out = net->output;
    *net = orig;
    return workload_amount;
}



void client_compute_shuffle_v2(network *netp, unsigned int number_of_jobs, std::string thread_name)
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
    int workload_amount = 0;
    for(cnt = 0; cnt < number_of_jobs; cnt ++){
        image sized;
	sized.w = net->w; sized.h = net->h; sized.c = net->c;
	id = cnt;
        load_image_by_number(&sized, id);
        float *X = sized.data;
	workload_amount = workload_amount + network_predict_dist_shuffle_v2(net, X, cnt);
	std::cout << workload_amount << std::endl;
        free_image(sized);
	if((cnt+1) == IMG_NUM) {
		std::cout << "Communication/synchronization overhead time is: " << commu_time/IMG_NUM << std::endl;
		std::cout << "Computation time is: " << comp_time/IMG_NUM << std::endl;
	}
    }
#ifdef NNPACK
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}



dataBlob* steal_and_return_shuffle_v2(network net, int *ready, const char *dest_ip, int portno)
{
     int sockfd;
     struct sockaddr_in serv_addr;
     sockfd = socket(AF_INET, SOCK_STREAM, 0);
     if (sockfd < 0) 
        sock_error("ERROR opening socket");
     bzero((char *) &serv_addr, sizeof(serv_addr));
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = inet_addr(dest_ip) ;
     serv_addr.sin_port = htons(portno);

     if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
        sock_error("ERROR connecting");

     char request_type[10] = "steals";
     char *blob_buffer;
     unsigned int bytes_length;
     int all;
     write_sock(sockfd, request_type, 10);
     read_sock(sockfd, (char*)&all, sizeof(all));

     if(all != -1){
	  int job_id = get_part_v2(all);
	  int frame = get_frame_v2(all);
	  int cli_id = get_cli_v2(all);

	  read_sock(sockfd, (char*)&bytes_length, sizeof(bytes_length));
	  blob_buffer = (char*)malloc(bytes_length);
	  if (blob_buffer==NULL) {printf("(char*)malloc(bytes_length) failed\n"); exit (1);}
	  read_sock(sockfd, blob_buffer, bytes_length);
          if(need_ir_data[job_id]==1){
	     //std::cout << "Stealing reuse data for partition number: "<< job_id << std::endl;
	     char *reuse_data;
	     unsigned int reuse_data_length;
	     int reuse_part_id;
	     read_sock(sockfd, (char*)&reuse_part_id, sizeof(reuse_part_id));
	     if(reuse_part_id != -1){//Otherwise equal to job_id
 		     *ready = 1;
                     get_ir_data_from_gateway(net, all);
	     }
           }
     }
     close(sockfd);
     dataBlob* ret = (new dataBlob((void*)blob_buffer, bytes_length, all)) ;
     return ret;
}






inline void steal_through_gateway_shuffle_v2(network *netp, std::string thread_name){
    netp->truth = 0;
    netp->train = 0;
    netp->delta = 0;
    int part;
#ifdef NNPACK
    nnp_initialize();
    netp->threadpool = pthreadpool_create(THREAD_NUM);
#endif
    network net = *netp;
    int startfrom = 0;
    int upto = STAGES-1;
    float* data;
    int part_id;
    int all;
    unsigned int size;
    char steal[10] = "steals";
    double t0;
    double t1 = 0; 
    struct sockaddr_in addr;
    int workload_amount = 0;
    int frame ;
    double time0 = 0.0;
    double time1 = 0.0;

    while(1){
	addr.sin_addr.s_addr = ask_gateway(steal, AP, SMART_GATEWAY);
	if(addr.sin_addr.s_addr == inet_addr("0.0.0.0")){
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		continue;
	}
        int ready = 0;
	dataBlob* blob = steal_and_return_shuffle_v2(*netp, &ready, inet_ntoa(addr.sin_addr), PORTNO);
        if(blob->getID() == -1){
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		continue;
	}
        int cli_id = get_client_id(inet_ntoa(addr.sin_addr));
	data = (float*)(blob -> getDataPtr());
	all = blob -> getID();
	part_id = get_part_v2(all);
	frame = get_frame_v2(all);
	size = blob -> getSize();
	std::cout << "Steal part " << part_id << " from client "<< cli_id <<", size is: "<< size << " with ready flag: "<< ready <<std::endl;
	//std::cout << "[steals] .... got task from cli "<< cli_id<< " , frame: " << frame << ", part: "<< part_id << std::endl;
	if( ready == 0 && need_ir_data[part_id]==1){
		time0 = what_time_is_it_now();
		net = forward_stage(part_id/PARTITIONS_W, part_id%PARTITIONS_W, data, startfrom, upto, net);
		time1 = what_time_is_it_now();
		comp_time = comp_time + (time1 - time0);
	}else{ 
		time0 = what_time_is_it_now();
		net = forward_stage_reuse_full(part_id/PARTITIONS_W, part_id%PARTITIONS_W, data, startfrom, upto, net);
		time1 = what_time_is_it_now();
		comp_time = comp_time + (time1 - time0);
	}
	free(data);
	delete blob;
	if(need_ir_data[part_id]==0){//Doesn't need intermediate data, which means it will generate IR data
		//std::cout << "[ir_data] .... Set coverage for stolen task from cli "<< cli_id<< " , frame: " << frame << ", part: "<< part_id << std::endl;
		set_global_and_local_coverage_v2(part_id, frame, cli_id);
		send_ir_data_to_gateway(net, all);
	}

	if((frame+1) == IMG_NUM) {
		std::cout << workload_amount <<std::endl;
		std::cout << "Communication/synchronization overhead time is: " << commu_time/IMG_NUM << std::endl;
		std::cout << "Computation time is: " << comp_time/IMG_NUM << std::endl;
	}
	workload_amount ++ ;
	put_result((void*)net.layers[upto].output, net.layers[upto].outputs*sizeof(float), all);
    }
#ifdef NNPACK
    pthreadpool_destroy(netp->threadpool);
    nnp_deinitialize();
#endif
}


void serve_steal_and_gather_result_shuffle_v2(network net, int portno)
{
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

   unsigned int bytes_length;
   char* blob_buffer;
   int job_id;
   int frame;
   int all;
   unsigned int id;
   char request_type[10];
   int cli_id = (CUR_CLI);
   double time0 = 0.0;
   double time1 = 0.0;

   while(1){
	//Recieving stealing request from client devices
	//TODO Need to handle fail on stealing

     	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	time0 = what_time_is_it_now();
	if (newsockfd < 0) sock_error("ERROR on accept");
        read_sock(newsockfd, request_type, 10); 
        if(strcmp (request_type,"result") == 0){
	     std::cout << "WARNING: Result should not be returned to the data resource device ... " << std::endl;
	     read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
	     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     blob_buffer = (char*)malloc(bytes_length);
	     read_sock(newsockfd, blob_buffer, bytes_length);
	     put_result((void*)blob_buffer, bytes_length, job_id);
	}else if(strcmp (request_type,"steals") == 0){
	     //std::cout << "Recving quest from " << inet_ntoa(cli_addr.sin_addr) <<std::endl;
	     try_get_job((void**)&blob_buffer, &bytes_length, &all);
	     if(blob_buffer == NULL) {
		all = -1;
	     	write_sock(newsockfd, (char*)&all, sizeof(all));
	     }else{
		int job_id = get_part_v2(all);
		int frame = get_frame_v2(all);
		if(need_ir_data[job_id]==0){
			//std::cout << "Serve steal of part number "<< job_id << " no need for reused data..." << std::endl;
			write_sock(newsockfd, (char*)&all, sizeof(all));
			write_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
			write_sock(newsockfd, blob_buffer, bytes_length);
		        //std::cout << "Got job "<< job_id << " from queue, "<<"job size is: "<< bytes_length <<", sending job "  << std::endl;
		}else if( (need_ir_data[job_id]==1) && (is_part_ready_v2(job_id, frame, cli_id)) ){
		     	write_sock(newsockfd, (char*)&all, sizeof(all));
		    	write_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
		    	write_sock(newsockfd, blob_buffer, bytes_length);
			//std::cout << "Serve the stealing of reuse data for partition number: "<< job_id << std::endl;
			write_sock(newsockfd, (char*)&job_id, sizeof(job_id));
		}else if( need_ir_data[job_id]==1 ) {
			bytes_length = input_ranges[job_id][0].w*input_ranges[job_id][0].h*net.layers[0].c*sizeof(float);
			write_sock(newsockfd, (char*)&all, sizeof(all));
		    	write_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
		    	write_sock(newsockfd, (char*)part_data[job_id], bytes_length);
			//std::cout << "Serve the stealing of reuse data for partition number: "<< job_id<<", well it is not ready yet ...." << std::endl;
			job_id = -1; 
			write_sock(newsockfd, (char*)&job_id, sizeof(job_id));
		}

	     }
	     free(blob_buffer);
	     time1 = what_time_is_it_now();
	     commu_time = commu_time + (time1 - time0);
        }else if(strcmp (request_type,"ir_data") == 0){
	     read_sock(newsockfd, (char*)&all, sizeof(all));
	     int frame = get_frame_v2(all);
	     int job_id = get_part_v2(all);
	     cli_id = get_cli_v2(all);
	     //std::cout << "Recved a set_coverage request from gateway..." << std::endl;
	     set_coverage_v2(job_id, frame, cli_id);
        }
     	close(newsockfd);
   }
   close(sockfd);

}


void steal_server_shuffle_v2(network net, std::string thread_name){
    clear_coverage_v2();
    serve_steal_and_gather_result_shuffle_v2(net, PORTNO);
}


void froward_result_to_gateway_v2(std::string thread_name){
    while(1){
	dataBlob* blob = result_queue.Dequeue();
	send_result(blob, AP, PORTNO);
    }
}


void victim_client_shuffle_v2(){
    unsigned int number_of_images = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_shuffle(0, STAGES-1, *netp);
    init_recv_counter();
    exec_control(START_CTRL);
    std::thread t1(client_compute_shuffle_v2, &net, number_of_images, "client_compute");
    std::thread t2(steal_server_shuffle_v2, net, "steal_server");
    std::thread t3(froward_result_to_gateway_v2, "froward_result_to_gateway");
    t1.join();
    t2.join();
    t3.join();
}


void idle_client_shuffle_v2(){
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_shuffle(0, STAGES-1, *netp);
    init_recv_counter();
    exec_control(START_CTRL);
    std::thread t1(steal_through_gateway_shuffle_v2, &net,  "steal_forward");
    std::thread t2(froward_result_to_gateway_v2, "froward_result_to_gateway");
    t1.join();
    t2.join();
}



 






