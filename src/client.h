#include "darknet_dist.h"

inline int forward_network_dist_gateway(network *netp, network orig, int frame)
{
    int workload_amount;
    int part;
    network net = *netp;

    int startfrom = 0;
    int upto = STAGES-1;

    float* stage_in = net.input; 
    int cli_id = get_client_id(CUR_CLI);
    fork_input(startfrom, stage_in, net);
    char reg[10] = "register";

    for(part = 0; part < PARTITIONS; part ++){
      printf("Putting jobs %d\n", part);
      put_job(part_data[part], input_ranges[part][startfrom].w*input_ranges[part][startfrom].h*net.layers[startfrom].c*sizeof(float), merge_v2(cli_id, frame, part));
    }
    ask_gateway(reg, AP, SMART_GATEWAY); //register number of tasks


    float* data;
    int part_id;
    unsigned int size;

    for(part = 0; 1; part ++){
       int all;
       try_get_job((void**)&data, &size, &all);
       part_id = get_part_v2(all);

       if(data == NULL) {
	   printf("%d parts out of the %d are processes locally, yeeha!\n", part, PARTITIONS); 
	   ask_gateway(reg, AP, SMART_GATEWAY); //remove the registration when we are running out of tasks
	   workload_amount = part;
	   break;
       }
       std::cout<< "Processing task "<< part_id <<std::endl;
       time0 = what_time_is_it_now();
       net = forward_stage(part_id/PARTITIONS_W, part_id%PARTITIONS_W,  data, startfrom, upto, net);
       time1 = what_time_is_it_now();
       comp_time = comp_time + (time1 - time0); 
       
       put_result(net.layers[upto].output, net.layers[upto].outputs* sizeof(float), all);
       free(data);
    }

    return workload_amount;

}



inline int network_predict_dist(network *net, float *input, int frame)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    int workload_amount = forward_network_dist_gateway(net, orig,  frame);
    float *out = net->output;
    *net = orig;
    return workload_amount;
}


void client_compute(network *netp, unsigned int number_of_jobs, std::string thread_name)
{
    int workload_amount = 0;
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
	workload_amount  = workload_amount + network_predict_dist(net, X, cnt);
        free_image(sized);
	std::cout << workload_amount << std::endl;
	if((cnt+1) == IMG_NUM) {
		std::cout << "Communication/synchronization overhead time is: " << commu_time << std::endl;
		std::cout << "Computation time is: " << comp_time << std::endl;
	}

    }
#ifdef NNPACK
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}


inline void steal_through_gateway(network *netp, std::string thread_name){
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
    int frame ;
    int workload_amount = 0;
    while(1){
        //t0 = get_real_time_now();
	addr.sin_addr.s_addr = ask_gateway(steal, AP, SMART_GATEWAY);
	//std::cout << "Stolen address from the gateway is: " << inet_ntoa(addr.sin_addr) << std::endl;
	if(addr.sin_addr.s_addr == inet_addr("0.0.0.0")){
		//If the stolen address is a broadcast address, steal again 
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		//std::cout << "Nothing is registered in the gateway device, sleep for a while" << std::endl;
		continue;
	}
	dataBlob* blob = steal_and_return(inet_ntoa(addr.sin_addr), PORTNO);
        if(blob->getID() == -1){
		//Have stolen nothing, this can happen if a registration remote call happens
		//after an check call happens to the gateway
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		//std::cout << "The victim has already finished current job list" << std::endl;
		//std::cout << "Wait for a while until next stealing iteration" << std::endl;
		continue;
	}
	data = (float*)(blob -> getDataPtr());
	all = blob -> getID();

        int cli_id = get_cli_v2(all);
	part_id = get_part_v2(all);
	frame = get_frame_v2(all);
	size = blob -> getSize();
	std::cout << "Steal part " << part_id <<", size is: "<< size <<std::endl;
        time0 = what_time_is_it_now();
	net = forward_stage(part_id/PARTITIONS_W, part_id%PARTITIONS_W, data, startfrom, upto, net);
        time1 = what_time_is_it_now();
	comp_time = comp_time + (time1 - time0);
	free(data);
	delete blob;
	//blob -> setData((void*)(net.layers[upto].output));
	//blob -> setSize(net.layers[upto].outputs*sizeof(float));
        //t1 =  get_real_time_now() - t0;
        //std::cout << "Exec cost is: "<<t1<< std::endl;
        //t0 =  get_real_time_now();
	//send_result(blob, inet_ntoa(addr.sin_addr), PORTNO);
	//send_result(blob, AP, SMART_GATEWAY);
	put_result((void*)net.layers[upto].output, net.layers[upto].outputs*sizeof(float), all);
	workload_amount++;
	std::cout << workload_amount <<std::endl;
        //t1 =  get_real_time_now() - t0;
        //std::cout << "Send result cost is: "<<t1<< std::endl;

    }
#ifdef NNPACK
    pthreadpool_destroy(netp->threadpool);
    nnp_deinitialize();
#endif
}


void serve_steal(int portno)
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
   int cli_id = get_client_id(CUR_CLI);
   while(1){
	//Recieving stealing request from client devices
	//TODO Need to handle fail on stealing

     	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	time0 = what_time_is_it_now();
	if (newsockfd < 0) sock_error("ERROR on accept");
        read_sock(newsockfd, request_type, 10); 
        if(strcmp (request_type,"result") == 0){
	     std::cout << "WARNING: Result should not be returned to the data resource device ... " << std::endl;
	     //std::cout << "At time " << g_t1 << ", recv result " << job_id  <<std::endl;  
	     //std::cout << "Recving result from " << inet_ntoa(cli_addr.sin_addr) <<std::endl;
	     read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
	     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     blob_buffer = (char*)malloc(bytes_length);
	     read_sock(newsockfd, blob_buffer, bytes_length);
	     //std::cout << "Got result "<< job_id << " from queue, "<<"result size is: "<< bytes_length  << std::endl;
	     put_result((void*)blob_buffer, bytes_length, job_id);
	     //std::cout << "At time " << g_t1 << ", put result " << job_id << " into res_queue" <<std::endl;  
	}else if(strcmp (request_type,"steals") == 0){
	     //std::cout << "Recving quest from " << inet_ntoa(cli_addr.sin_addr) <<std::endl;
	     try_get_job((void**)&blob_buffer, &bytes_length, &all);
	     if(blob_buffer == NULL) {bytes_length = 4; blob_buffer = (char*)malloc(bytes_length+1); all = -1;}
	     //std::cout << "Got job "<< job_id << " from queue, "<<"job size is: "<< bytes_length <<", sending job "  << std::endl;
	     write_sock(newsockfd, (char*)&all, sizeof(all));
	     write_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     write_sock(newsockfd, blob_buffer, bytes_length);
	     free(blob_buffer);
	     time1 = what_time_is_it_now();
             commu_time = commu_time + (time1 - time0);
        }
	//free(blob_buffer);//
     	close(newsockfd);
   }
   close(sockfd);

}


void steal_server(std::string thread_name){
   serve_steal( PORTNO );
}

void froward_result_to_gateway(std::string thread_name){
    while(1){
	dataBlob* blob = result_queue.Dequeue();
	send_result(blob, AP, PORTNO);
    }
}


void victim_client(){
    unsigned int number_of_jobs = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    g_t1 = 0;
    g_t0 = what_time_is_it_now();
    std::thread t1(client_compute, &net, number_of_jobs, "client_compute");
    std::thread t2(steal_server, "steal_server");
    std::thread t3(froward_result_to_gateway, "froward_result_to_gateway");
    t1.join();
    t2.join();
    t3.join();
}


void idle_client(){
    unsigned int number_of_jobs = IMG_NUM;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    g_t1 = 0;
    g_t0 = what_time_is_it_now();
    std::thread t1(steal_through_gateway, &net,  "steal_forward");
    std::thread t2(froward_result_to_gateway, "froward_result_to_gateway");
    t1.join();
    t2.join();
}



 






