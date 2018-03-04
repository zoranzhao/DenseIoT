#include "darknet_dist_mr.h"
//ACT_CLI
void cal_workload_mapping(){
   int min = PARTITIONS/ACT_CLI;
   int remain = PARTITIONS%ACT_CLI;
   for(int i = ACT_CLI - 1; i >= 0; i--){
	if(remain > 0)
	  assigned_task_num[i] = min + 1;
	else
	  assigned_task_num[i] = min;
	remain --;
   }

}

inline int bind_port_client_share(int portno);


void send_result_share(dataBlob* blob, const char *dest_ip, int portno)
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

     char *blob_buffer;
     unsigned int bytes_length;
     int job_id;
     blob_buffer = (char*)(blob -> getDataPtr());
     job_id = blob -> getID();
     bytes_length = blob -> getSize();
     write_sock(sockfd, (char*)&job_id, sizeof(job_id));
     write_sock(sockfd, (char*)&bytes_length, sizeof(bytes_length));
     write_sock(sockfd, blob_buffer, bytes_length);
     close(sockfd);
}


void send_one_number(unsigned int number, const char *dest_ip, int portno)
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
     unsigned int bytes_length = number;
     write_sock(sockfd, (char*)&bytes_length, sizeof(bytes_length));
     close(sockfd);
}

void gateway_require_data(char* request_type, const char *cli_ip, int portno);



void task_share(network net, int number_of_images, int portno)
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
   double t0, t1;

   for(int id = 0; id < number_of_images; id++){
     //Receive the data from a single client;
     gateway_require_data("start", BLUE1, SMART_GATEWAY);
     int cli_id;
     newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
     g_t0 = what_time_is_it_now();

     t0 = what_time_is_it_now();
     read_sock(newsockfd, (char*)&cli_id, sizeof(cli_id));
     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
     blob_buffer = (char*)malloc(bytes_length);
     read_sock(newsockfd, blob_buffer, bytes_length);
     close(newsockfd);
     t1 = what_time_is_it_now();
     commu_time = commu_time + t1 - t0;
     if(print_gateway)
        std::cout << "Receiving the entire input data to be distributed from client" << inet_ntoa(cli_addr.sin_addr) << std::endl;
     cal_workload_mapping();
     //Distribute the data 
     fork_input(0, (float*)blob_buffer, net);
     t0 = what_time_is_it_now();
     int part = 0;
     for(int cli_cnt = 0; cli_cnt < ACT_CLI; cli_cnt ++ ){
	std::cout << "Sending to client" << addr_list[cli_cnt] << " Total task num is: " << assigned_task_num[cli_cnt] << std::endl;
        send_one_number(assigned_task_num[cli_cnt], addr_list[cli_cnt], portno );
	for(int i = 0; i < assigned_task_num[cli_cnt]; i ++ ){
		if(print_gateway)
		  std::cout << "Sending the partition "<< part << " to client" << addr_list[cli_cnt] << std::endl;
		bytes_length = input_ranges[part][0].w*input_ranges[part][0].h*net.layers[0].c*sizeof(float);
		dataBlob* blob = new dataBlob(part_data[part], bytes_length, part); 
		send_result_share(blob, addr_list[cli_cnt], portno);
	       	free(part_data[part]);
		delete blob;
		part++;
	}
     }
     t1 = what_time_is_it_now();
     commu_time = commu_time + t1 - t0;



     for(int part_cnt = 0; part_cnt < PARTITIONS; part_cnt ++ ){
	  newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	  t0 = what_time_is_it_now();
	  read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
	  read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
          std::cout << "Receiving stage result at layer from client" << inet_ntoa(cli_addr.sin_addr)<< " part "<< job_id << std::endl;
	  blob_buffer = (char*)malloc(bytes_length);
	  read_sock(newsockfd, blob_buffer, bytes_length);
	  t1 = what_time_is_it_now();
	  commu_time = commu_time + t1 - t0;
     	  close(newsockfd);
          recv_data[id][cli_id][job_id]=(float*)blob_buffer;
     }
     g_t1 = g_t1 + what_time_is_it_now() - g_t0;
     std::cout <<"Total latency is: "<< g_t1/((float)(id + 1)) << std::endl;
     std::cout <<"Communication latency is: "<< commu_time/((float)(id + 1)) << std::endl;
     std::cout << "Data from client " << cli_id << " has been fully collected and begin to compute ..."<< std::endl;
     ready_queue.Enqueue(cli_id);
   }
   close(sockfd);
}


inline void gateway_compute_share(network *netp, int cli_id, int image_id)
{
    network net = *netp;
    int upto = STAGES-1;

    size_t stage_outs =  (stage_output_range.w)*(stage_output_range.h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  


    for(int part = 0; part < PARTITIONS; part ++){
       join_output(part, recv_data[image_id][cli_id][part],  stage_out, upto, net);
       free(recv_data[image_id][cli_id][part]);
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


void gateway_service_share(network net, int number_of_images, std::string thread_name){

    net.truth = 0;
    net.train = 0;
    net.delta = 0;
    srand(2222222);
#ifdef NNPACK
    nnp_initialize();
    net.threadpool = pthreadpool_create(THREAD_NUM);
#endif
    int cli_id;
    int id = 0;
    for(id = 0; id < number_of_images; id++){
	cli_id = ready_queue.Dequeue();
	gateway_compute_share(&net, cli_id, id);


	#ifdef DEBUG_DIST
	image sized;
	sized.w = net.w; sized.h = net.h; sized.c = net.c;

	load_image_by_number(&sized, id);

	image **alphabet = load_alphabet();
	list *options = read_data_cfg((char*)"cfg/coco.data");
	char *name_list = option_find_str(options, (char*)"names", (char*)"data/names.list");
	char **names = get_labels(name_list);
	char filename[256];
	char outfile[256];
	float thresh = .24;
	float hier_thresh = .5;
	float nms=.3;
	sprintf(filename, "data/val2017/%d.jpg", id);
	sprintf(outfile, "%d", id);
	layer l = net.layers[net.n-1];
	float **masks = 0;
	if (l.coords > 4){
		    masks = (float **)calloc(l.w*l.h*l.n, sizeof(float*));
		    for(int j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float *)calloc(l.coords-4, sizeof(float *));
	}
	float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));
	image im = load_image_color(filename,0,0);
	box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
	if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
	save_image(im, outfile);
	free(boxes);
	free_ptrs((void **)probs, l.w*l.h*l.n);
	if (l.coords > 4){
			free_ptrs((void **)masks, l.w*l.h*l.n);
	}
	free_image(im);
	#endif
	free_image(sized);
    }
#ifdef NNPACK
    pthreadpool_destroy(net.threadpool);
    nnp_deinitialize();
#endif
}


void gateway_sync_share(network net, int number_of_images, std::string thread_name){
    task_share(net, number_of_images, PORTNO);
}


void smart_gateway_share(){
    int number_of_images = 4;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1); 
    network net = reshape_network(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    std::thread t1(gateway_sync_share, net, number_of_images, "gateway_sync_share");
    std::thread t2(gateway_service_share, net, number_of_images, "gateway_service_share");

    g_t1 = 0;
    t1.join();
    t2.join();
}

