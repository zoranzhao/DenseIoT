#include "darknet_dist.h"
#include "serialization_v2.h"

inline int merge_v2(int cli, int frame, int part){
   int all = 0;
   all = (cli << 16) + (frame << 8) + part;  
   return all;
}

inline int get_cli_v2(int all){
   int cli = 0;
   cli = all >> 16;  
   return cli;
}

inline int get_part_v2(int all){
   int part = 0;
   part = all & 0x00ff;  
   return part;
}

inline int get_frame_v2(int all){
   int frame = 0;
   frame = all >> 8;  
   frame = frame & 0x00ff;  
   return frame;
}


void notify_ir_ready(const char *dest_ip, int all,  int portno)
{
     int sockfd;
     int cli_frame_part = all; 
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
     char request_type[10] = "ir_data";
     write_sock(sockfd, request_type, 10);
     write_sock(sockfd, (char*)&cli_frame_part, sizeof(cli_frame_part));
     close(sockfd);
}


inline void gateway_compute(network *netp, int all);


void gateway_service_shuffle_v2(network net, std::string thread_name){

    net.truth = 0;
    net.train = 0;
    net.delta = 0;
    srand(2222222);
#ifdef NNPACK
    nnp_initialize();
    net.threadpool = pthreadpool_create(THREAD_NUM);
#endif
    int all;
    int id = 0;
    while(1){
	all = ready_queue.Dequeue();
	gateway_compute(&net, all);
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
	id += 1;
    }
#ifdef NNPACK
    pthreadpool_destroy(net.threadpool);
    nnp_deinitialize();
#endif
}

void collect_result(network net, int portno)
{  
   int task_total;
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
   int job_num;
   char request_type[10];
   std::list< std::string > job_list;

   int all;
   int job_id;
   int cli_id;
   int frame;
   char *blob_buffer;



   while(1){
     	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);

	if (newsockfd < 0) sock_error("ERROR on accept");
        read_sock(newsockfd, request_type, 10); 
	if(strcmp (request_type,"result") == 0){
	     read_sock(newsockfd, (char*)&all, sizeof(all));
	     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     blob_buffer = (char*)malloc(bytes_length);
	     read_sock(newsockfd, blob_buffer, bytes_length);
	     //std::cout << "Recving result from " << inet_ntoa(cli_addr.sin_addr) << "   ...    " << cli_addr.sin_addr.s_addr << std::endl;
	     cli_id = get_cli_v2(all);
             job_id = get_part_v2(all);
             frame = get_frame_v2(all);
	     std::cout << "[result]  .....Data from client " << cli_id << " part "<< job_id <<" is collected ... "<< " frame is: "<< frame <<std::endl;
	     //std::cout << "Data from client " << cli_id << " part "<< job_id <<" is collected ... "<< " size is: "<< bytes_length <<std::endl;
             frame_counters[cli_id][job_id]++;
	     //unsigned int recv_counters[IMG_NUM][CLI_NUM];
	     //float* recv_data[IMG_NUM][CLI_NUM][PARTITIONS];
             recv_data[frame][cli_id][job_id]=(float*)blob_buffer;
	     recv_counters[frame][cli_id] = recv_counters[frame][cli_id] + 1; 
	     //std::cout << "recv_counters "<< frame <<"..."<< cli_id <<"..."<< recv_counters[frame][cli_id] <<std::endl;
	     if(recv_counters[frame][cli_id] == PARTITIONS) {
		  std::cout << "Data from client " << cli_id << " have been fully collected ..." <<std::endl;
		  g_t1 = what_time_is_it_now() - g_t0;
		  std::cout << g_t1/(frame+1) << std::endl;
		  //std::cout << "Data from client " << cli_id << " has been fully collected and begin to compute ..."<< std::endl;
		  all = merge(cli_id, frame);
		  ready_queue.Enqueue(all);
	     }
        }
     	close(newsockfd);
   }
   close(sockfd);
}


void task_and_ir_recorder(network net, int portno)
{  
   int task_total;
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
   int job_num;
   char request_type[10];
   std::list< std::string > job_list;

   int all;
   int job_id;
   int cli_id;
   int frame;
   char *blob_buffer;

   bool g_t0_init = true;
   while(1){
     	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	if(g_t0_init){g_t0 = what_time_is_it_now(); g_t0_init=false;}
	if (newsockfd < 0) sock_error("ERROR on accept");
        read_sock(newsockfd, request_type, 10); 
        if(strcmp (request_type,"register") == 0){
	     std::cout << "Recving task registration from " << inet_ntoa(cli_addr.sin_addr) <<std::endl;
	     read_sock(newsockfd, (char*)&job_num, sizeof(job_num));
	     if(job_num > 0){
		job_list.push_back( std::string(inet_ntoa(cli_addr.sin_addr)) );
		std::cout << "Register task" << std::endl;
	     }else{
		job_list.remove(  std::string(inet_ntoa(cli_addr.sin_addr)) );
		std::cout << "Delete task" << std::endl;
             }
	}else if(strcmp (request_type,"steals") == 0){
	     //std::cout << "Recving quest from " << inet_ntoa(cli_addr.sin_addr) <<std::endl;
	     in_addr_t victim_addr;
	     if( !(job_list.empty()) ){
   	        std::string victim = job_list.front();
	        job_list.pop_front();
	        job_list.push_back(victim);
	        victim_addr = inet_addr(victim.c_str());
	     }else{
		victim_addr = inet_addr("0.0.0.0");
	     }
	     write_sock(newsockfd, (char*)(&victim_addr), sizeof(in_addr_t));
	     //std::cout << "Task list is ... : " << std::endl;
	     //for (std::list< std::string >::iterator it=job_list.begin(); it!=job_list.end(); ++it){
		//std::cout <<  *it << std::endl;
	     //}
        }else if(strcmp (request_type,"ir_data") == 0){//TODO IR data from different images and clients
     	     read_sock(newsockfd, (char*)&all, sizeof(all));
	     cli_id = get_cli_v2(all);
             job_id = get_part_v2(all);
             frame = get_frame_v2(all);
	     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     blob_buffer = (char*)malloc(bytes_length);
	     read_sock(newsockfd, blob_buffer, bytes_length);
             //std::cout << "[ir_data]  ..... Recved reuse data for partition number: "<< job_id << std::endl;
	     result_ir_data_deserialization(net, job_id, (float*)blob_buffer, 0, STAGES-1);
	     free(blob_buffer);
	     if( get_client_id( inet_ntoa(cli_addr.sin_addr) ) != cli_id )
	        notify_ir_ready(addr_list[cli_id], all, PORTNO);//TODO
        }else if(strcmp (request_type,"ir_data_r") == 0){//TODO IR data from different images and clients
	     //get_local_coverage_v2(part_id, frame, resource);
     	     read_sock(newsockfd, (char*)&all, sizeof(all));
	     cli_id = get_cli_v2(all);
             job_id = get_part_v2(all);
             frame = get_frame_v2(all);
	     //std::cout << "[ir_data_r]  ..... Getting a ir reqeust, frame number is: " << frame<<", resource is: "<< cli_id << std::endl;
	     bool *req = (bool*)malloc(4*sizeof(bool));
             read_sock(newsockfd, (char*)req, 4*sizeof(bool));
	     unsigned int reuse_size;
	     float* reuse_data = req_ir_data_serialization_v2(net, job_id, 0, STAGES-1, req, &reuse_size);
	     free(req);
             write_sock(newsockfd, (char*)&(reuse_size), sizeof(reuse_size));
             write_sock(newsockfd, (char*)reuse_data, reuse_size);
             free(reuse_data);
        }
     	close(newsockfd);
   }
   close(sockfd);
}
/*
void shuffle_task_order(){
    int index = 0;
    for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
	for(int p_w = (p_h) % 2; p_w < PARTITIONS_W; p_w = p_w + 2){ 
		task_list[index]=part_id[p_h][p_w];
		index++;
	}
    }
    for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
	for(int p_w = (p_h+1) % 2; p_w < PARTITIONS_W; p_w = p_w + 2){ 
		task_list[index]=part_id[p_h][p_w];
		index++;
	}
    }
}
*/

void gateway_sync_and_ir(network net, std::string thread_name){
    init_recv_counter();
    clear_coverage_v2();

    task_and_ir_recorder(net, SMART_GATEWAY);
}

void gateway_collect_result(network net, std::string thread_name){
    collect_result(net, PORTNO);
}

void smart_gateway_shuffle_v2(){
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_shuffle(0, STAGES-1, *netp);
    std::thread t1(gateway_sync_and_ir, net, "gateway_sync_and_ir");
    std::thread t2(gateway_service_shuffle_v2, net, "gateway_service");
    std::thread t3(gateway_collect_result, net, "gateway_collect_result");
    exec_control(START_CTRL);
    g_t1 = 0;
    t1.join();
    t2.join();
    t3.join();
}

