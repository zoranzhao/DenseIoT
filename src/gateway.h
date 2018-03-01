#include "darknet_dist.h"

inline int get_cli(int all){
   int cli = 0;
   cli = all >> 8;  
   return cli;
}

inline int get_part(int all){
   int part = 0;
   part = all & 0x00ff;  
   return part;
}

inline int get_frame(int all){
   int part = 0;
   part = all & 0x00ff;  
   return part;
}

inline int merge(int cli, int part);

void init_recv_counter(){
//unsigned int recv_counters[IMG_NUM][CLI_NUM];
//unsigned int frame_counters[CLI_NUM][PARTITIONS];
    for(int i; i < IMG_NUM; i ++){
	for(int j; j < CLI_NUM; j ++){
	   recv_counters[i][j] = 0; 
	}
    }
    for(int i; i < CLI_NUM; i ++){
	for(int j; j < PARTITIONS; j ++){
	   frame_counters[i][j] = 0; 
	}
    }

}


void task_recorder(int portno)
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

   int job_id;
   char *blob_buffer;
   init_recv_counter();

   bool g_t0_init = true;
   while(1){
     	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	if(g_t0_init){g_t0 = what_time_is_it_now(); g_t0_init=false;}
	if (newsockfd < 0) sock_error("ERROR on accept");
        read_sock(newsockfd, request_type, 10); 
        if(strcmp (request_type,"register") == 0){
	     //std::cout << "Recving task registration from " << inet_ntoa(cli_addr.sin_addr) <<std::endl;
	     read_sock(newsockfd, (char*)&job_num, sizeof(job_num));
	     if(job_num > 0){
		job_list.push_back( std::string(inet_ntoa(cli_addr.sin_addr)) );
		//std::cout << "Register task" << std::endl;
	     }else{
		job_list.remove(  std::string(inet_ntoa(cli_addr.sin_addr)) );
		//std::cout << "Delete task" << std::endl;
             }
	     //std::cout << "Task list is ... : " << std::endl;
	     //for (std::list< std::string >::iterator it=job_list.begin(); it!=job_list.end(); ++it){
		//std::cout <<  *it << std::endl;
	     //}
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
        }else if(strcmp (request_type,"result") == 0){
             int all;
	     read_sock(newsockfd, (char*)&all, sizeof(all));
	     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     blob_buffer = (char*)malloc(bytes_length);
	     read_sock(newsockfd, blob_buffer, bytes_length);
	     std::cout << "Recving result from " << inet_ntoa(cli_addr.sin_addr) << "   ...    " << cli_addr.sin_addr.s_addr << std::endl;
	     int cli_id = get_cli(all);
             job_id = get_part(all);
	     std::cout << "Data from client " << cli_id << " part "<< job_id <<" is collected ... "<< " size is: "<< bytes_length <<std::endl;
	     //std::cout << "Data from client " << cli_id << " part "<< job_id <<" is collected ... "<< " size is: "<< bytes_length <<std::endl;
             int frame_num = frame_counters[cli_id][job_id];
             frame_counters[cli_id][job_id]++;
	     //unsigned int recv_counters[IMG_NUM][CLI_NUM];
	     //float* recv_data[IMG_NUM][CLI_NUM][PARTITIONS];
             recv_data[frame_num][cli_id][job_id]=(float*)blob_buffer;
	     recv_counters[frame_num][cli_id] = recv_counters[frame_num][cli_id] + 1; 
	     std::cout << "recv_counters "<< frame_num <<"..."<< cli_id <<"..."<< recv_counters[frame_num][cli_id] <<std::endl;
	     if(recv_counters[frame_num][cli_id] == PARTITIONS) {
		  //std::cout << "Data from client " << cli_id << " have been fully collected ..." <<std::endl;
		  g_t1 = what_time_is_it_now() - g_t0;
		  std::cout << g_t1/(frame_num+1) << std::endl;
		  std::cout << "Data from client " << cli_id << " has been fully collected and begin to compute ..."<< std::endl;
		  all = merge(cli_id, frame_num);
		  ready_queue.Enqueue(all);
	     }
        }

     	close(newsockfd);
   }
   close(sockfd);
}



inline void gateway_compute(network *netp, int cli_id)
{
    network net = *netp;
    int upto = STAGES-1;

    size_t stage_outs =  (stage_output_range.w)*(stage_output_range.h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  


    for(int part = 0; part < PARTITIONS; part ++){
       join_output(part, recv_data[get_frame(cli_id)][get_cli(cli_id)][part],  stage_out, upto, net);
       free(recv_data[get_frame(cli_id)][get_cli(cli_id)][part]);
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


void gateway_service(std::string thread_name){
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network(0, STAGES-1, *netp);
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
    while(1){
	cli_id = ready_queue.Dequeue();
	g_t1 = what_time_is_it_now() - g_t0;
	std::cout << g_t1 << std::endl;
	std::cout << "Data from client " << cli_id << " has been fully collected and begin to compute ..."<< std::endl;
	gateway_compute(&net, cli_id);


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

void gateway_sync(std::string thread_name){
    task_recorder(SMART_GATEWAY);
}



void smart_gateway(){
    std::thread t1(gateway_sync, "gateway_sync");
    std::thread t2(gateway_service, "gateway_service");
    exec_control(START_CTRL);
    g_t0 = what_time_is_it_now();
    g_t1 = 0;
    t1.join();
    t2.join();
}

