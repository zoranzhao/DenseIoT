#include "darknet_dist_mr.h"
//ACT_CLI

void send_result_mr(dataBlob* blob, const char *dest_ip, int portno)
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


void data_map_reduce(network net, int portno)
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
   int job_id;
   unsigned int bytes_length;
   char *blob_buffer;
   std::list< std::string > cli_list;
   std::list< int > job_id_list;
   while(1){
     //Receive the data from a single client;
     int cli_id;
     newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
     read_sock(newsockfd, (char*)&cli_id, sizeof(cli_id));
     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
     blob_buffer = (char*)malloc(bytes_length);
     read_sock(newsockfd, blob_buffer, bytes_length);
     close(newsockfd);
     std::cout << "Receiving the entire input data to be distributed from client" << inet_ntoa(cli_addr.sin_addr) << std::endl;

     //Distribute the data 
     fork_input_mr(0, (float*)blob_buffer, net);
     for(int cli_cnt = 0; cli_cnt < ACT_CLI; cli_cnt ++ ){
        std::cout << "Sending the partition "<< cli_cnt << " to client" << addr_list[cli_cnt] << std::endl;
	bytes_length = input_ranges_mr[cli_cnt][0].w*input_ranges_mr[cli_cnt][0].h*net.layers[0].c*sizeof(float);
	dataBlob* blob = new dataBlob(part_data_mr[cli_cnt], bytes_length, cli_cnt); 
        send_result_mr(blob, addr_list[cli_cnt], portno);
       	free(part_data_mr[cli_cnt]);
	delete blob;
     }
     for(int ii = 0; ii < STAGES - 1; ii ++){
        for(int cli_cnt = 0; cli_cnt < ACT_CLI; cli_cnt ++ ){
     	  newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	  read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
	  read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	  blob_buffer = (char*)malloc(bytes_length);
	  read_sock(newsockfd, blob_buffer, bytes_length);
          std::cout << "Receiving IR result at layer"<<ii<<" from client" << inet_ntoa(cli_addr.sin_addr)<< " part "<< job_id << std::endl;
	  result_ir_data_deserialization_mr(net, job_id, (float*)blob_buffer, ii);
	  free(blob_buffer);
     	  close(newsockfd);
  	  cli_list.push_back( std::string(inet_ntoa(cli_addr.sin_addr)) );
          job_id_list.push_back(job_id);
	}

	for(int cli_cnt = 0; cli_cnt < ACT_CLI; cli_cnt ++ ){
	  std::string cur_addr = cli_list.front();
	  cli_list.pop_front();
	  int cur_id = job_id_list.front();
	  job_id_list.pop_front();
	  blob_buffer = (char*) req_ir_data_serialization_mr(net, cur_id, ii+1);
	  bytes_length = req_ir_data_size_mr[cur_id/PARTITIONS_W][cur_id%PARTITIONS_W][ii+1]* sizeof(float);
	  dataBlob* blob = new dataBlob(blob_buffer, bytes_length, cur_id); 
          std::cout << "Sending IR result at layer"<<(ii+1)<<" to client" << cur_addr << " part "<< cur_id << std::endl;
	  send_result_mr(blob, cur_addr.c_str(), portno);
	  free(blob_buffer);
	  delete blob;
	}
     }
     for(int cli_cnt = 0; cli_cnt < ACT_CLI; cli_cnt ++ ){
	  newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	  read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
	  read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	  blob_buffer = (char*)malloc(bytes_length);
	  read_sock(newsockfd, blob_buffer, bytes_length);
     	  close(newsockfd);
          std::cout << "Receiving IR result at layer from client" << inet_ntoa(cli_addr.sin_addr)<< " part "<< job_id << std::endl;
          recv_data_mr[job_id]=(float*)blob_buffer;
     }
     ready_queue.Enqueue(cli_id);
     break;
   }
   close(sockfd);
}


inline void gateway_compute_mr(network *netp, int cli_id)
{
    network net = *netp;
    int upto = STAGES-1;

    size_t stage_outs =  (stage_output_range.w)*(stage_output_range.h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  


    for(int part = 0; part < PARTITIONS; part ++){
       join_output_mr(part, recv_data_mr[part],  stage_out, upto, net);
       free(recv_data_mr[part]);
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


void gateway_service_mr(network net, std::string thread_name){

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
	gateway_compute_mr(&net, cli_id);


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


void gateway_sync_mr(network net, std::string thread_name){
    data_map_reduce(net, PORTNO);
}


void smart_gateway_mr(){
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_mr(0, STAGES-1, *netp);
    std::thread t1(gateway_sync_mr, net,  "gateway_sync_mr");
    std::thread t2(gateway_service_mr, net, "gateway_service_mr");
    exec_control(START_CTRL);
    g_t0 = what_time_is_it_now();
    g_t1 = 0;
    t1.join();
    t2.join();
}
