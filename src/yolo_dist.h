#include "darknet_dist.h"




void steal(unsigned int number_of_jobs, std::string thread_name){
   for(unsigned int i = 0; i < number_of_jobs; i++){
	std::cout << AP << "   " << "Steal " << i << "th task!" <<std::endl;
   	dataBlob* data = steal_and_return(AP, PORTNO);
   }
}





//Load images by name
void load_image_by_number(image* img, unsigned int id){
    int h = img->h;
    int w = img->w;
    char filename[256];
    sprintf(filename, "data/val2017/%d.jpg", id);
//#ifdef NNPACK
//    int c = img->c;
//    image im = load_image_thread(filename, 0, 0, c, net->threadpool);
//    image sized = letterbox_image_thread(im, w, h, net->threadpool);
//#else
    image im = load_image_color(filename, 0, 0);
    image sized = letterbox_image(im, w, h);
//#endif
    free_image(im);
    img->data = sized.data;
}



//Load images from file into the shared dequeue 
void local_producer(unsigned int number_of_jobs, std::string thread_name){
    int h;
    int w;
    int c;
    //std::thread::id this_id = std::this_thread::get_id();
    extract_network_cfg_input((char*)"cfg/yolo.cfg", &h, &w, &c);
    char filename[256];
    unsigned int id = 0;//5000 > id > 0
    unsigned int size;

#ifdef DEBUG_DIST
    std::ofstream ofs (thread_name + ".log", std::ofstream::out);
#endif 
    for(id = 0; id < number_of_jobs; id ++){
         sprintf(filename, "data/val2017/%d.jpg", id);
//#ifdef NNPACK
//         image im = load_image_thread(filename, 0, 0, c, net->threadpool);
//         image sized = letterbox_image_thread(im, w, h, net->threadpool);
//#else
         image im = load_image_color(filename, 0, 0);
         image sized = letterbox_image(im, w, h);
//#endif
         size = (w)*(h)*(c);
         put_job(sized.data, size*sizeof(float), id);
#ifdef DEBUG_DIST
	 ofs << "Put task "<< id <<", size is: " << size << std::endl;  
#endif 
         free_image(im);
    }
#ifdef DEBUG_DIST
    ofs.close();
#endif 

    //free_network(net);
    //ofs << "Put task "<< id <<", size is: " << size << std::endl;   
    //std::cout << "Thread "<< this_id <<" put task "<< id <<", size is: " << size << std::endl; 
    //return im;  
}

void get_image(image* im, int* im_id){
    unsigned int size;
    float* data;
    int id;
    get_job((void**)&data, &size, &id);
    *im_id = id;
    im->data = data;
    printf("Processing task id is %d\n", id);
}


void serve_steal_and_gather_result(int portno)
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
   unsigned int id;

#ifdef DEBUG_DISTRIOT
   std::ofstream ofs ("layer_data_time.log", std::ofstream::out);
#endif 


   char request_type[10];
   while(1){
	//Recieving stealing request from client devices
	//TODO Need to handle fail on stealing

     	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	if (newsockfd < 0) sock_error("ERROR on accept");
        read_sock(newsockfd, request_type, 10); 
        if(strcmp (request_type,"result") == 0){
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
	     try_get_job((void**)&blob_buffer, &bytes_length, &job_id);
	     if(blob_buffer == NULL) {bytes_length = 4; blob_buffer = (char*)malloc(bytes_length+1); }
	     //std::cout << "Got job "<< job_id << " from queue, "<<"job size is: "<< bytes_length <<", sending job "  << std::endl;
	     write_sock(newsockfd, (char*)&job_id, sizeof(job_id));
	     write_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     write_sock(newsockfd, blob_buffer, bytes_length);
	     free(blob_buffer);
        }

#ifdef DEBUG_DISTRIOT
	ofs  << (t1 - t0) <<std::endl;  
#endif 
	//free(blob_buffer);//
     	close(newsockfd);
   }
#ifdef DEBUG_DISTRIOT
   ofs.close();
#endif 
   close(sockfd);

}


void steal_server(std::string thread_name){
   serve_steal_and_gather_result( PORTNO );
}


//"cfg/imagenet1k.data" "cfg/densenet201.cfg" "densenet201.weights" "data/dog.jpg"
//char *datacfg, char *cfgfile, char *weightfile, char *filename, int top
void run_densenet()
{
    int top = 5;

    //network *net = load_network("cfg/alexnet.cfg", "alexnet.weights", 0);
    //network *net = load_network("cfg/vgg-16.cfg", "vgg-16.weights", 0);
    network *net = load_network("cfg/resnet50.cfg", "resnet50.weights", 0);
    //network *net = load_network("cfg/resnet152.cfg", "resnet152.weights", 0);
    //network *net = load_network("cfg/densenet201.cfg", "densenet201.weights", 0);
    //network *net = load_network("cfg/densenet201.cfg", "densenet201.weights", 0);
    set_batch_network(net, 1);
    srand(2222222);

#ifdef NNPACK
    nnp_initialize();
    net->threadpool = pthreadpool_create(THREAD_NUM);
#endif

    list *options = read_data_cfg("cfg/imagenet1k.data");
    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = (int* )calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if("data/dog.jpg"){
            strncpy(input, "data/dog.jpg", 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image r = letterbox_image(im, net->w, net->h);
        //resize_network(net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict_dist(net, X);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
            //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        break;
    }

#ifdef NNPACK
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}


void local_consumer(network *netp, unsigned int number_of_jobs, std::string thread_name)
{

    network *net = netp;
    srand(2222222);
#ifdef NNPACK
    nnp_initialize();
    net->threadpool = pthreadpool_create(THREAD_NUM);
#endif

#ifdef DEBUG_DIST
    image **alphabet = load_alphabet();
    list *options = read_data_cfg((char*)"cfg/coco.data");
    char *name_list = option_find_str(options, (char*)"names", (char*)"data/names.list");
    char **names = get_labels(name_list);
    char filename[256];
    char outfile[256];
    float thresh = .24;
    float hier_thresh = .5;
    float nms=.3;
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
        double t1=what_time_is_it_now();
	network_predict_dist(net, X);
        double t2=what_time_is_it_now();

#ifdef DEBUG_DIST
	sprintf(filename, "data/val2017/%d.jpg", id);
	sprintf(outfile, "%d", id);
        layer l = net->layers[net->n-1];
        float **masks = 0;
        if (l.coords > 4){
            masks = (float **)calloc(l.w*l.h*l.n, sizeof(float*));
            for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float *)calloc(l.coords-4, sizeof(float *));
        }
        float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));
	printf("%s: Predicted in %f s.\n", filename, t2 - t1);
        image im = load_image_color(filename,0,0);
        box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
        get_region_boxes(l, im.w, im.h, net->w, net->h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
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
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}


void local_consumer_prof(network *netp, unsigned int number_of_jobs, std::string thread_name)
{

    network *net = netp;
    srand(2222222);
#ifdef NNPACK
    nnp_initialize();
    net->threadpool = pthreadpool_create(THREAD_NUM);
#endif

#ifdef DEBUG_DIST
    image **alphabet = load_alphabet();
    list *options = read_data_cfg((char*)"cfg/coco.data");
    char *name_list = option_find_str(options, (char*)"names", (char*)"data/names.list");
    char **names = get_labels(name_list);
    char filename[256];
    char outfile[256];
    float thresh = .24;
    float hier_thresh = .5;
    float nms=.3;
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
        double t1=what_time_is_it_now();
	network_predict_dist_prof(net, X);
        double t2=what_time_is_it_now();

#ifdef DEBUG_DIST
	sprintf(filename, "data/val2017/%d.jpg", id);
	sprintf(outfile, "%d", id);
        layer l = net->layers[net->n-1];
        float **masks = 0;
        if (l.coords > 4){
            masks = (float **)calloc(l.w*l.h*l.n, sizeof(float*));
            for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float *)calloc(l.coords-4, sizeof(float *));
        }
        float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));
	printf("%s: Predicted in %f s.\n", filename, t2 - t1);
        image im = load_image_color(filename,0,0);
        box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
        get_region_boxes(l, im.w, im.h, net->w, net->h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
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
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}



inline void steal_forward_with_gateway(network *netp, std::string thread_name){
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
    int upto = 7;
    size_t stage_outs =  (stage_output_range.w)*(stage_output_range.h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
    float* stage_in = net.input; 
    float* data;
    int part_id;
    unsigned int size;
    char steal[10] = "steals";
    double t0;
    double t1 = 0; 
    struct sockaddr_in addr;

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
        //t1 =  get_real_time_now() - t0;
        //std::cout << "Steal cost is: "<<t1<< std::endl;
        //t0 = get_real_time_now();
	data = (float*)(blob -> getDataPtr());
	part_id = blob -> getID();
	size = blob -> getSize();
	std::cout << "Steal part " << part_id <<", size is: "<< size <<std::endl;
	net = forward_stage(part_id/PARTITIONS_W, part_id%PARTITIONS_W, data, startfrom, upto, net);
	free(data);
	blob -> setData((void*)(net.layers[upto].output));
	blob -> setSize(net.layers[upto].outputs*sizeof(float));
        //t1 =  get_real_time_now() - t0;
        //std::cout << "Exec cost is: "<<t1<< std::endl;
        //t0 =  get_real_time_now();
	send_result(blob, inet_ntoa(addr.sin_addr), PORTNO);
	//send_result(blob, AP, SMART_GATEWAY);
        //t1 =  get_real_time_now() - t0;
        //std::cout << "Send result cost is: "<<t1<< std::endl;
	delete blob;
    }
#ifdef NNPACK
    pthreadpool_destroy(netp->threadpool);
    nnp_deinitialize();
#endif
}


inline void steal_forward_local(network *netp, std::string thread_name){

    netp->truth = 0;
    netp->train = 0;
    netp->delta = 0;

    int part;
    network net = *netp;

    int startfrom = 0;
    int upto = 7;

    size_t stage_outs =  (stage_output_range.w)*(stage_output_range.h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
    float* stage_in = net.input; 

    float* data;
    int part_id;
    unsigned int size;

    while(1){
        get_job((void**)&data, &size, &part_id);
	std::cout << "Steal part " << part_id <<", size is: "<< size <<std::endl;
	net = forward_stage(part_id/PARTITIONS_W, part_id%PARTITIONS_W,  data, startfrom, upto, net);
	free(data);
        put_result((void*)(net.layers[upto].output), net.layers[upto].outputs*sizeof(float), part_id);
    }

}


void compute_with_local_stealer(){
    unsigned int number_of_jobs = 5;
    network *netp1 = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp1, 1);
    network net1 = reshape_network(0, 7, *netp1);

    network *netp2 = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp2, 1);
    network net2 = reshape_network(0, 7, *netp2);

    std::thread t1(local_consumer, &net1, number_of_jobs, "local_consumer");
    std::thread t2(steal_forward_local, &net2, "steal_forward");
    t1.join();
    t2.join();
}


void compute_local(){
    unsigned int number_of_jobs = 5;
    network *netp1 = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp1, 1);
    network net1 = reshape_network(0, STAGES-1, *netp1);

    std::thread t1(local_consumer, &net1, number_of_jobs, "local_consumer");
    t1.join();
}




void recv_data_prof(int portno)
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
   unsigned int id;
   blob_buffer = (char*)malloc(48000000);
   while(1){
     	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
	if (newsockfd < 0) sock_error("ERROR on accept");
	read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	read_sock(newsockfd, blob_buffer, bytes_length); 
     	close(newsockfd);
   }
   close(sockfd);

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

   while(1){
     	newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
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
	     read_sock(newsockfd, (char*)&job_id, sizeof(job_id));
	     read_sock(newsockfd, (char*)&bytes_length, sizeof(bytes_length));
	     blob_buffer = (char*)malloc(bytes_length);
	     read_sock(newsockfd, blob_buffer, bytes_length);
	     int cli_id = 0;
	     std::cout << "Data from client " << cli_id << " part "<< job_id <<" is collected ... "<< " size is: "<< bytes_length <<std::endl;
             recv_data[cli_id][job_id]=(float*)blob_buffer;
	     recv_counters[cli_id] = recv_counters[cli_id] + 1; 
	     if(recv_counters[cli_id] == PARTITIONS) {
		  std::cout << "Data from client " << cli_id << " have been fully collected ..." <<std::endl;
                  recv_counters[cli_id] = 0;
		  ready_queue.Enqueue(cli_id);
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
       join_output(part, recv_data[cli_id][part],  stage_out, upto, net);
       free(recv_data[cli_id][part]);
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


void toggle_gateway(){
    char start_msg[10] = "start_gw";
    ask_gateway(start_msg, GATEWAY, START_CTRL);

}

void smart_gateway(){
    std::thread t1(gateway_sync, "gateway_sync");
    std::thread t2(gateway_service, "gateway_service");
    exec_control(START_CTRL);
    t1.join();
    t2.join();
}


void server_prof(){
    recv_data_prof(PORTNO);
}


void idle_client(){
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    std::thread t1(steal_forward_with_gateway, &net,  "steal_forward");
    t1.join();
}


void victim_result_to_gateway(std::string thread_name){
    while(1){
	dataBlob* blob = result_queue.Dequeue();
	send_result(blob, AP, SMART_GATEWAY);
    }
}



void victim_client(){
    unsigned int number_of_jobs = 1;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network(0, STAGES-1, *netp);
    exec_control(START_CTRL);
    g_t1 = 0;
    g_t0 = what_time_is_it_now();
    std::thread t1(local_consumer, &net, number_of_jobs, "local_consumer");
    std::thread t2(steal_server, "steal_server");
    std::thread t3(victim_result_to_gateway, "victim_result_to_gateway");
    t1.join();
    t2.join();
    t3.join();
}


void victim_client_local(){
    unsigned int number_of_jobs = 1;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network(0, STAGES-1, *netp);
    std::thread t1(local_consumer_prof, &net, number_of_jobs, "local_consumer_prof");
    t1.join();
}

 



