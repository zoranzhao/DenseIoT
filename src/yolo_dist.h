#include "darknet_dist.h"
#include "gateway.h"
#include "gateway_shuffle.h"
#include "client.h"
#include "client_shuffle.h"
#include "client_mr.h"
#include "gateway_mr.h"

void get_image(image* im, int* im_id){
    unsigned int size;
    float* data;
    int id;
    get_job((void**)&data, &size, &id);
    *im_id = id;
    im->data = data;
    printf("Processing task id is %d\n", id);
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




inline void steal_forward_local(network *netp, std::string thread_name){

    netp->truth = 0;
    netp->train = 0;
    netp->delta = 0;

    int part;
    network net = *netp;

    int startfrom = 0;
    int upto = STAGES-1;

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
    network net1 = reshape_network(0, STAGES-1, *netp1);

    network *netp2 = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp2, 1);
    network net2 = reshape_network(0, STAGES-1, *netp2);

    std::thread t1(client_compute, &net1, number_of_jobs, "client_compute");
    std::thread t2(steal_forward_local, &net2, "steal_forward");
    t1.join();
    t2.join();
}


void compute_local(){
    unsigned int number_of_jobs = 5;
    network *netp1 = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp1, 1);
    network net1 = reshape_network(0, STAGES-1, *netp1);

    std::thread t1(client_compute, &net1, number_of_jobs, "client_compute");
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


inline void forward_network_dist(network *netp, network orig)
{
    int part;
    network net = *netp;

    int startfrom = 0;
    int upto = STAGES-1;

    size_t stage_outs =  (stage_output_range.w)*(stage_output_range.h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
    float* stage_in = net.input; 
/*
    fork_input_reuse(startfrom, stage_in, net);
    fork_input(startfrom, stage_in, net);
    for(part = 0; part < PARTITIONS; part ++){
      printf("Putting jobs %d\n", part);
      put_job(part_data[part], input_ranges[part][startfrom].w*input_ranges[part][startfrom].h*net.layers[startfrom].c*sizeof(float), part);
    }
*/

    //fork_input(startfrom, stage_in, net);
    //fork_input_reuse(startfrom, stage_in, net);

    fork_input_reuse(startfrom, stage_in, net);
    for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
	for(int p_w = (p_h) % 2; p_w < PARTITIONS_W; p_w = p_w + 2){ 
		printf("Putting jobs %d\n", part_id[p_h][p_w]);
		put_job(reuse_part_data[part_id[p_h][p_w]], 
			reuse_input_ranges[part_id[p_h][p_w]][startfrom].w*reuse_input_ranges[part_id[p_h][p_w]][startfrom].h*net.layers[startfrom].c*sizeof(float), 
			part_id[p_h][p_w]);
	}
    }
    for(int p_h = 0; p_h < PARTITIONS_H; p_h++){
	for(int p_w = (p_h+1) % 2; p_w < PARTITIONS_W; p_w = p_w + 2){ 
		printf("Putting jobs %d\n", part_id[p_h][p_w]);
		put_job(reuse_part_data[part_id[p_h][p_w]], 
			reuse_input_ranges[part_id[p_h][p_w]][startfrom].w*reuse_input_ranges[part_id[p_h][p_w]][startfrom].h*net.layers[startfrom].c*sizeof(float), 
			part_id[p_h][p_w]);
	}
    }



    float* data;
    int part_id;
    unsigned int size;

    double t1 = 0.0;
    double t0 = get_real_time_now();

    for(part = 0; 1; part ++){
       try_get_job((void**)&data, &size, &part_id);
       if(data == NULL) {printf("%d parts out of the %d are processes locally\n", part, PARTITIONS); break;}
       std::cout << "=======================: " << part_id << std::endl;




       //net = forward_stage( part_id/PARTITIONS_W, part_id%PARTITIONS_W,  data, startfrom, upto, net);
       //net = forward_stage_reuse( part_id/PARTITIONS_W, part_id%PARTITIONS_W, data, startfrom, upto, net);
       if(part_id==3||part_id==5||part_id==7||part_id==1){
		std::cout << "For partition number: "<< part_id << ", the size of reuse data to be retrieved: "<< ir_data_size[part_id]<< std::endl;
		float* reuse_data = req_ir_data_serialization(net, part_id, startfrom, upto);
		req_ir_data_deserialization(net, part_id, reuse_data, startfrom, upto);
       }       

       net = forward_stage_reuse_full( part_id/PARTITIONS_W, part_id%PARTITIONS_W, data, startfrom, upto, net);


       if(part_id==0||part_id==2||part_id==4||part_id==6||part_id==8){
		std::cout << "For partition number: "<< part_id << ", the size of reuse data to be stored: "<< result_ir_data_size[part_id]<< std::endl;
		float* reuse_data = result_ir_data_serialization(net, part_id, startfrom, upto);
		result_ir_data_deserialization(net, part_id, reuse_data, startfrom, upto);
       }       


       join_output(part_id, net.layers[upto].output,  stage_out, upto, net);
       free(data);
    }

    for(part = part; part < PARTITIONS; part ++){
       get_result((void**)&data, &size, &part_id);
       printf("Getting result %d from other stealers\n", part_id);
       join_output(part_id, data,  stage_out, upto, net);
       free(data);
    }
    t1 = t1 + get_real_time_now() - t0;
    std::cout << "Processing overhead is: " << t1 << std::endl;


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


void send_data_prof(char *blob_buffer, unsigned int bytes_length, const char *dest_ip, int portno)
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
     write_sock(sockfd, (char*)&bytes_length, sizeof(bytes_length));
     write_sock(sockfd, blob_buffer, bytes_length);
     close(sockfd);
}


inline void forward_network_dist_prof(network *netp)
{
    network net = *netp;
    int i;
//Profiling of communication and computation time
/*
    double t0 = what_time_is_it_now();
    double t1 = 0;
    FILE *layer_exe;
    layer_exe = fopen("layer_exe_time.log", "w");  

    FILE *layer_comm;
    layer_comm = fopen("layer_comm_time.log", "w");  
*/

//Profiling of execution memory footprint
/*
    FILE *layer_input;
    FILE *layer_output;
    FILE *layer_weight; 
    FILE *layer_other; 

    layer_input  = fopen("layer_input.log", "w"); 
    layer_output = fopen("layer_output.log", "w");  
    layer_weight = fopen("layer_weight.log", "w");
    layer_other  = fopen("layer_other.log", "w");
*/
    
/*
    std::cout << "[";
    for(i = 0; i < net.n; ++i){//print layer list
	if(net.layers[i].type == CONVOLUTIONAL)
	  std::cout << "\"conv\", " ;
	if(net.layers[i].type == MAXPOOL)
	  std::cout << "\"maxpool\", " ;
	if(net.layers[i].type == ROUTE)
	  std::cout << "\"route\", " ;
	if(net.layers[i].type == REORG)
	  std::cout << "\"reorg\", "  ;
	if(net.layers[i].type == REGION)
	  std::cout << "\"region\""  ;
    }
    std::cout << "]"<<std::endl;
*/
    for(i = 0; i < net.n; ++i){//Iteratively execute the layers
        net.index = i;
        if(net.layers[i].delta){	       
            fill_cpu(net.layers[i].outputs * net.layers[i].batch, 0, net.layers[i].delta, 1);
        }

//Profiling of execution memory footprint
/*
        fprintf(layer_input, "%f\n", (float)(net.layers[i].inputs*sizeof(float))/1024.0/1024.0 );
        fprintf(layer_output, "%f\n", (float)(net.layers[i].outputs*sizeof(float))/1024.0/1024.0 );
        fprintf(layer_other, "%f\n", (float)(net.layers[i].out_c*4*sizeof(float))/1024.0/1024.0 );

        if(net.layers[i].type == CONNECTED)
           fprintf(layer_weight, "%f\n", (float)(net.layers[i].outputs*net.layers[i].inputs*sizeof(float))/1024.0/1024.0 );
	else
           fprintf(layer_weight, "%f\n", (float)(net.layers[i].nweights*sizeof(float))/1024.0/1024.0 );
*/
        net.layers[i].forward(net.layers[i], net);
//Profiling of communication and computation time
/*
        t0 = what_time_is_it_now();
        net.layers[i].forward(net.layers[i], net);
        t1 = what_time_is_it_now() - t0;
        fprintf(layer_exe, "%f\n", t1);

        t0 = what_time_is_it_now();
	send_data_prof((char*)(net.layers[i].output), net.layers[i].outputs*sizeof(float), BLUE1, PORTNO);
        t1 = what_time_is_it_now() - t0;
        fprintf(layer_comm, "%f\n", t1);
*/
        net.input = net.layers[i].output;  //Layer output
        if(net.layers[i].truth) {
            net.truth = net.layers[i].output;
        }
        printf("Index %d, Layer %s, input data byte num is: %ld, output data byte num is: %ld\n", 
		i, get_layer_string(net.layers[i].type), net.layers[i].inputs*sizeof(float), net.layers[i].outputs*sizeof(float));
    }

//Profiling of communication and computation time
/*
    fclose(layer_comm);
    fclose(layer_exe);
*/


//Profiling of execution memory footprint
/*
    fclose(layer_input);
    fclose(layer_weight);
    fclose(layer_output);
    fclose(layer_other);
*/
}

inline float *network_predict_dist_prof(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    //forward_network_dist_prof(net);
    forward_network_dist(net, orig);
    float *out = net->output;
    *net = orig;
    return out;
}



void client_compute_prof(network *netp, unsigned int number_of_jobs, std::string thread_name)
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


void toggle_gateway(){
    char start_msg[10] = "start_gw";
    ask_gateway(start_msg, GATEWAY, START_CTRL);

}

void server_prof(){
    recv_data_prof(PORTNO);
}



void victim_client_local(){
    unsigned int number_of_jobs = 1;
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_shuffle(0, STAGES-1, *netp);
    //network net = reshape_network(0, STAGES-1, *netp);
    std::thread t1(client_compute_prof, &net, number_of_jobs, "client_compute_prof");
    t1.join();
}


