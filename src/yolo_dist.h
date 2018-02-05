#include "darknet_dist.h"
#define  THREAD_NUM 1


void remote_consumer(unsigned int number_of_jobs, std::string thread_name){
	serve_steal(number_of_jobs, PORTNO);
	//serve_steal_fake(number_of_jobs, PORTNO);

}


void remote_producer(unsigned int number_of_jobs, std::string thread_name){
   for(unsigned int i = 0; i < number_of_jobs; i++){
	std::cout << AP << "   " << "Steal only " << i <<std::endl;
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
        float *predictions = network_predict_dist_test(net, X);
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






//"cfg/coco.data" "cfg/yolo.cfg" "yolo.weights" "data/dog.jpg"
void local_consumer(unsigned int number_of_jobs, std::string thread_name)
{

    //load_images("local_producer");

    network *net = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    //network *net = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(net, 1);


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
	network_predict_dist_prof_exe(net, X);
	//network_predict_dist_test(net, X);
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


inline void steal_forward(network *netp){

    int part;
    network net = *netp;

    int startfrom = 0;
    int upto = 7;

    net = reshape_network(startfrom, upto, net);

    size_t stage_outs =  (net.layers[upto].out_w)*(net.layers[upto].out_h)*(net.layers[upto].out_c);
    float* stage_out = (float*) malloc( sizeof(float) * stage_outs );  
    float* stage_in = net.input; 

    float* data;
    int part_id;
    unsigned int size;

    dataBlob* blob = steal_and_return(AP, PORTNO);
    data = (float*)(blob -> getDataPtr());
    part_id = blob -> getID();
    size = blob -> getSize();

    net = forward_stage(part_id, data, startfrom, upto, net);

    
    free(data);
    delete blob;

    //results
    
    



    //Recover the network
    for(int i = 0; i < upto+1; ++i){
	layer l = net.layers[i];
	net.layers[i].h = original_ranges[i].h; net.layers[i].out_h = original_ranges[i].h/l.stride; 
	net.layers[i].w = original_ranges[i].w; net.layers[i].out_w = original_ranges[i].w/l.stride; 
	net.layers[i].outputs = net.layers[i].out_h * net.layers[i].out_w * l.out_c; 
	net.layers[i].inputs = net.layers[i].h * net.layers[i].w * l.c; 
    }

}


void produce_consume_serve(){
    std::thread lp(local_producer, 40, "local_producer1");
    std::thread lc(local_consumer, 20, "local_consumer1");
    std::thread rc(remote_consumer, 20, "remote_consumer1");
    lp.join();
    lc.join();
    rc.join();
}

//layer-based communication profiling
void consume_serve(){
    unsigned int layers = 32;
/*
    local_consumer(1, "local_consumer1");
    remote_consumer(layers, "remote_consumer1");
*/

    std::thread lc(local_consumer, 1, "local_consumer1");//pushing 
    std::thread rc(remote_consumer, layers, "remote_consumer1");
    rc.join();
    lc.join();


}
void steal_only(){
    unsigned int layers = 32;
    std::thread rp(remote_producer, layers, "remote_producer1");
    rp.join();
}
//layer-based communication profiling



void produce_serve(){
    std::thread lp(local_producer, 20, "local_producer1");
    std::thread rc(remote_consumer, 20, "remote_consumer1");
    lp.join();
    rc.join();
}



void consume_only(){
    std::thread lc(local_consumer, 5, "local_consumer1");
    lc.join();
}


void steal_consume(){
    std::thread rp(remote_producer, 20, "remote_producer1");
    std::thread lc(local_consumer, 20, "local_consumer1");
    rp.join();
    lc.join();
}


void produce_consume(){
    std::thread lp(local_producer, 1, "local_producer1");
    std::thread lc(local_consumer, 1, "local_consumer1");
    lp.join();
    lc.join();
}




