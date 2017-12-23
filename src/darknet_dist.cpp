extern "C"{
#include "darknet.h"
}
#include "riot.h"

#include <fstream>

#define PORTNO 11111
#define SRV_IP "10.145.85.169"

#define AP "192.168.42.1"

#define PINK0    "192.168.42.16"
#define BLUE0    "192.168.42.14"
#define ORANGE0  "192.168.42.15"

#define PINK1    "192.168.42.11"
#define BLUE1    "192.168.42.12"
#define ORANGE1  "192.168.42.13"


//Generate ramdom jobs into the queue
void job_producer(unsigned int number_of_jobs, std::string thread_name){
    //std::ofstream ofs (thread_name + ".log", std::ofstream::out);
    std::thread::id this_id = std::this_thread::get_id();
    unsigned int size;
    char* data;
    for(unsigned int id = 0; id < number_of_jobs; id++){
        size=(id+1)*1000;
        data = (char*)malloc(size);
        put_job(data, size, id);
        //ofs << "Thread "<< this_id <<" put task "<< id <<", size is: " << size << std::endl;   
        std::cout << "Thread "<< this_id <<" put task "<< id <<", size is: " << size << std::endl;   

    }
    //ofs.close();
}



void job_consumer(unsigned int number_of_jobs, std::string thread_name){

    std::ofstream ofs (thread_name + ".log", std::ofstream::out);
    std::thread::id this_id = std::this_thread::get_id();
    unsigned int size;
    char* data;
    int id;
    for(unsigned int i = 0; i < number_of_jobs; i++){
	get_job((void**)&data, &size, &id);
        ofs << "Thread "<< this_id <<" got task "<< id <<", size is: " << size << std::endl;
    }
    ofs.close();
    free(data);

}

void busy_steal_jobs(unsigned int number_of_jobs){
   for(unsigned int i = 0; i < number_of_jobs; i++){
   	steal_and_push(SRV_IP, PORTNO, i);
   }
}


void test_busy_client(){

   std::thread remote_consumer(serve_steal, PORTNO);   
   std::thread local_producer(job_producer, 200, "local_producer");
   std::thread local_consumer(job_consumer, 10,  "local_consumer");

   remote_consumer.join();
   local_producer.join();
   local_consumer.join();

}

void test_spare_client(){

   std::thread remote_producer(busy_steal_jobs, 100);
   std::thread local_consumer(job_consumer, 90, "local_consumer");

   local_consumer.join();
   remote_producer.join();

}

void test_local(){
   
   std::thread local_producer(job_producer, 120, "local_producer");
   std::thread local_consumer(job_consumer, 10, "local_consumer");

   local_producer.join();
   local_consumer.join();

}


void test_detector_dist(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    printf("thresh %f hier_thresh %f\n",  thresh, hier_thresh);
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    char buff[256];
    char *input = buff;
    int j;
    float nms=.3;
#ifdef NNPACK
    nnp_initialize();
    net->threadpool = pthreadpool_create(4);
#endif

    while(1){

        strncpy(input, filename, 256);

#ifdef NNPACK
	image im = load_image_thread(input, 0, 0, net->c, net->threadpool);
	image sized = letterbox_image_thread(im, net->w, net->h, net->threadpool);
#else
	image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);

#endif
        layer l = net->layers[net->n-1];

        box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));
        float **masks = 0;
        if (l.coords > 4){
            masks = (float **)calloc(l.w*l.h*l.n, sizeof(float*));
            for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float *)calloc(l.coords-4, sizeof(float *));
        }

        float *X = sized.data;
        double t1=what_time_is_it_now();
	network_predict_dist(net, X);
        double t2=what_time_is_it_now();
	printf("%s: Predicted in %f s.\n", input, t2 - t1);
        get_region_boxes(l, im.w, im.h, net->w, net->h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

        if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
/*      if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions");
            cvWaitKey(0);
            cvDestroyAllWindows();
#endif
        }
*/
        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
        if (filename) break;
    }
#ifdef NNPACK
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}




int main(int argc, char **argv)
{

    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    test_detector_dist(argv[1], argv[2], argv[3], argv[4], .24, .5, outfile, fullscreen);

    return 0;
}

