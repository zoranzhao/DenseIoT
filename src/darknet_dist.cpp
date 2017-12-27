#include "darknet.h"
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

#define DEBUG_DIST 1

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

void load_images(std::string thread_name){
    network *net = parse_network_cfg("cfg/yolo.cfg");
    char filename[256];
    int id = 0;//5000 > id > 0
    unsigned int size;
#ifdef DEBUG_DIST
    std::ofstream ofs (thread_name + ".log", std::ofstream::out);
#endif
    for(id = 0; id < 100; id ++){
         sprintf(filename, "data/val2017/%d.jpg", id);
#ifdef DEBUG_DIST
	 ofs << "Task file is: "<< filename << std::endl;  
#endif 
//#ifdef NNPACK
//         image im = load_image_thread(filename, 0, 0, net->c, net->threadpool);
//         image sized = letterbox_image_thread(im, net->w, net->h, net->threadpool);
//#else
         image im = load_image_color(filename, 0, 0);
         image sized = letterbox_image(im, net->w, net->h);
//#endif

         size = (net->w)*(net->h)*(net->c);

         put_job(sized.data, size, 0);
#ifdef DEBUG_DIST
	 ofs << "Put task "<< id <<", size is: " << size << std::endl;  
#endif 
    }

    //ofs << "Put task "<< id <<", size is: " << size << std::endl;   
    //std::cout << "Thread "<< this_id <<" put task "<< id <<", size is: " << size << std::endl; 
    //return im;  
}

void get_image(image* im){
    unsigned int size;
    float* data;
    int id;
    get_job((void**)&data, &size, &id);
    im->data = data;
}

//"cfg/coco.data" "cfg/yolo.cfg" "yolo.weights" "data/dog.jpg"
void test_detector_dist(std::string thread_name)
{






    network *net = load_network("cfg/yolo.cfg", "yolo.weights", 0);
    set_batch_network(net, 1);

    load_images("local_producer");
    srand(2222222);
#ifdef NNPACK
    nnp_initialize();
    net->threadpool = pthreadpool_create(4);
#endif

    int j;
    int id = 0;//5000 > id > 0
    for(id = 0; id < 5; id ++){
        image sized;
	sized.w = net->w; sized.h = net->h; sized.c = net->c;

        get_image(&sized);
        printf("Input image size is %d\n", sized.w*sized.h*sized.c);
        printf("Input image w is %d\n", sized.w);
        printf("Input image h is %d\n", sized.h);
        printf("Input image c is %d\n", sized.c);

        float *X = sized.data;
        double t1=what_time_is_it_now();
	network_predict_dist(net, X);
        double t2=what_time_is_it_now();
#ifdef DEBUG_DIST
	char filename[256];
	char outfile[256];
        float thresh = .24;
        float hier_thresh = .5;
	float nms=.3;
        image **alphabet = load_alphabet();
        list *options = read_data_cfg("cfg/coco.data");
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
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
#endif
        //free_image(im);
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

    }
#ifdef NNPACK
    pthreadpool_destroy(net->threadpool);
    nnp_deinitialize();
#endif
}




int main(int argc, char **argv)
{
    //std::thread local_producer(load_images, "local_producer");
    //std::thread local_consumer(test_detector_dist, "local_consumer");

    //local_producer.join();
    //local_consumer.join();
    test_detector_dist("local_consumer");
    return 0;
}

