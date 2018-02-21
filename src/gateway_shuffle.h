#include "darknet_dist.h"

void task_recorder(int portno);
inline void gateway_compute(network *netp, int cli_id);
void gateway_sync(std::string thread_name);

void gateway_service_shuffle(std::string thread_name){
    network *netp = load_network((char*)"cfg/yolo.cfg", (char*)"yolo.weights", 0);
    set_batch_network(netp, 1);
    network net = reshape_network_shuffle(0, STAGES-1, *netp);
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



void smart_gateway_shuffle(){
    std::thread t1(gateway_sync, "gateway_sync");
    std::thread t2(gateway_service_shuffle, "gateway_service");
    exec_control(START_CTRL);
    g_t0 = what_time_is_it_now();
    g_t1 = 0;
    t1.join();
    t2.join();
}

