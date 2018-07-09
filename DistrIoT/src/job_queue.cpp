//Author: Zhuoran Zhao
//Last modified: 12/01/2017
//A thread-safe unblocking queue to enable the distributed job-stealing execution
//Pending dependency-free execution jobs will be pushed into queue, proxy thread will steal
//from the queue upon remote request 
#include <job_queue.h>

jobQueue <dataBlob*> job_queue;
jobQueue <dataBlob*> result_queue;
jobQueue <int> ready_queue;
