//Author: Zhuoran Zhao
//Last modified: 12/01/2017
//A thread-safe unblocking queue to enable the distributed job-stealing execution
//Pending dependency-free execution jobs will be pushed into queue, proxy thread will steal
//from the queue upon remote request 
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <string>
#include <data_blob.h>


#ifndef JOB_QUEUE__H
#define JOB_QUEUE__H

template <class T>
class jobQueue {
   public:
     jobQueue();
     jobQueue(size_t capacity);
     ~jobQueue();
     T Dequeue();
     T TryDequeue();
     void Enqueue(T& item);
     unsigned int Size();

   private:
     std::queue <T> queue_;
     std::mutex q_mutex;
     std::condition_variable non_empty;
     std::condition_variable non_full;
     size_t capacity_;
};


template <class T>
jobQueue<T>::jobQueue(size_t capacity){
   capacity_ = capacity;
}

template <class T>
jobQueue<T>::jobQueue(){
   capacity_ = 32;
}

template <class T>
jobQueue<T>::~jobQueue(){

}


template <class T>
unsigned int jobQueue<T>::Size(){
   return queue_.size();
}

template <class T>
void jobQueue<T>::Enqueue(T& item)
{
    std::unique_lock<std::mutex> lk(q_mutex);
    if (queue_.size() == capacity_) {
        non_full.wait(lk);
    	queue_.push(item);
    }
    else {
    	queue_.push(item);
    }
    lk.unlock();
    non_empty.notify_one();
}

template <class T>
T jobQueue<T>::Dequeue()
{
    T ret;
    std::unique_lock<std::mutex> lk(q_mutex);
    if (queue_.empty()) {
        non_empty.wait(lk);
        ret = queue_.front();
        queue_.pop();
    }
    else {
        ret = queue_.front();
        queue_.pop();
    }
    lk.unlock();
    non_full.notify_one();
    return ret;
}

template <class T>
T jobQueue<T>::TryDequeue()
{
    T ret;
    std::unique_lock<std::mutex> lk(q_mutex);
    if (queue_.empty()) {
	ret = NULL;
    }
    else {
        ret = queue_.front();
        queue_.pop();
    }
    lk.unlock();
    non_full.notify_one();
    return ret;
}


extern jobQueue <dataBlob*> job_queue;
extern jobQueue <dataBlob*> result_queue;
extern jobQueue <int> ready_queue;

#endif //JOB_QUEUE__H
