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


int main() 
{
   //test_local();
   test_busy_client();
   //test_spare_client();
   return 0;
}

