//Author: Zhuoran Zhao
//Last modified: 12/21/2017
//Application programming interface for distributed job stealing on IoT devices
//Use C-based socket programming for network communication 
//Use C++ multi-threading wrapper for pthread library
#include "distriot.h"

char* addr_list[CLI_NUM] = {BLUE1, ORANGE1, PINK1, BLUE0, ORANGE0, PINK0};
double g_t0, g_t1;


int get_client_id(const char* ip_addr){
   for(int i = 0; i < CLI_NUM; i++){
	if(strcmp(ip_addr, addr_list[i]) == 0){
		return i;
	}
   }
   return (-1);//
}

double get_real_time_now()
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec + now.tv_nsec*1e-9;
}

void sock_error(const char *msg)
{
    perror(msg);
    exit(1);
}


//Blocking call for getting and putting jobs
void put_job(void* data, unsigned int size, int id){
   dataBlob* blob = new dataBlob(data, size, id);
   job_queue.Enqueue(blob);
}

void put_result(void* data, unsigned int size, int id){
   dataBlob* blob = new dataBlob(data, size, id);
   result_queue.Enqueue(blob);
}

void get_job(void** data, unsigned int* size, int* id){
   dataBlob* blob = job_queue.Dequeue();
   *size = blob -> getSize();
   *data = blob -> getDataPtr();
   *id = blob -> getID();
   delete blob;
}

void get_result(void** data, unsigned int* size, int* id){
   dataBlob* blob = result_queue.Dequeue();
   *size = blob -> getSize();
   *data = blob -> getDataPtr();
   *id = blob -> getID();
   delete blob;
}


//Non-blocking calls for getting jobs and putting results
void try_get_job(void** data, unsigned int* size, int* id){
   dataBlob* blob = job_queue.TryDequeue();
   if(blob == NULL){
      *size = -1;
      *data = NULL;
      *id = -1;
   }
   else{
      *size = blob -> getSize();
      *data = blob -> getDataPtr();
      *id = blob -> getID();
   }
   delete blob;
}
void try_get_result(void** data, unsigned int* size, int* id){
   dataBlob* blob = result_queue.TryDequeue();
   if(blob == NULL){
      *size = -1;
      *data = NULL;
      *id = -1;
   }
   else{
      *size = blob -> getSize();
      *data = blob -> getDataPtr();
      *id = blob -> getID();
   }
   delete blob;
}




void read_sock(int sock, char* buffer, unsigned int bytes_length){
    size_t bytes_read = 0;
    int n;
    while (bytes_read < bytes_length){
	n = recv(sock, buffer + bytes_read, bytes_length - bytes_read, 0);
        if( n < 0 ) sock_error("ERROR reading socket");
        bytes_read += n;
        //std::cout << "Read size is " << bytes_read << std::endl;
    }

};

void write_sock(int sock, char* buffer, unsigned int bytes_length){
    size_t bytes_written = 0;
    int n;
    //std::cout << "Writing size is " << bytes_length << std::endl;
    while (bytes_written < bytes_length) {
	n = send(sock, buffer + bytes_written, bytes_length - bytes_written, 0);
        if( n < 0 ) sock_error("ERROR writing socket");
        bytes_written += n;
        //std::cout << "Written size is " << bytes_written << std::endl;
    }

};




dataBlob* steal_and_return(const char *dest_ip, int portno)
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

     char request_type[10] = "steals";
     write_sock(sockfd, request_type, 10);

     char *blob_buffer;
     unsigned int bytes_length;
     int job_id;

     read_sock(sockfd, (char*)&job_id, sizeof(job_id));
     read_sock(sockfd, (char*)&bytes_length, sizeof(bytes_length));
     blob_buffer = (char*)malloc(bytes_length);
     if (blob_buffer==NULL) {printf("(char*)malloc(bytes_length) failed\n"); exit (1);}
     read_sock(sockfd, blob_buffer, bytes_length);

     close(sockfd);
     dataBlob* ret = (new dataBlob((void*)blob_buffer, bytes_length, job_id)) ;
     return ret;

}


void send_result(dataBlob* blob, const char *dest_ip, int portno)
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

     char request_type[10] = "result";
     write_sock(sockfd, request_type, 10);

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


in_addr_t ask_gateway(char* request_type, const char *gateway_ip, int portno)
{
     int sockfd;
     struct sockaddr_in serv_addr;
     sockfd = socket(AF_INET, SOCK_STREAM, 0);
     if (sockfd < 0) 
        sock_error("ERROR opening socket");
     bzero((char *) &serv_addr, sizeof(serv_addr));
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = inet_addr(gateway_ip) ;
     serv_addr.sin_port = htons(portno);
     if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
	sock_error("ERROR connecting");
     //char request_type[10] = "steals";
     in_addr_t victim_addr;
     if(strcmp (request_type,"register") == 0){
         write_sock(sockfd, request_type, 10);
         unsigned int job_num = job_queue.Size();
	 std::cout << "The number of jobs to register at the gateway device : " << job_queue.Size() << "   "<< job_num << std::endl;
         write_sock(sockfd, (char*)&job_num, sizeof(job_num));
     }else if(strcmp (request_type,"steals") == 0){
         write_sock(sockfd, request_type, 10);
         read_sock(sockfd, (char*)&victim_addr, sizeof(in_addr_t));
     }else if(strcmp (request_type,"start") == 0){
         write_sock(sockfd, request_type, 10);
	 std::cout << "Send the start signal to the gateway" << std::endl;
     }else if(strcmp (request_type,"start_gw") == 0){
         write_sock(sockfd, request_type, 10);
	 std::cout << "Send the start signal to the gateway" << std::endl;
     }else{
	 std::cout << "Reqeust type is not supported" << std::endl;
     }
     close(sockfd);
     return victim_addr;
}



void exec_control(int portno)
{  
   std::cout << "Start exec_control at gateway" << std::endl;
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
   char request_type[10];
   newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
   read_sock(newsockfd, request_type, 10); 
   if(strcmp (request_type, "start_gw") == 0){
	std::cout << "Recving start signal at gateway from " << inet_ntoa(cli_addr.sin_addr) << std::endl;
	char start_msg[10] = "start";
 	for(int i = 0; i < ACT_CLI; i ++)
    	   ask_gateway(start_msg, addr_list[i], START_CTRL);
   }else if(strcmp (request_type, "start") == 0){
	std::cout << "Recving start signal from ..." << inet_ntoa(cli_addr.sin_addr) <<std::endl;
	std::cout << "Begin to do the work ..." << std::endl;
   }else{
	std::cout << "Something is wrong in the ctrl message ..." <<std::endl;
   }
   close(newsockfd);
   close(sockfd);
}







