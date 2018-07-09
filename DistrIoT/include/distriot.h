//Author: Zhuoran Zhao
//Last modified: 12/21/2017
//Application programming interface for distributed job stealing on IoT devices
//Use C-based socket programming for network communication 
//Use C++ multi-threading wrapper for pthread library
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include "job_queue.h"
#include "data_blob.h"
#include <fstream>
#include <string>
#include <list>
#include <iostream>
#include <chrono>
#include <thread>
//#define DEBUG_DISTRIOT 0
#include "config.h"

#ifndef DISTRIOT__H
#define DISTRIOT__H


extern char* addr_list[CLI_NUM];
extern double g_t0, g_t1;


double get_real_time_now();
void sock_error(const char *msg);
//Blocking call for getting and putting jobs
void put_job(void* data, unsigned int size, int id);

void put_result(void* data, unsigned int size, int id);
void get_job(void** data, unsigned int* size, int* id);
void get_result(void** data, unsigned int* size, int* id);

//Non-blocking calls for getting jobs and putting results
void try_get_job(void** data, unsigned int* size, int* id);
void try_get_result(void** data, unsigned int* size, int* id);
void read_sock(int sock, char* buffer, unsigned int bytes_length);
void write_sock(int sock, char* buffer, unsigned int bytes_length);
dataBlob* steal_and_return(const char *dest_ip, int portno);
void send_result(dataBlob* blob, const char *dest_ip, int portno);
in_addr_t ask_gateway(char* request_type, const char *gateway_ip, int portno);
void exec_control(int portno);
int get_client_id(const char* ip_addr);



#endif //DISTRIOT__H


