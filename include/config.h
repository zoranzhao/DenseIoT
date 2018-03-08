#ifndef CONFIG__H
#define CONFIG__H


#define PORTNO 11111 //Service for job stealing and sharing
#define SMART_GATEWAY 11112 //Service for a smart gateway 
#define START_CTRL 11113 //Control the start and stop of a service
#define SRV "10.145.80.46"

#define GATEWAY "10.157.89.51"//"192.168.4.1"
#define AP "192.168.4.1"

#define BLUE1    "192.168.4.9"
#define ORANGE1  "192.168.4.8"
#define PINK1    "192.168.4.4"

#define BLUE0    "192.168.4.14"
#define ORANGE0  "192.168.4.15"
#define PINK0    "192.168.4.16"

#define CLI_NUM 6
#define IMG_NUM 4

#define DEBUG_DIST 0
#define STAGES 16
#define PARTITIONS_W_MAX 6
#define PARTITIONS_H_MAX 6
#define PARTITIONS_MAX 36
#define THREAD_NUM 1

extern int ACT_CLI;
extern int CUR_CLI;
extern int DATA_CLI;
extern int PARTITIONS;
extern int PARTITIONS_W;
extern int PARTITIONS_H;
//#define ACT_CLI 1
//#define CUR_CLI BLUE1

#endif
