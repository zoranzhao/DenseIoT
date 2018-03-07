#include "yolo_dist.h"

int main(int argc, char **argv)
{
    std::cout << "ACT_CLI " << atoi(argv[1]) << std::endl;
    std::cout << "CUR_CLI " << atoi(argv[1]) << std::endl;
    ACT_CLI = atoi(argv[1]);
    CUR_CLI = atoi(argv[1]);

    if(0 == strcmp(argv[2], "toggle")){  
	std::cout <<"toggle" << std::endl;
	toggle_gateway();
    }else if(0 == strcmp(argv[2], "steal")){
	PARTITIONS_W = atoi(argv[4]);
	PARTITIONS_H = atoi(argv[5]);
	PARTITIONS = PARTITIONS_W*PARTITIONS_H;
	std::cout << "P_W: "<<PARTITIONS_W << ", P_H: " << PARTITIONS_H << std::endl;
	std::cout <<"nonshuffle" << std::endl;
	if(0 == strcmp(argv[3], "idle")){
		std::cout <<"idle" << std::endl;
		idle_client();
	}else if(0 == strcmp(argv[3], "victim")){
		std::cout <<"victim" << std::endl;
		victim_client();
	}else if(0 == strcmp(argv[3], "gateway")){
		std::cout <<"gateway" << std::endl;
		smart_gateway();
	}


    }else if(0 == strcmp(argv[2], "shuffle")){
	std::cout <<"shuffle" << std::endl;
	PARTITIONS_W = atoi(argv[4]);
	PARTITIONS_H = atoi(argv[5]);
	PARTITIONS = PARTITIONS_W*PARTITIONS_H;
	std::cout << "P_W: "<<PARTITIONS_W << ", P_H: " << PARTITIONS_H << std::endl;
	if(0 == strcmp(argv[3], "idle")){
		std::cout <<"idle" << std::endl;
		idle_client_shuffle_v2();
	}else if(0 == strcmp(argv[3], "victim")){
		std::cout <<"victim" << std::endl;
		victim_client_shuffle_v2();
	}else if(0 == strcmp(argv[3], "gateway")){
		std::cout <<"gateway" << std::endl;
		smart_gateway_shuffle_v2();
	}


    }else if(0 == strcmp(argv[2], "mapreduce")){
	std::cout <<"mapreduce" << std::endl;
	PARTITIONS_W = atoi(argv[4]);
	PARTITIONS_H = atoi(argv[5]);
	PARTITIONS = PARTITIONS_W*PARTITIONS_H;
	std::cout << "P_W: "<<PARTITIONS_W << ", P_H: " << PARTITIONS_H << std::endl;
	if(0 == strcmp(argv[3], "idle")){
		std::cout <<"idle" << std::endl;
		idle_client_mr();
	}else if(0 == strcmp(argv[3], "victim")){
		std::cout <<"victim" << std::endl;
		busy_client_mr();
	}else if(0 == strcmp(argv[3], "gateway")){
		std::cout <<"gateway" << std::endl;
		smart_gateway_mr();
	}

    }else if(0 == strcmp(argv[2], "share")){
	std::cout <<"share" << std::endl;
	PARTITIONS_W = atoi(argv[4]);
	PARTITIONS_H = atoi(argv[5]);
	PARTITIONS = PARTITIONS_W*PARTITIONS_H;
	std::cout << "P_W: "<<PARTITIONS_W << ", P_H: " << PARTITIONS_H << std::endl;
	if(0 == strcmp(argv[3], "idle")){
		std::cout <<"idle" << std::endl;
		idle_client_share();
	}else if(0 == strcmp(argv[3], "victim")){
		std::cout <<"victim" << std::endl;
		busy_client_share();
	}else if(0 == strcmp(argv[3], "gateway")){
		std::cout <<"gateway" << std::endl;
		smart_gateway_share();
	}
    }

    //victim_client_shuffle();
    //idle_client_shuffle();
    //smart_gateway_shuffle();
    //server_prof();
    //victim_client_local();
    //victim_client_local_mr();

    return 0;
}

