//Author: Zhuoran Zhao
//Last modified: 12/01/2017
//Data structure holding a data pointer and data size
//I usually use three lines to describe one source file
//So I don't know what to say here... 
#include <data_blob.h>

dataBlob::dataBlob(){
	data_ = NULL;
	size_ = 0;
	id_ = -1;
};

dataBlob::dataBlob(void* data, size_t size, int id){
	data_ = data;
	size_ = size;
	id_ = id;
};



dataBlob::~dataBlob(){};

void dataBlob::setData(void* data){
	data_ = data;
};

void dataBlob::setSize(size_t size){
	size_ = size;
};

void* dataBlob::getDataPtr(){
	return data_;
};

size_t* dataBlob::getSizePtr(){
	return &size_;
};

size_t dataBlob::getSize(){
	return size_;
};

void dataBlob::setID(int id){
	id_ = id;
};

int dataBlob::getID(){
	return id_;
};

