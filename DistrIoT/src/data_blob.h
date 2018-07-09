//Author: Zhuoran Zhao
//Last modified: 12/01/2017
//Data structure holding a data pointer and data size
//I usually use three lines to describe one source file
//So I don't know what to say here... 
#include <iostream>
#include <string>

#ifndef DATA_BLOB__H
#define DATA_BLOB__H

class dataBlob {
   public:
     dataBlob();
     dataBlob(void* data, size_t size, int id);
     ~dataBlob();

     void setData(void* data);
     void setSize(size_t size);
     void setID(int id);

     void* getDataPtr();
     size_t* getSizePtr();
     size_t getSize();
     int getID();

   private:
     size_t size_;
     void* data_;
     int id_;
};






#endif //DATA_BLOB__H
