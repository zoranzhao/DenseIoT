#Recompile the dependencies
cd ../darknet-nnpack/ && make && cd ../DistrIoT && make clean && make && cd ../DenseIoT && make clean && make
