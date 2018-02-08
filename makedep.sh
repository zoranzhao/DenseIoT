#Recompile the dependencies
cd ../darknet-nnpack/ && make && cd ../DistrIoT && make && cd ../DenseIoT && make clean && make
