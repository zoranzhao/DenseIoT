NNPACK=1
ARM_NEON=1
OPENMP=0
DEBUG=0
VPATH=./src
EXEC=darknet_dist
OBJDIR=./obj/

CC=gcc
CXX=g++
AR=ar
ARFLAGS=rcs

LDFLAGS= -lm -pthread 
CFLAGS=-Wall -fPIC
DARKNET= ../darknet-nnpack
DISTRIOT= ../DistrIoT
COMMON= -I$(DARKNET)/include/ -I$(DARKNET)/src/ -I$(DISTRIOT)/include/ -I$(DISTRIOT)/src/  -Iinclude/ -Isrc/ 
LDLIB=-L$(DARKNET) -l:libdarknet.a -L$(DISTRIOT) -l:libdistriot.a 


ifeq ($(OPENMP), 1) 
CFLAGS+= -fopenmp
endif

ifeq ($(DEBUG), 1) 
OPTS+=-O0 -g -std=c++11
else
OPTS= -std=c++11 -Ofast
endif
ifeq ($(NNPACK), 1)
COMMON+= -DNNPACK
CFLAGS+= -DNNPACK
LDFLAGS+= -lnnpack -lpthreadpool
endif

ifeq ($(ARM_NEON), 1)
COMMON+= -DARM_NEON
CFLAGS+= -DARM_NEON -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize
endif

CFLAGS+=$(OPTS)

#OBJ = job_queue.o data_blob.o
EXECOBJA = darknet_dist.o global_var.o

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
#OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard */*.h) Makefile

all: obj $(EXEC)


$(EXEC): $(EXECOBJ)
	$(CXX) $(COMMON) $(CFLAGS) $^ -o $@  $(LDLIB) $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@

test:
	./$(EXEC) ${ARGS}

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(EXEC) $(EXECOBJ) *.log *.png *.dat
