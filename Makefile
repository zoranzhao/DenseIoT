NNPACK=0
ARM_NEON=0
VPATH=./src
EXEC=darknet_dist
OBJDIR=./obj/

CC=gcc
CXX=g++
AR=ar
ARFLAGS=rcs
OPTS= -std=c++11 -Ofast
LDFLAGS= -lm -pthread 
CFLAGS=-Wall -fPIC
DARKNET= ../darknet-nnpack
RIOT= ../riot
COMMON= -I$(DARKNET)/include/ -I$(DARKNET)/src/ -I$(RIOT)/include/ -I$(RIOT)/src/  -Iinclude/ -Isrc/ 
LDLIB=-L$(DARKNET) -l:libdarknet.a -L$(RIOT) -l:libriot.a 


ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
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
EXECOBJA = darknet_dist.o

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
#OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard */*.h) Makefile

all: obj $(EXEC)


$(EXEC): $(EXECOBJ)
	$(CXX) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIB)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@

test:
	./$(EXEC)

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(EXEC) $(EXECOBJ) *.log *.png
