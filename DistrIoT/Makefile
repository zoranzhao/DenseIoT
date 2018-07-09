VPATH=./src:./examples
SLIB=libdistriot.so
ALIB=libdistriot.a
EXEC=test
OBJDIR=./obj/
CONFIGPATH=../include

CC=gcc
CXX=g++
AR=ar
ARFLAGS=rcs
OPTS= -std=c++11 -Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/ -I$(CONFIGPATH)
CFLAGS=-Wall -fPIC


ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

OBJ = job_queue.o data_blob.o distriot.o
EXECOBJA = test_thread.o

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard */*.h) Makefile

all: obj  $(SLIB) $(ALIB) 


$(EXEC): $(EXECOBJ) $(ALIB)
	$(CXX) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CXX) $(CFLAGS) -shared $^ -o $@

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@


obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) *.log
