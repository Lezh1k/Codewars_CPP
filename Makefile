CC=g++
#CC=clang++
AS=as
LD=ld
LINK=$(CC)

BUILD_DIR=build
OBJ_DIR=$(BUILD_DIR)/obj
BIN_DIR=bin

LIBS := -lm -lpthread
DEFS := -D_USE_MATH_DEFINES
WARN_LEVEL = -Wall -Wextra -pedantic

PRG = codewars_cpp
INCLUDES = -Iinc
CXXFLAGS := $(INCLUDES) $(DEFS) $(WARN_LEVEL) -pipe -O0 -g -std=gnu++2a
debug: CXXFLAGS += -O0 -g3
debug: all
release: CXXFLAGS += -O2
release: all

LDFLAGS = $(LIBS)

SRC_C := $(wildcard *.cpp) $(wildcard src/*.cpp)

OBJECTS := $(SRC_C:%.cpp=$(OBJ_DIR)/%.o)
all: directories $(PRG)

$(PRG): $(BIN_DIR)/$(PRG)
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CC) $(CXXFLAGS) -o $@ -c $<

$(BIN_DIR)/$(PRG): $(OBJECTS)
	@mkdir -p $(@D)
	$(LINK) -o $(BIN_DIR)/$(PRG).elf $^ $(LDFLAGS) $(LIBS)

.PHONY: directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)/*
	@rm -rf $(BIN_DIR)/*

.PHONY: mrproper
mrproper:
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BIN_DIR)
