# Modified from the colby makefile guide.
IDIR =./include
SRC_DIR = ./src
# gcc compiler
# CC=/opt/homebrew/Cellar/gcc/11.2.0/bin/gcc-11
# clang compiler
CC = clang++
INCLUDE_DIR = -I$(IDIR)
CFLAGS= -std=c++17 -O3
LDFLAGS= `pkg-config --libs opencv4`
BUILD_DIR = ./target
ODIR=$(BUILD_DIR)/obj

LIBS=

BINARY_NAME = main

BINARY_PATH = $(BUILD_DIR)/binary
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(ODIR)/%.o, $(SRC))

default: build 

# poached from: https://stackoverflow.com/questions/1950926/create-directories-using-make-file
# @mkdir -p $(@D)

$(ODIR)/%.o: $(SRC_DIR)/%.cpp 
	@mkdir -p $(@D)
	@echo "$< -> $@"
	@$(CC) $(CFLAGS) -c -o $@ $< $(INCLUDE_DIR) `pkg-config pkg-config --cflags opencv4`

$(BINARY_PATH)/$(BINARY_NAME) : $(OBJ_FILES)
	@mkdir -p $(@D)
	@echo "Linking..."
	@$(CC) $(OBJ_FILES) -o $(BINARY_PATH)/$(BINARY_NAME) $(LDFLAGS)
	@echo "Compilation Succeeded."

build: $(BINARY_PATH)/$(BINARY_NAME)

init: $(ODIR) $(BINARY_PATH)

run: build 
	@$(BINARY_PATH)/$(BINARY_NAME)

.PHONY: clean build run

clean: 
	@echo "Removing target directory.."
	@rm -r $(BUILD_DIR)
	@echo "Cleaned!"