# # Address-sanitizer library
# ASAN_FLAGS := -lasan
# ifneq ($(OS),Windows_NT)
# 	UNAME_S := $(shell uname -s)
# 	ifeq ($(UNAME_S),Darwin)
# 		# macOS requires different a flag
# 		ASAN_FLAGS := -fsanitize=address
# 	endif
# endif

# gopro_data_extractor : gopro_data_extraction.o GPMF_parser.o GPMF_utils.o GPMF_mp4reader.o GPMF_print.o
# 		gcc -o gopro_data_extractor gopro_data_extraction.o GPMF_parser.o GPMF_utils.o GPMF_mp4reader.o GPMF_print.o $(ASAN_FLAGS)

# gopro_data_extraction.o : gopro_data_extraction.c
# 		gcc -g -c gopro_data_extraction.c
# GPMF_mp4reader.o : gpmf-parser/demo/GPMF_mp4reader.c gpmf-parser/GPMF_parser.h
# 		gcc -g -c gpmf-parser/demo/GPMF_mp4reader.c
# GPMF_print.o : gpmf-parser/demo/GPMF_print.c gpmf-parser/GPMF_parser.h
# 		gcc -g -c gpmf-parser/demo/GPMF_print.c
# GPMF_parser.o : gpmf-parser/GPMF_parser.c gpmf-parser/GPMF_parser.h
# 		gcc -g -c gpmf-parser/GPMF_parser.c
# GPMF_utils.o : gpmf-parser/GPMF_utils.c gpmf-parser/GPMF_utils.h
# 		gcc -g -c gpmf-parser/GPMF_utils.c
# clean :
# 		rm gopro_data_extractor *.o


CC = gcc
CFLAGS = -fPIC -g

# Address-sanitizer library
ASAN_FLAGS := -lasan
ifneq ($(OS),Windows_NT)
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        # macOS requires different a flag
        ASAN_FLAGS := -fsanitize=address
    endif
endif

# Target executable
TARGET = gopro_data_extractor

# Source files
SRCS = gopro_data_extraction.c \
       gpmf-parser/demo/GPMF_mp4reader.c \
       gpmf-parser/demo/GPMF_print.c \
       gpmf-parser/GPMF_parser.c \
       gpmf-parser/GPMF_utils.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default rule to compile all object files and build the executable
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(ASAN_FLAGS)

# Rule to compile for ACCL mode
accl: CFLAGS += -DACCL_MODE
accl: clean $(TARGET)

# Rule to compile for GYRO mode
gyro: CFLAGS += -DGYRO_MODE
gyro: clean $(TARGET)

# Rules to compile object files
%.o: %.c
	$(CC) -g -c $(CFLAGS) $< -o $@

# Clean rule
clean:
	rm -f $(TARGET) *.o