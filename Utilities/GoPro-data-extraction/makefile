# Address-sanitizer library
ASAN_FLAGS := -lasan
ifneq ($(OS),Windows_NT)
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		# macOS requires different a flag
		ASAN_FLAGS := -fsanitize=address
	endif
endif

gopro_data_extractor : gopro_data_extraction.o GPMF_parser.o GPMF_utils.o GPMF_mp4reader.o GPMF_print.o
		gcc -o gopro_data_extractor gopro_data_extraction.o GPMF_parser.o GPMF_utils.o GPMF_mp4reader.o GPMF_print.o $(ASAN_FLAGS)

gopro_data_extraction.o : gopro_data_extraction.c
		gcc -g -c gopro_data_extraction.c
GPMF_mp4reader.o : gpmf-parser/demo/GPMF_mp4reader.c gpmf-parser/GPMF_parser.h
		gcc -g -c gpmf-parser/demo/GPMF_mp4reader.c
GPMF_print.o : gpmf-parser/demo/GPMF_print.c gpmf-parser/GPMF_parser.h
		gcc -g -c gpmf-parser/demo/GPMF_print.c
GPMF_parser.o : gpmf-parser/GPMF_parser.c gpmf-parser/GPMF_parser.h
		gcc -g -c gpmf-parser/GPMF_parser.c
GPMF_utils.o : gpmf-parser/GPMF_utils.c gpmf-parser/GPMF_utils.h
		gcc -g -c gpmf-parser/GPMF_utils.c
clean :
		rm gopro_data_extractor *.o