#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "gpmf-parser/GPMF_parser.h"
#include "gpmf-parser/demo/GPMF_mp4reader.h"
#include "gpmf-parser/GPMF_utils.h"

#define SHOW_GPMF_STRUCTURE 0
#ifdef ACCL_MODE
#define SHOW_THIS_FOUR_CC "ACCL"
#elif GYRO_MODE
#define SHOW_THIS_FOUR_CC "GYRO"
#else
#define SHOW_THIS_FOUR_CC "GYRO"
#endif

extern void PrintGPMF(GPMF_stream *ms);
GPMF_ERR readMP4File(char *filename, char *output_folder);

uint32_t show_gpmf_structure = SHOW_GPMF_STRUCTURE;
uint32_t show_this_four_cc = STR2FOURCC(SHOW_THIS_FOUR_CC);

FILE *csvFileAccel;
FILE *csvFileGyro;

int main(int argc, char *argv[])
{
	GPMF_ERR ret = GPMF_OK;

	// Input: file_path  output_folder
	if (argc > 1)
	{ // Running through terminal
		ret = readMP4File(argv[1],
						  argv[2]);
	}
	else
	{
		ret = readMP4File("/RUD-PT/rudpt_ws/src/UnderwaterMapping/Utilities/GoPro-to-bag/videos/left/1,1_0_0_10_left.MP4",
						  "/RUD-PT/rudpt_ws/src/UnderwaterMapping/Utilities/GoPro-data-extraction/Output/1,1_0_0_10/Metadata");
	}

	if (ret == 0)
	{
		printf("File created successfully.\n");
	}
	else
	{
		printf("Error creating file.\n");
	}

	printf("\n");
	return 0;
}

GPMF_ERR readMP4File(char *filename, char *output_folder)
{
	GPMF_ERR ret = GPMF_OK;
	GPMF_stream metadata_stream = {0}, *ms = &metadata_stream;
	double metadatalength;
	uint32_t *payload = NULL;
	uint32_t payloadsize = 0;
	size_t payloadres = 0;

	// Search for GPMF Track
	size_t mp4handle = OpenMP4Source(filename, MOV_GPMF_TRAK_TYPE, MOV_GPMF_TRAK_SUBTYPE, 0);
	if (mp4handle == 0)
	{
		printf("error: %s is an invalid MP4/MOV or it has no GPMF data\n\n", filename);
		return GPMF_ERROR_BAD_STRUCTURE;
	}

	metadatalength = GetDuration(mp4handle);
	if (metadatalength > 0.0)
	{
		char *output_path;
		output_path = malloc(128); // Make space for the new string (should check the return value ...)
		if (output_path == NULL)
		{ // Memory allocation failed; handle the error
			fprintf(stderr, "Memory allocation failed\n");
			exit(1); // Exit the program or handle the error appropriately
		}
		strcpy(output_path, output_folder); // Copy name into the new var
		// If output folder is specified add "/" in between
		if ((output_path != NULL) && (output_path[0] != '\0'))
			strcat(output_path, "/");

		// Open CSV files for data saving
		if (SHOW_THIS_FOUR_CC == "ACCL")
		{
			strcat(output_path, "outputAccl.csv"); // Add the extension
			csvFileAccel = fopen(output_path, "w");
			if (csvFileAccel == NULL)
			{
				printf("error: could not open outputAccl.csv");
				return 1; // Exit the program with an error code
			}
		}
		if (SHOW_THIS_FOUR_CC == "GYRO")
		{
			strcat(output_path, "outputGyro.csv"); // Add the extension
			csvFileGyro = fopen(output_path, "w");
			if (csvFileGyro == NULL)
			{
				printf("error: could not open outputGyro.csv");
				return 1; // Exit the program with an error code
			}
		}
		free(output_path);

		// Print out video framerate
		uint32_t fr_num, fr_dem;
		uint32_t frames = GetVideoFrameRateAndCount(mp4handle, &fr_num, &fr_dem);
		if (frames)
		{
			printf("VIDEO FRAMERATE:\n  %.3f with %d frames\n", (float)fr_num / (float)fr_dem, frames);
		}

		// Run througn payloads
		uint32_t index, payloads = GetNumberPayloads(mp4handle);
		for (index = 0; index < payloads; index++)
		{
			double in = 0.0, out = 0.0; // times
			payloadsize = GetPayloadSize(mp4handle, index);
			payloadres = GetPayloadResource(mp4handle, payloadres, payloadsize);
			payload = GetPayload(mp4handle, payloadres, index);
			if (payload == NULL)
				goto cleanup;

			ret = GetPayloadTime(mp4handle, index, &in, &out);
			if (ret != GPMF_OK)
				goto cleanup;

			ret = GPMF_Init(ms, payload, payloadsize);
			if (ret != GPMF_OK)
				goto cleanup;

			if (show_gpmf_structure)
			{
				printf("GPMF STRUCTURE:\n");
				// Output (printf) all the contained GPMF data within this payload
				ret = GPMF_Validate(ms, GPMF_RECURSE_LEVELS); // optional
				if (GPMF_OK != ret)
				{
					if (GPMF_ERROR_UNKNOWN_TYPE == ret)
					{
						printf("Unknown GPMF Type within, ignoring\n");
						ret = GPMF_OK;
					}
					else
					{
						printf("Invalid GPMF Structure\n");
					}
				}

				GPMF_ResetState(ms);

				GPMF_ERR nextret;
				do
				{
					printf("  ");
					PrintGPMF(ms); // printf current GPMF KLV

					nextret = GPMF_Next(ms, GPMF_RECURSE_LEVELS | GPMF_TOLERANT);

					while (nextret == GPMF_ERROR_UNKNOWN_TYPE) // or just using GPMF_Next(ms, GPMF_RECURSE_LEVELS|GPMF_TOLERANT) to ignore and skip unknown types
						nextret = GPMF_Next(ms, GPMF_RECURSE_LEVELS);

				} while (GPMF_OK == nextret);
				GPMF_ResetState(ms);
			}

			// TODO: Insert loop to iterate through the desired fourcc
			while (GPMF_OK == GPMF_FindNext(ms, STR2FOURCC("STRM"), GPMF_RECURSE_LEVELS | GPMF_TOLERANT))
			{ // GoPro Hero5/6/7 Accelerometer)
				if (GPMF_VALID_FOURCC(show_this_four_cc))
				{
					if (GPMF_OK != GPMF_FindNext(ms, show_this_four_cc, GPMF_RECURSE_LEVELS | GPMF_TOLERANT))
						continue;
				}
				else
				{
					ret = GPMF_SeekToSamples(ms);
					if (GPMF_OK != ret)
						continue;
				}

				char *rawdata = (char *)GPMF_RawData(ms);
				uint32_t key = GPMF_Key(ms);
				GPMF_SampleType type = GPMF_Type(ms);
				uint32_t samples = GPMF_Repeat(ms);
				uint32_t elements = GPMF_ElementsInStruct(ms);

				if (samples)
				{
					uint32_t buffersize = samples * elements * sizeof(double);
					GPMF_stream find_stream;
					double *ptr, *tmpbuffer = (double *)malloc(buffersize);

#define MAX_UNITS 64
#define MAX_UNITLEN 8
					char units[MAX_UNITS][MAX_UNITLEN] = {""};
					uint32_t unit_samples = 1;

					char complextype[MAX_UNITS] = {""};
					uint32_t type_samples = 1;

					if (tmpbuffer)
					{
						uint32_t i, j;

						// Search for any units to display
						GPMF_CopyState(ms, &find_stream);
						if (GPMF_OK == GPMF_FindPrev(&find_stream, GPMF_KEY_SI_UNITS, GPMF_CURRENT_LEVEL | GPMF_TOLERANT) ||
							GPMF_OK == GPMF_FindPrev(&find_stream, GPMF_KEY_UNITS, GPMF_CURRENT_LEVEL | GPMF_TOLERANT))
						{
							char *data = (char *)GPMF_RawData(&find_stream);
							uint32_t ssize = GPMF_StructSize(&find_stream);
							if (ssize > MAX_UNITLEN - 1)
								ssize = MAX_UNITLEN - 1;
							unit_samples = GPMF_Repeat(&find_stream);

							for (i = 0; i < unit_samples && i < MAX_UNITS; i++)
							{
								memcpy(units[i], data, ssize);
								units[i][ssize] = 0;
								data += ssize;
							}
						}

						// Search for TYPE if Complex
						GPMF_CopyState(ms, &find_stream);
						type_samples = 0;
						if (GPMF_OK == GPMF_FindPrev(&find_stream, GPMF_KEY_TYPE, GPMF_CURRENT_LEVEL | GPMF_TOLERANT))
						{
							char *data = (char *)GPMF_RawData(&find_stream);
							uint32_t ssize = GPMF_StructSize(&find_stream);
							if (ssize > MAX_UNITLEN - 1)
								ssize = MAX_UNITLEN - 1;
							type_samples = GPMF_Repeat(&find_stream);

							for (i = 0; i < type_samples && i < MAX_UNITS; i++)
							{
								complextype[i] = data[i];
							}
						}

						// GPMF_FormattedData(ms, tmpbuffer, buffersize, 0, samples); // Output data in LittleEnd, but no scale
						if (GPMF_OK == GPMF_ScaledData(ms, tmpbuffer, buffersize, 0, samples, GPMF_TYPE_DOUBLE)) // Output scaled data as floats
						{

							ptr = tmpbuffer;
							int pos = 0;
							for (i = 0; i < samples; i++)
							{
								// THIS IS WHAT PRINTS ACCEL NAME
								// printf("  %c%c%c%c ", PRINTF_4CC(key));

								for (j = 0; j < elements; j++)
								{
									if (SHOW_THIS_FOUR_CC == "ACCL")
									{
										fprintf(csvFileAccel, "%.10f,", *ptr++);
									}
									if (SHOW_THIS_FOUR_CC == "GYRO")
									{
										fprintf(csvFileGyro, "%.10f,", *ptr++);
									}
								}

								// THIS IS WHAT PRINTS THE NEWLINE
								if (SHOW_THIS_FOUR_CC == "ACCL")
								{
									fprintf(csvFileAccel, "\n");
								}
								if (SHOW_THIS_FOUR_CC == "GYRO")
								{
									fprintf(csvFileGyro, "\n");
								}
							}
						}
						free(tmpbuffer);
					}
				}
			}
			GPMF_ResetState(ms);
		}

		mp4callbacks cbobject;
		cbobject.mp4handle = mp4handle;
		cbobject.cbGetNumberPayloads = GetNumberPayloads;
		cbobject.cbGetPayload = GetPayload;
		cbobject.cbGetPayloadSize = GetPayloadSize;
		cbobject.cbGetPayloadResource = GetPayloadResource;
		cbobject.cbGetPayloadTime = GetPayloadTime;
		cbobject.cbFreePayloadResource = FreePayloadResource;
		cbobject.cbGetEditListOffsetRationalTime = GetEditListOffsetRationalTime;

		printf("COMPUTED SAMPLERATES:\n");
		// Find all the available Streams and compute they sample rates
		while (GPMF_OK == GPMF_FindNext(ms, GPMF_KEY_STREAM, GPMF_RECURSE_LEVELS | GPMF_TOLERANT))
		{
			if (GPMF_OK == GPMF_SeekToSamples(ms))
			{ // find the last FOURCC within the stream
				double start, end;
				uint32_t fourcc = GPMF_Key(ms);

				double rate = GetGPMFSampleRate(cbobject, fourcc, STR2FOURCC("SHUT"), GPMF_SAMPLE_RATE_PRECISE, &start, &end); // GPMF_SAMPLE_RATE_FAST);
				printf("  %c%c%c%c sampling rate = %fHz (time %f to %f)\",\n", PRINTF_4CC(fourcc), rate, start, end);
			}
		}

		// Close CVS file after data saving
		if (SHOW_THIS_FOUR_CC == "ACCL")
		{
			fclose(csvFileAccel);
		}
		if (SHOW_THIS_FOUR_CC == "GYRO")
		{
			fclose(csvFileGyro);
		}

	cleanup:
		if (payloadres)
			FreePayloadResource(mp4handle, payloadres);
		if (ms)
			GPMF_Free(ms);
		CloseSource(mp4handle);
	}
	return ret;
}

/*
if (fourCCs[i] == "ACCL") {
	//printf("  %c%c%c%c ", PRINTF_4CC(key));
	for (j = 0; j < elements; j++) {
		fprintf(csvFileAccel, "%.10f,", *ptr++);
	}
	fprintf(csvFileAccel, "\n");
} else if (fourCCs[i] == "GYRO") {
	//printf("  %c%c%c%c ", PRINTF_4CC(key));
	for (j = 0; j < elements; j++) {
		fprintf(csvFileGyro, "%.10f,", *ptr++);
	}
	fprintf(csvFileGyro, "\n");
}
*/
