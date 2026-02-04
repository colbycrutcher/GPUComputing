#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pgmUtility.h"

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s -c  centerRow centerCol radius        input.pgm output.pgm\n"
        "  %s -e  edgeWidth                        input.pgm output.pgm\n"
        "  %s -l  p1row p1col p2row p2col          input.pgm output.pgm\n"
        "  %s -ce centerRow centerCol radius edgeWidth input.pgm output.pgm\n"
        "  %s -c -e centerRow centerCol radius edgeWidth input.pgm output.pgm\n",
        prog, prog, prog, prog, prog);
    exit(1);
}

static void freeHeader(char **header)
{
    for (int i = 0; i < rowsInHeader; i++) {
        free(header[i]);
        header[i] = NULL;
    }
}

int main(int argc, char **argv)
{
    char *header[rowsInHeader] = {0};
    int numRows = 0, numCols = 0;
    int *pixels = NULL;

    FILE *in = NULL;
    FILE *out = NULL;

    if (argc < 2) usage(argv[0]);

    // Special case: "-c -e ..." (two flags)
    if (strcmp(argv[1], "-c") == 0 && argc >= 3 && strcmp(argv[2], "-e") == 0) {
        if (argc != 9) usage(argv[0]);

        int centerRow = atoi(argv[3]);
        int centerCol = atoi(argv[4]);
        int radius    = atoi(argv[5]);
        int edgeWidth = atoi(argv[6]);
        const char *inName  = argv[7];
        const char *outName = argv[8];

        in = fopen(inName, "r");
        out = fopen(outName, "w");
        if (!in || !out) {
            fprintf(stderr, "Error opening input/output file.\n");
            goto cleanup;
        }

        pixels = pgmRead(header, &numRows, &numCols, in);
        if (!pixels) {
            fprintf(stderr, "Error reading PGM.\n");
            goto cleanup;
        }

        pgmDrawCircle(pixels, numRows, numCols, centerRow, centerCol, radius, header);
        pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);

        pgmWrite((const char**)header, pixels, numRows, numCols, out);
        goto cleanup;
    }

    // Normal case: one flag in argv[1]
    if (argv[1][0] != '-') usage(argv[0]);

    switch (argv[1][1])
    {
        case 'c':   // -c centerRow centerCol radius in out
        {
            if (argc != 7) usage(argv[0]);

            int centerRow = atoi(argv[2]);
            int centerCol = atoi(argv[3]);
            int radius    = atoi(argv[4]);
            const char *inName  = argv[5];
            const char *outName = argv[6];

            in = fopen(inName, "r");
            out = fopen(outName, "w");
            if (!in || !out) {
                fprintf(stderr, "Error opening input/output file.\n");
                goto cleanup;
            }

            pixels = pgmRead(header, &numRows, &numCols, in);
            if (!pixels) {
                fprintf(stderr, "Error reading PGM.\n");
                goto cleanup;
            }

            pgmDrawCircle(pixels, numRows, numCols, centerRow, centerCol, radius, header);
            pgmWrite((const char**)header, pixels, numRows, numCols, out);
            break;
        }

        case 'e':   // -e edgeWidth in out
        {
            if (argc != 5) usage(argv[0]);

            int edgeWidth = atoi(argv[2]);
            const char *inName  = argv[3];
            const char *outName = argv[4];

            in = fopen(inName, "r");
            out = fopen(outName, "w");
            if (!in || !out) {
                fprintf(stderr, "Error opening input/output file.\n");
                goto cleanup;
            }

            pixels = pgmRead(header, &numRows, &numCols, in);
            if (!pixels) {
                fprintf(stderr, "Error reading PGM.\n");
                goto cleanup;
            }

            pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);
            pgmWrite((const char**)header, pixels, numRows, numCols, out);
            break;
        }

        case 'l':   // -l p1row p1col p2row p2col in out
        {
            if (argc != 8) usage(argv[0]);

            int p1row = atoi(argv[2]);
            int p1col = atoi(argv[3]);
            int p2row = atoi(argv[4]);
            int p2col = atoi(argv[5]);
            const char *inName  = argv[6];
            const char *outName = argv[7];

            in = fopen(inName, "r");
            out = fopen(outName, "w");
            if (!in || !out) {
                fprintf(stderr, "Error opening input/output file.\n");
                goto cleanup;
            }

            pixels = pgmRead(header, &numRows, &numCols, in);
            if (!pixels) {
                fprintf(stderr, "Error reading PGM.\n");
                goto cleanup;
            }

            pgmDrawLine(pixels, numRows, numCols, header, p1row, p1col, p2row, p2col);
            pgmWrite((const char**)header, pixels, numRows, numCols, out);
            break;
        }

        default:
            // Bonus: "-ce ..." is not handled by argv[1][1] because argv[1][1]=='c'
            // so we handle it here by checking the full flag string.
            if (strcmp(argv[1], "-ce") == 0) {
                if (argc != 8) usage(argv[0]);

                int centerRow = atoi(argv[2]);
                int centerCol = atoi(argv[3]);
                int radius    = atoi(argv[4]);
                int edgeWidth = atoi(argv[5]);
                const char *inName  = argv[6];
                const char *outName = argv[7];

                in = fopen(inName, "r");
                out = fopen(outName, "w");
                if (!in || !out) {
                    fprintf(stderr, "Error opening input/output file.\n");
                    goto cleanup;
                }

                pixels = pgmRead(header, &numRows, &numCols, in);
                if (!pixels) {
                    fprintf(stderr, "Error reading PGM.\n");
                    goto cleanup;
                }

                pgmDrawCircle(pixels, numRows, numCols, centerRow, centerCol, radius, header);
                pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);
                pgmWrite((const char**)header, pixels, numRows, numCols, out);
                break;
            }

            usage(argv[0]);
    }

cleanup:
    if (in) fclose(in);
    if (out) fclose(out);
    free(pixels);
    freeHeader(header);
    return 0;
}
