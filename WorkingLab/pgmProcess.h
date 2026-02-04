#ifndef PGM_PROCESS_H
#define PGM_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

void launchDrawCircle(int *d_pixels, int numRows, int numCols,
                      int centerRow, int centerCol, int radius);

void launchDrawEdge(int *d_pixels, int numRows, int numCols, int edgeWidth);

void launchDrawLine(int *d_pixels, int numRows, int numCols,
                    int p1row, int p1col, int p2row, int p2col);

#ifdef __cplusplus
}
#endif

#endif
