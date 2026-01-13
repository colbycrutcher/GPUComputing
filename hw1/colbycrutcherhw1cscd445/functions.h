#ifndef FUNCTIONS_H
#define FUNCTIONS_H

typedef struct {
    char word[64];
    int count;
} Word;

void processBuffer(const char *buf, long n, Word *words, int *wordCount, int maxWords);

int cmpAlpha(const void *a, const void *b);
int cmpCountDesc(const void *a, const void *b);

void writeTable(const char *filename, Word *words, int wordCount);

#endif
