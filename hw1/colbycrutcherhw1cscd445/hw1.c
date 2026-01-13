#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "functions.h"

int main() {

    const char *inputFile = "testfile2";   // Always use testfile2

    // read whole file first
    FILE *fp = fopen(inputFile, "rb");
    if (!fp) {
        printf("ERROR: could not open input file %s\n", inputFile);
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    char *buffer = (char *)malloc(size);
    if (!buffer) {
        printf("ERROR: malloc failed\n");
        fclose(fp);
        return 1;
    }

    long bytesRead = (long)fread(buffer, 1, size, fp);
    fclose(fp);

    if (bytesRead != size) {
        printf("ERROR: file read mismatch\n");
        free(buffer);
        return 1;
    }

    // processing: tokenize + count + sort 
    const int MAX_WORDS = 200000;

    Word *words = (Word *)malloc(sizeof(Word) * MAX_WORDS);
    if (!words) {
        printf("ERROR: malloc words failed\n");
        free(buffer);
        return 1;
    }

    int wordCount = 0;
    processBuffer(buffer, size, words, &wordCount, MAX_WORDS);

    // Make copies so we can sort two different ways 
    Word *alpha = (Word *)malloc(sizeof(Word) * wordCount);
    Word *occur = (Word *)malloc(sizeof(Word) * wordCount);
    if (!alpha || !occur) {
        printf("ERROR: malloc copy failed\n");
        free(words);
        free(buffer);
        free(alpha);
        free(occur);
        return 1;
    }

    memcpy(alpha, words, sizeof(Word) * wordCount);
    memcpy(occur, words, sizeof(Word) * wordCount);

    qsort(alpha, wordCount, sizeof(Word), cmpAlpha);
    qsort(occur, wordCount, sizeof(Word), cmpCountDesc);

    writeTable("sortedWord.txt", alpha, wordCount);
    writeTable("sortedOccur.txt", occur, wordCount);

    printf("Done.\n");
    printf("Distinct words: %d\n", wordCount);
    printf("Wrote: sortedWord.txt and sortedOccur.txt\n");

    free(alpha);
    free(occur);
    free(words);
    free(buffer);
    return 0;
}
