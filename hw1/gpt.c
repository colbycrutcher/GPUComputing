#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* A simple struct to store one word and its count */
typedef struct {
    char word[64];  
    int count;
} Word;

/* Make a string lowercase (in place) */
void toLowerStr(char *s) {
    for (int i = 0; s[i] != '\0'; i++) {
        s[i] = (char)tolower((unsigned char)s[i]);
    }
}

/* Returns 1 if we should keep this word, 0 if we discard it */
int keepWord(const char *w) {
    int len = (int)strlen(w);
    if (len == 1) {
        return (w[0] == 'a' || w[0] == 'i');  // only keep a/i
    }
    return (len > 1);
}

/* Find word index in array, or -1 if not found */
int findWord(Word *words, int wordCount, const char *w) {
    for (int i = 0; i < wordCount; i++) {
        if (strcmp(words[i].word, w) == 0) {
            return i;
        }
    }
    return -1;
}

/* Add word or increment its count */
void addOrIncWord(Word *words, int *wordCount, int maxWords, const char *w) {
    int idx = findWord(words, *wordCount, w);
    if (idx >= 0) {
        words[idx].count++;
    } else {
        if (*wordCount >= maxWords) {
            printf("ERROR: Too many distinct words. Increase MAX_WORDS.\n");
            exit(1);
        }
        strcpy(words[*wordCount].word, w);
        words[*wordCount].count = 1;
        (*wordCount)++;
    }
}

/* Tokenize buffer by scanning characters (no strtok) */
void processBuffer(const char *buf, long n, Word *words, int *wordCount, int maxWords) {
    char current[64];
    int curLen = 0;

    for (long i = 0; i <= n; i++) {
        char c = (i < n) ? buf[i] : '\0';   // end acts like delimiter

        if (isalpha((unsigned char)c)) {
            if (curLen < 63) {             // keep space for '\0'
                current[curLen++] = c;
            }
        } else {
            if (curLen > 0) {
                current[curLen] = '\0';
                toLowerStr(current);

                if (keepWord(current)) {
                    addOrIncWord(words, wordCount, maxWords, current);
                }

                curLen = 0; // reset for next word
            }
        }
    }
}

/* Compare for alphabetical sort */
int cmpAlpha(const void *a, const void *b) {
    const Word *wa = (const Word *)a;
    const Word *wb = (const Word *)b;
    return strcmp(wa->word, wb->word);
}

/* Compare for count-descending sort (tie-break alphabetical) */
int cmpCountDesc(const void *a, const void *b) {
    const Word *wa = (const Word *)a;
    const Word *wb = (const Word *)b;

    if (wa->count != wb->count) {
        return (wb->count - wa->count); // bigger count first
    }
    return strcmp(wa->word, wb->word);
}

/* Write the table to a file (simple fixed-width formatting) */
void writeTable(const char *filename, Word *words, int wordCount) {
    FILE *out = fopen(filename, "w");
    if (!out) {
        printf("ERROR: could not open output file %s\n", filename);
        exit(1);
    }

    fprintf(out, "|--------------------|-------|\n");
    fprintf(out, "|English Word         | Count |\n");
    fprintf(out, "|--------------------|-------|\n");

    for (int i = 0; i < wordCount; i++) {
        fprintf(out, "|%-20s|%7d|\n", words[i].word, words[i].count);
        fprintf(out, "|--------------------|-------|\n");
    }

    fclose(out);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    const char *inputFile = argv[1];

    /* -------- read whole file first (I/O separate from processing) -------- */
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

    /* -------- Part 1 processing: tokenize + count + sort -------- */
    // NOTE: Pick a big max. testfile2 can have lots of distinct words.
    // If you hit the limit, increase it.
    const int MAX_WORDS = 200000;
    Word *words = (Word *)malloc(sizeof(Word) * MAX_WORDS);
    if (!words) {
        printf("ERROR: malloc words failed\n");
        free(buffer);
        return 1;
    }

    int wordCount = 0;

    processBuffer(buffer, size, words, &wordCount, MAX_WORDS);

    /* Make copies so we can sort two different ways */
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

    writeTable("sortedAlphabetical.txt", alpha, wordCount);
    writeTable("sortedFrequency.txt", occur, wordCount);

    printf("Done.\n");
    printf("Distinct words: %d\n", wordCount);
    printf("Wrote: sortedAlphabetical.txt and sortedFrequency.txt\n");

    free(alpha);
    free(occur);
    free(words);
    free(buffer);
    return 0;
}
