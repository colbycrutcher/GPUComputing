#include "functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// makes string lowercase (in place)
static void toLowerStr(char *s) {
    for (int i = 0; s[i] != '\0'; i++) {
        s[i] = (char)tolower((unsigned char)s[i]);
    }
}

// Returns 1 if we should keep this word, 0 if we discard it
static int keepWord(const char *w) {
    int len = (int)strlen(w);
    if (len == 1) {
        return (w[0] == 'a' || w[0] == 'i');  // only keep a/i
    }
    return (len > 1);
}

// Find word index in array, or -1 if not found 
static int findWord(Word *words, int wordCount, const char *w) {
    for (int i = 0; i < wordCount; i++) {
        if (strcmp(words[i].word, w) == 0) {
            return i;
        }
    }
    return -1;
}

// Add word or increment its count 
static void addOrIncWord(Word *words, int *wordCount, int maxWords, const char *w) {
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

// Tokenize buffer by scanning characters (no strtok) 
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

// Compare for alphabetical sort 
int cmpAlpha(const void *a, const void *b) {
    const Word *wa = (const Word *)a;
    const Word *wb = (const Word *)b;
    return strcmp(wa->word, wb->word);
}

// Compares for count-descending sort (tie-breaker alphabetical)
int cmpCountDesc(const void *a, const void *b) {
    const Word *wa = (const Word *)a;
    const Word *wb = (const Word *)b;

    if (wa->count != wb->count) {
        return (wb->count - wa->count); // bigger count first
    }
    return strcmp(wa->word, wb->word);
}

// Writes the table to a file
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
