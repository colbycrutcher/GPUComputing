#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_WORDS 1000000
#define MAX_WORD_LEN 100

typedef struct {
    char word[MAX_WORD_LEN];
    int count;
} WordEntry;

WordEntry wordList[MAX_WORDS];
int wordCount = 0;

/* Find word in list */
int findWord(const char *w) {
    for (int i = 0; i < wordCount; i++) {
        if (strcmp(wordList[i].word, w) == 0) {
            return i;
        }
    }
    return -1;
}

void addWord(const char *w) {
    int idx = findWord(w);
    if (idx >= 0) {
        wordList[idx].count++;
        return;
    }

    if (wordCount >= MAX_WORDS) {
        fprintf(stderr, "Too many unique words (MAX_WORDS=%d). Increase MAX_WORDS.\n", MAX_WORDS);
        exit(1);
    }

    strncpy(wordList[wordCount].word, w, MAX_WORD_LEN - 1);
    wordList[wordCount].word[MAX_WORD_LEN - 1] = '\0';
    wordList[wordCount].count = 1;
    wordCount++;
}

void extractOneLine(char *line) {
    char word[MAX_WORD_LEN];
    int i = 0, j = 0;

    while (line[i] != '\0') {
        if (isalpha((unsigned char)line[i])) {
            if (j < MAX_WORD_LEN - 1) {
                word[j++] = (char)tolower((unsigned char)line[i]);
            } else {
                /* Word is too long — keep consuming letters but don't overflow buffer */
                // do nothing, just skip extra letters
            }
        } else {
            if (j > 0) {
                word[j] = '\0';
                addWord(word);
                j = 0;
            }
        }
        i++;
    }

    if (j > 0) {
        word[j] = '\0';
        addWord(word);
    }
}


/* Sort alphabetically */
int compareAlpha(const void *a, const void *b) {
    return strcmp(((WordEntry*)a)->word, ((WordEntry*)b)->word);
}

/* Sort by occurrence (descending) */
int compareOccur(const void *a, const void *b) {
    return ((WordEntry*)b)->count - ((WordEntry*)a)->count;
}

int main() {
    FILE *file = fopen("testfile2", "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    char line[1024];

    /* Read file line-by-line */
    while (fgets(line, sizeof(line), file)) {
        extractOneLine(line);
    }

    fclose(file);

    /* Alphabetical Output */
    qsort(wordList, wordCount, sizeof(WordEntry), compareAlpha);

    printf("\n===== Words Sorted Alphabetically =====\n");
    printf("Word\tCount\n");
    printf("----------------\n");
    for (int i = 0; i < wordCount; i++) {
        printf("%s\t%d\n", wordList[i].word, wordList[i].count);
    }

    /* Occurrence Output */
    qsort(wordList, wordCount, sizeof(WordEntry), compareOccur);

    printf("\n===== Words Sorted by Occurrence =====\n");
    printf("Word\tCount\n");
    printf("----------------\n");
    for (int i = 0; i < wordCount; i++) {
        printf("%s\t%d\n", wordList[i].word, wordList[i].count);
    }

    return 0;
}
