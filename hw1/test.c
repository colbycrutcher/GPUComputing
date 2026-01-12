#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>


typedef struct WordItem {
    char *word;              // stored in lowercase for easy case-insensitive handling
    int count;
    struct WordItem *next;
} WordItem; 

/* ---------- Linked List Helpers ---------- */

static WordItem *create_word_item(const char *word_lower) {
    WordItem *node = (WordItem *)malloc(sizeof(WordItem));
    if (!node) { perror("malloc"); exit(1); }
    node->word = strdup(word_lower);
    if (!node->word) { perror("strdup"); exit(1); }
    node->count = 1;
    node->next = NULL;
    return node;
}

static void free_word_list(WordItem *head) {
    while (head) {
        WordItem *tmp = head->next;
        free(head->word);
        free(head);
        head = tmp;
    }
}

/* Find word in list (case-insensitive). We store lowercase, so strcmp is enough here. */
static WordItem *find_word(WordItem *head, const char *word_lower) {
    for (WordItem *cur = head; cur; cur = cur->next) {
        if (strcmp(cur->word, word_lower) == 0) return cur;
    }
    return NULL;
}

/* Insert at head (simple) */
static void insert_word(WordItem **head, const char *word_lower) {
    WordItem *node = create_word_item(word_lower);
    node->next = *head;
    *head = node;
}

/* Update occurrence or insert if not found */
static void upsert_word(WordItem **head, const char *word_lower, int add_count) {
    WordItem *found = find_word(*head, word_lower);
    if (found) {
        found->count += add_count;
    } else {
        WordItem *node = create_word_item(word_lower);
        node->count = add_count;
        node->next = *head;
        *head = node;
    }
}

/* ---------- “substring method” from pseudocode ---------- */
/* Copies aline[start..start+len-1] into a new malloc'd string, lowercased. */
static char *subStringLower(const char *aline, int start, int len) {
    char *s = (char *)malloc((size_t)len + 1);
    if (!s) { perror("malloc"); exit(1); }
    for (int k = 0; k < len; k++) {
        s[k] = (char)tolower((unsigned char)aline[start + k]);
    }
    s[len] = '\0';
    return s;
}

/* Single-letter discard rule: keep only "a" or "i" */
static bool keep_word(const char *w) {
    size_t n = strlen(w);
    if (n == 1) return (w[0] == 'a' || w[0] == 'i');
    return (n > 1);
}

/* ---------- PSEUDOCODE FUNCTION (close match) ---------- */
WordItem *extractOneLine(char *aline) {
    bool inWord = false;
    WordItem *wordList = NULL;   // "Create a new empty linkedList wordList"
    int i = 0;                   // start index
    int start = 0;               // start position of current word
    int wordLen = 0;             // length of current word

    while (aline[i] != '\n' && aline[i] != '\0') {
        if (isalpha((unsigned char)aline[i])) {
            if (inWord == false) {
                start = i;       // save start
                inWord = true;
            }
            wordLen++;
        } else if (inWord == true) {
            /*
              We have found an English word starting at index start, with length wordLen.
              Use substring method to copy it into newWord.
            */
            char *newWord = subStringLower(aline, start, wordLen);

            if (keep_word(newWord)) {
                /*
                  if newWord not exist: create WordItem and insert
                  else update occurrence
                */
                WordItem *found = find_word(wordList, newWord);
                if (!found) {
                    insert_word(&wordList, newWord);
                } else {
                    found->count++;
                }
            }

            free(newWord);

            /* Reset wordLen and inWord */
            wordLen = 0;
            inWord = false;
        }

        i++; // increase i by one for next character
    }

    /* NOTE: if line ends with a letter (no delimiter before \n/\0), pseudocode doesn't finalize.
       But real input might have that. We'll safely finalize to avoid missing the last word. */
    if (inWord == true && wordLen > 0) {
        char *newWord = subStringLower(aline, start, wordLen);
        if (keep_word(newWord)) {
            WordItem *found = find_word(wordList, newWord);
            if (!found) insert_word(&wordList, newWord);
            else found->count++;
        }
        free(newWord);
    }

    return wordList; // Return wordList
}

/* ---------- Merge per-line list into global list ---------- */
static void merge_into_global(WordItem **global, WordItem *lineList) {
    for (WordItem *cur = lineList; cur; cur = cur->next) {
        upsert_word(global, cur->word, cur->count);
    }
}

/* ---------- Sorting Support (linked list -> array) ---------- */
typedef struct {
    char *word;
    int count;
} Entry;

static int cmp_alpha(const void *a, const void *b) {
    const Entry *ea = (const Entry *)a;
    const Entry *eb = (const Entry *)b;
    return strcmp(ea->word, eb->word);
}

static int cmp_count_desc(const void *a, const void *b) {
    const Entry *ea = (const Entry *)a;
    const Entry *eb = (const Entry *)b;
    if (eb->count != ea->count) return eb->count - ea->count;
    return strcmp(ea->word, eb->word); // tie-breaker
}

static Entry *list_to_array(WordItem *head, int *out_n) {
    int n = 0;
    for (WordItem *cur = head; cur; cur = cur->next) n++;

    Entry *arr = (Entry *)malloc((size_t)n * sizeof(Entry));
    if (!arr) { perror("malloc"); exit(1); }

    int i = 0;
    for (WordItem *cur = head; cur; cur = cur->next) {
        arr[i].word = cur->word;   // points into list storage (don’t free separately)
        arr[i].count = cur->count;
        i++;
    }

    *out_n = n;
    return arr;
}

/* ---------- Output formatting (stdout) ---------- */
static void print_table(const Entry *arr, int n) {
    printf("|------------------|--------|\n");
    printf("| English Word     | Count  |\n");
    printf("|------------------|--------|\n");
    for (int i = 0; i < n; i++) {
        printf("| %-16s | %-6d |\n", arr[i].word, arr[i].count);
        printf("|------------------|--------|\n");
    }
}

/* ---------- Main ---------- */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) { perror("fopen"); return 1; }

    /* Read file line-by-line, call extractOneLine for each line, merge into global list */
    WordItem *global = NULL;

    /* Big enough for test files; can be increased if needed */
    size_t cap = 4096;
    char *line = (char *)malloc(cap);
    if (!line) { perror("malloc"); return 1; }

    while (fgets(line, (int)cap, fp)) {
        /* Ensure line ends with '\n' or '\0' as the pseudocode assumes */
        WordItem *oneLineList = extractOneLine(line);
        merge_into_global(&global, oneLineList);
        free_word_list(oneLineList);
    }

    free(line);
    fclose(fp);

    /* Create array & sort alphabetically */
    int n = 0;
    Entry *arr = list_to_array(global, &n);

    qsort(arr, (size_t)n, sizeof(Entry), cmp_alpha);
    printf("\n=== SORTED BY WORD (ALPHABETICAL) ===\n");
    print_table(arr, n);

    /* Sort by count descending */
    qsort(arr, (size_t)n, sizeof(Entry), cmp_count_desc);
    printf("\n=== SORTED BY COUNT (DESCENDING) ===\n");
    print_table(arr, n);

    free(arr);
    free_word_list(global);
    return 0;
}
