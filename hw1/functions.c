#include <stdio.h>
#include <stdlib.h>
#include <string.h>

   //Linked List
    struct Node {
        char word;           // Data field
        struct Node* next;  // Pointer to the next node
    };
    struct Node* head = NULL;


void singleLetter(char *word){
    if (strlen(word) == 1) {
            if(word[0] == 'a' || word[0] == 'i'){
                printf("Keep word: %s\n", word);
        } else {
            printf("Discard word: %s\n", word);
        }
    }
}


//Print the Linked List For testing
void printList() {
    struct Node* current = head;
    while (current != NULL) {
        printf("%d ", current->word);
        current = current->next;
    }
    printf("\n");
}

// void printTables(char word){
//     for 
// }