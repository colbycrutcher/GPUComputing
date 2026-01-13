#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "functions.c"

int main() {
    FILE *file;
    char word[100];
    
    file = fopen("testfile1", "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    
    while (fscanf(file, "%s", word) == 1) {
       // printf("%s\n", word);
        // checkIfWordinList(word);
        singleLetter(word);
        
    }

    printList();
    



    fclose(file);
    
    return 0;
}


