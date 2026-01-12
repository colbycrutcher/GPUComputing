#include <stdio.h>
#include <stdlib.h>
#include <string.h>





void singleLetter(char *word){
    if (strlen(word) == 1) {
            if(word[0] == 'a' || word[0] == 'i'){
                printf("Keep word: %s\n", word);
        } else {
            printf("Discard word: %s\n", word);
        }
    }
    
}

int main() {
    FILE *file;
    char word[100];
    
    file = fopen("testfile1", "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    
    while (fscanf(file, "%s", word) == 1) {
        printf("%s\n", word); 
        // checkIfWordinList(word);
        singleLetter(word);
        
    }
    



    fclose(file);
    
    return 0;
}


