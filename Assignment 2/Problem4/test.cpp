#include "pkeystore.h"
#include <bits/stdc++.h>
using namespace std;

extern int total_jobs;
extern pthread_t threads[4];

int main(int argc, char **argv){

    total_jobs = stoi(argv[1]);
    //operations array
    struct operation ops[total_jobs];
    
    ops[0].type = 0;
    ops[0].key = 1;
    ops[0].value = 100;

    ops[1].type = 0;
    ops[1].key = 2;
    ops[1].value = 200;

    ops[2].type = 0;
    ops[2].key = 3;
    ops[2].value = 300;

    ops[3].type = 1;
    ops[3].key = 1;
    ops[3].value = 50;

    ops[4].type = 2;
    ops[4].key = 2;
    ops[4].value = 150;

    ops[5].type = 3;
    ops[5].key = 3;
    ops[5].value = 250;

    ops[6].type = 3;
    ops[6].key = 1;
    ops[6].value = 100;

    ops[7].type = 3;
    ops[7].key = 1;
    ops[7].value = 100;

    ops[8].type = 3;
    ops[8].key = 1;
    ops[8].value = 100;

    ops[9].type = 3;
    ops[9].key = 1;
    ops[9].value = 100;

    for(int i=0; i<total_jobs; i++){
        enqueue(&ops[i]);
    }

    for(int i = 0; i<4; i++){
        pthread_join(threads[i], NULL);
    }

}