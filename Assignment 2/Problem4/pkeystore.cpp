#include <bits/stdc++.h>
#include "pkeystore.h"
using namespace std;

pthread_t threads[4];

unordered_map<uint32_t, uint64_t> hash_table;
queue<struct operation*>job_queue;


pthread_mutex_t mutex1;
pthread_mutex_t mutex2;
pthread_mutex_t mutex3;

int jd=0;         //number of job done
int flag = 0;    //to create threads
int total_jobs; //specified in test.cpp

void *Work_to_do(void *args){

    uint8_t curr_type;
    uint32_t curr_key;
    uint64_t curr_value;

    while(true){
        
        pthread_mutex_lock(&mutex3);
        if(jd == total_jobs){
            pthread_mutex_unlock(&mutex3);
            break;
        }
        pthread_mutex_unlock(&mutex3);


        pthread_mutex_lock(&mutex1);
        if(job_queue.size() == 0){
            pthread_mutex_unlock(&mutex1);
            continue;
        }
        curr_type = job_queue.front()->type;
        curr_key = job_queue.front()->key;
        curr_value = job_queue.front()->value;
        job_queue.pop();
        pthread_mutex_unlock(&mutex1);


        pthread_mutex_lock(&mutex2);
        if(curr_type == 0)
        {   //insert
            hash_table[curr_key] = curr_value;
            // pthread_mutex_lock(&mutex3);
            // jd++;
            // pthread_mutex_unlock(&mutex3); 
        }
        else if(curr_type == 1)
        {   //update
            if(hash_table.count(curr_key))
            {
                hash_table[curr_key] = curr_value;
            }
            else{
                cout<<"Value we are updating in not present in hash table "<<endl;
            }
            // pthread_mutex_lock(&mutex3);
            // jd++;
            // pthread_mutex_unlock(&mutex3); 
        }
        else if(curr_type == 2)
        {   //delete
            if(hash_table.count(curr_key))
            {
                hash_table.erase(curr_key);
            }
            else{
                cout<<"Value we are deleting in not present in hash table "<<endl;
            }
            // pthread_mutex_lock(&mutex3);
            // jd++;
            // pthread_mutex_unlock(&mutex3); 
        }
        else if(curr_type == 3)
        {     //find
            if(hash_table.count(curr_key))
            {
                cout<<"Key = "<<curr_key<<"Value = "<<hash_table[curr_key]<<endl; 
            }
            else
            {
                cout << "Key is not present in our hash table" <<endl;
            }
            // pthread_mutex_lock(&mutex3);
            // jd++;
            // pthread_mutex_unlock(&mutex3); 
        }
        pthread_mutex_unlock(&mutex2);

        pthread_mutex_lock(&mutex3);
        jd++;
        pthread_mutex_unlock(&mutex3);
    }
    return (void *)NULL;
}

int enqueue(struct operation *op){

    

    pthread_mutex_lock(&mutex1);
    if(job_queue.size() == 8){
        pthread_mutex_unlock(&mutex1);
        return -1;
    }
    job_queue.push(op);
    pthread_mutex_unlock(&mutex1);

    if(flag == 0){
        //creating thread for the first time
        for(int i = 0; i<4; i++){
            pthread_create(&threads[i], NULL, Work_to_do, NULL);
        }
        flag = 1;
    }

    return job_queue.size() - 1;
}