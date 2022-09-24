#include<bits/stdc++.h>
#include<pthread.h>
using namespace std;

int P;
int W;
int B;

fstream newfile;
fstream outfile;

int number_of_elements=0;
int start_index=0;

pthread_mutex_t mutex1;
pthread_mutex_t mutex2;
pthread_mutex_t mutex3;
pthread_mutex_t mutex4;
// pthread_mutex_t mutex5;


int terminate_signal=0;


bool prime(int n){
    if(n<=1)return false;
    for(int i=2;i*i<=n;i++){
        if(n%i==0)return false;
    }
    return true;
}

void * writing_to_buffer(void *shared_buffer){
    
    string s1;
    int flag=1;
    
    pthread_mutex_lock(&mutex2);
    while(getline(newfile,s1)){
        int *arr=(int *)shared_buffer;
        flag=1;
        while(flag==1)
        {
            pthread_mutex_lock(&mutex1);
            if(number_of_elements<B){
                arr[(start_index+number_of_elements)%B]=stoi(s1);
                number_of_elements++;
                flag=0;
            } 

            pthread_mutex_unlock(&mutex1);
        }
    }
    pthread_mutex_unlock(&mutex2);
    
    pthread_mutex_lock(&mutex3);
    terminate_signal=terminate_signal+1;
    pthread_mutex_unlock(&mutex3);

    return (void*)NULL;
}



void *reading_from_buffer(void* shared_buffer){
    
     
    int *arr=(int *)shared_buffer;

    //  pthread_mutex_lock(&mutex3);
    while((terminate_signal<P) || (number_of_elements> 0))
    {
        pthread_mutex_lock(&mutex1);
        
        if(number_of_elements>0)
        {
            int curr=arr[start_index];
            start_index=(start_index+1)%B;
            number_of_elements--;
            pthread_mutex_unlock(&mutex1);

            if(prime(curr)){
                pthread_mutex_lock(&mutex4);
                outfile<<curr<<endl;
                pthread_mutex_unlock(&mutex4);
            }

            pthread_mutex_lock(&mutex1);
        }
        pthread_mutex_unlock(&mutex1);
    }

    //  pthread_mutex_unlock(&mutex3);
    return (void*)NULL;
    
}



int main(int argc,char *argv[]){

    
    P=stoi(argv[1]);
    B=stoi(argv[2]);
    W=stoi(argv[3]);

    int *shared_buffer= (int *)malloc(4*B);

    pthread_mutex_init(&mutex1,NULL);
    pthread_mutex_init(&mutex2,NULL);
    pthread_mutex_init(&mutex3,NULL);
    pthread_mutex_init(&mutex4,NULL);


    pthread_t producer_threads[P];
    pthread_t worker_threads[W];

    
    newfile.open("input.txt",ios::in);
    outfile.open("prime.txt",ios::out);

    for(int i=0;i<P;i++){
        pthread_create(&producer_threads[i],NULL,writing_to_buffer,(void *)shared_buffer);
    }
    for(int i=0;i<W;i++){
        pthread_create(&worker_threads[i],NULL,reading_from_buffer,(void *)shared_buffer);
    }

    for(int i=0;i<P;i++){
        pthread_join(producer_threads[i],NULL);
    }


    for(int i=0;i<W;i++){
        pthread_join(worker_threads[i],NULL);
    }

}

