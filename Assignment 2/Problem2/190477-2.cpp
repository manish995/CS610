#include<bits/stdc++.h>
#include<random>
using namespace std;

int n;
int p;

struct thread_args{
    int * start_arr;
    int start_index;
    int sublist;
};


void generate_rand_nums(int max,int *arr){
    int range_from=0;
    int range_to=max-1;
    random_device rd;
    mt19937 gen(20);
    uniform_int_distribution<int> distr(range_from,range_to);
    for(int i=0;i<max;i++){
        arr[i]=distr(gen);
        // cout<<arr[i]<<" ";

    }
    // cout<<endl;
}

void * count_inversion(void * args){
    struct thread_args * temp= (struct thread_args *)args;
    int * res=(int *)malloc(4);
    *res=0;
    for(int i=temp->start_index;i<temp->start_index+temp->sublist;i++)
     {
        for(int j=i+1;j<n;j++){
            if(temp->start_arr[i]  >temp->start_arr[j] ){
                *res=*res+1;
            }
        }
    }

    return (void *)res;
}

int main(int argc, char * argv[]){

    n=stoi(argv[1]);
    p=stoi(argv[2]);

    pthread_t threads[min(n,p)];

    struct thread_args Argument_to_threads[min(n,p)];
    int * res[min(n,p)];
    int * arr= (int *)malloc(4*n);

    generate_rand_nums(n,arr);

    int curr=0;
    for(int i=0;i<min(n,p);i++){
        
        if(i<n%p){
            Argument_to_threads[i].start_arr=arr;
            Argument_to_threads[i].start_index=curr;
            Argument_to_threads[i].sublist=n/p+1;
            curr=curr+n/p+1;
        }
        else{
            Argument_to_threads[i].start_arr=arr;
            Argument_to_threads[i].start_index=curr;
            Argument_to_threads[i].sublist=n/p;
            curr=curr+n/p;
        }

        
    }
    
    // else{
    //     for(int i=0;i<min(n,p);i++){
    //         Argument_to_threads[i].start_arr=arr;
    //         Argument_to_threads[i].start_index=i;
    //         Argument_to_threads[i].sublist=1;
    //     }
    // }

    // for(int i=0;i<min(n,p);i++){
    //     cout<<Argument_to_threads[i].start_index<<" "<<Argument_to_threads[i].sublist<<endl;
    // }


    for(int i=0;i<min(n,p);i++){
        pthread_create(&threads[i],NULL,count_inversion,(void*)&Argument_to_threads[i]);
    }
    for(int i=0;i<min(n,p);i++){
        pthread_join(threads[i],(void**)&res[i]);
    }
    long long total_inversion=0;
    for(int i=0;i<min(n,p);i++){
        // cout<<*res[i]<<endl;
        total_inversion+=*res[i];

    }

    cout<<total_inversion<<endl;
    return 0;

}