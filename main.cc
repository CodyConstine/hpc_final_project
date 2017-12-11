#include <iostream>
#include <math.h>
#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <chrono>
#include "predict.h"

using namespace std::chrono;


int main(int argv, char ** argc) {
    int n = atoi(argc[1]);
    double error_bound = atof(argc[2]);
    int seconds = atoi(argc[3]);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    predict(n,error_bound,seconds);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    printf("predict returned after Wall Time: %f seconds\n",duration/1e6);

    return 0;
}
