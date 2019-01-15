#include <bits/stdc++.h>
using namespace std;

#include "definitions.hpp"
#include "utils.hpp"

#include "algos/algo.hpp"
#include "algos/sta.hpp"
#include "algos/anchor.hpp"
#include "algos/agglomerative.hpp"
#include "algos/anchor_fast.hpp"

Algo* anchor(InputData *inputData){
    MAX_VOCAB = inputData->vocabSize+1;

    #ifdef DEBUG
    inputData->printInfo();
    #endif

    Algo *algo = new Algo();
    switch (inputData->algId){
        case ALGO_STA_DUMP:
            algo = new STABasedAlgo();
            break;
        case ALGO_GREEDY:
            algo = new Anchor();
            break;
        case ALGO_LAZY_GREEDY:
            algo = new AnchorFast();
            break;
        case ALGO_CLUSTER:
            algo = new AnchorAgglomerativeCluster();
            break;
        case ALGO_LAZY_CLUSTER:
            algo = new AnchorFastAgglomerativeCluster();
            break;
        default:
            cerr<<"Unknown algId: "<<inputData->algId<<endl;
    }
    algo->setInputData(inputData);
    algo->compute();

    return algo;
}