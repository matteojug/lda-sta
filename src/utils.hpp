#pragma once

#include "stdc++.h"
using namespace std;

#include "definitions.hpp"

// Distance utils
template<typename T> inline T distanceL1(const vector<T> &a, const vector<T> &b){
    T ret = 0;
    for (int i = 0; i < a.size(); i++)
        ret += abs(a[i]-b[i]);
    return ret;
}
template<typename T> inline T distanceLinf(const vector<T> &a, const vector<T> &b){
    T ret = abs(a[0]-b[0]);
    for (int i = 1; i < a.size(); i++)
        ret = max(ret, abs(a[i]-b[i]));
    return ret;
}
template<typename T> inline T innerProduct(const vector<T> &a, const vector<T> &b){
    T ret = 0;
    for (int i = 0; i < a.size(); i++)
        ret += abs(a[i]*b[i]);
    return ret;
}

// Clamp a value between the two bounds
template<typename T> inline T clamp(const T &x, const T &minVal, const T &maxVal){
    return min(max(x, minVal), maxVal);
}

// Partition utility
template<class V> using Partition = vector<multiset<V>>;
template<class V> void partition(vector<V> &vec, int index, Partition<V> &currentPartition, function<void(const Partition<V>&)>& lambda){
    if (index == vec.size()){
        lambda(currentPartition);
        return;
    }
    // Add the current item to one partition
    for (int i = 0; i < currentPartition.size(); i++){
        currentPartition[i].insert(vec[index]);
        partition(vec, index+1, currentPartition, lambda);
        currentPartition[i].erase(currentPartition[i].find(vec[index]));
    }
    // Create a new singleton partition
    currentPartition.push_back((multiset<V>){vec[index]});
    partition(vec, index+1, currentPartition, lambda);
    currentPartition.pop_back();
}
// Given a vector, compute all the partitions of it, and for each of those the function lambda is called with it as argument
template<class V> void partition(vector<V> &vec, function<void(const Partition<V>&)>& lambda){
    Partition<V> tmpPartition;
    return partition(vec, 0, tmpPartition, lambda);
}

// Returns the top-k indices of v (based on the values of it)
template<class V> vector<int> getTopKIdx(vector<V> &v, int k){
    priority_queue<pair<V,int> > pq;
    for (int i = 0; i < v.size(); i++){
        pq.push(make_pair(-v[i], i)); // cheap reverse trick
        if (pq.size() > k)
            pq.pop();
    }
    vector<int> ids(pq.size());
    while (!pq.empty()){
        auto top = pq.top(); pq.pop();
        ids[pq.size()] = top.second;
    }
    return ids;
}

void dumpNgrams(const NgramsVec &ngramsVec, const string &path, bool permute = true){
    FILE *fout = fopen(path.c_str(), "w");
    if (fout == NULL)
        debl("dumpNgrams failed with errno:"<<errno);
    for (auto &ngrams : ngramsVec){
        map<Ngram,double> sortedNgrams;
        for (auto &ngram : ngrams){
            auto v = ngram.first.toVec();
            vector<int> perm(v->size());
            for (int i = 0; i < perm.size(); i++) perm[i] = i;
            do {
                vector<int> tmp;
                for (auto &i : perm) tmp.push_back(v->at(i));
                sortedNgrams[Ngram(tmp.begin(), tmp.end())] = ngram.second;
            } while (next_permutation(perm.begin(), perm.end()) && permute);
            delete v;
        }
        for (auto &ngram : sortedNgrams){
            auto v = ngram.first.toVec();
            for (int i = 0; i < v->size(); i++){
                if (i) fprintf(fout, " ");
                fprintf(fout, "%d", v->at(i));
            }
            delete v;
            fprintf(fout, "\t%.15f\n", ngram.second);
        }
    }
    fclose(fout);
    debl("# Dumped ngramms");
}

// Timer class to get times for algo phases
struct Timer {
    chrono::time_point<chrono::steady_clock> ts;
    Timer(){
        reset();
    }
    void reset(){
        ts = chrono::steady_clock::now();
    }
    long long time(){
        return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-ts).count();
    }
    void printTime(string tag){
        deblTime("[T] "<<tag<<": "<<time()<<"ms");
    }
};
