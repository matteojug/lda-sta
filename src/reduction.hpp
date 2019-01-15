#pragma once
#include "definitions.hpp"
#include "utils.hpp"

void lda2staSingleAdd(NgramsVec &sta, const Ngram &ldaNgram, Real ngramCount, int k, Real alpha, bool insertIfZero = false){

    Real ka = alpha * k, l = ldaNgram.size;
    Real partitionsSum = 0;
    
    function<void(const Partition<int>&)> lambda = [&](const Partition<int> &partition) {
        if (partition.size() < 2) return;
        Real product = 1;
        for (auto &x : partition){
            Ngram tmpNgram(x.begin(), x.end());
            if (!sta[tmpNgram.size].count(tmpNgram))
                product *= 0;
            else
                product *= ka * sta[tmpNgram.size][tmpNgram];
        }
        partitionsSum += product;
    };
    auto ngramVec = ldaNgram.toVec();
    partition(*ngramVec, lambda);
    delete ngramVec;

    Real dCoeff = gamma(ka + l) / (gamma(ka + 1) * gamma(l));
    Real minusCoeff = 1. / (ka * gamma(l));
    Real prob = clamp<Real>(dCoeff * ngramCount - minusCoeff * partitionsSum, 0, 1);
    if (prob < EPS && !insertIfZero) return;
    
    sta[l][ldaNgram] = prob;
}
void lda2sta(NgramsVec &sta, const NgramsVec &lda, int k, Real alpha, bool initSTA, function<bool(const Ngram&)> &ngramFilter){
    if (initSTA){
        sta.clear();
        sta.resize(lda.size());
        sta[1] = lda[1]; // Single word dist is the same
    }
    for (int l = 2; l < lda.size(); l++){
        for (auto &ngram : lda[l]){
            if (!ngramFilter(ngram.first)) continue;
            lda2staSingleAdd(sta, ngram.first, ngram.second, k, alpha);
        }
    }
}
void lda2sta(NgramsVec &sta, const NgramsVec &lda, int k, Real alpha, bool initSTA = true){
    function<bool(const Ngram&)> filter = [](const Ngram &x){ return true; };
    lda2sta(sta, lda, k, alpha, initSTA, filter);
}