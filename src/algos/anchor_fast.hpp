#pragma once
#include "anchor.hpp"

class AnchorFast : virtual public Anchor {

    protected:

    vector<Real> ngramCounts;

    void getNgrams_Rho(NgramsVec &ngrams){
        debl("Using getNgrams_Rho");
        Ngram single, pair;
        int count;
        ngramCounts.assign(3,0);
        for (auto &doc : inputData->corpus){
            count = 0;
            for (int i = 0; i < doc.size(); i++){
                single.clear(); single.add(doc[i].first);
                addNgrams(ngrams, single, doc[i].second);
                if (doc[i].second > 1){
                    pair = single; pair.add(doc[i].first);
                    addNgrams(ngrams, pair, doc[i].second*(doc[i].second-1));
                }
                count += doc[i].second;
            }
            ngramCounts[1] += count;
            ngramCounts[2] += count*(count-1);
        }
        for (int i = 1; i < ngrams.size(); i++){
            debl(i<<"-grams count:"<<ngramCounts[i]<<", unique:"<<ngrams[i].size());
            for (auto &x : ngrams[i])
                x.second /= ngramCounts[i];
        }
    }

    unordered_map<Ngram,int> candidateSet;
    vector<unordered_map<int,int>> postingList;

    int requireTauCalls;
    void requireTau(const Ngram &cand1, const Ngram &cand2){
        Ngram key = Ngram::join(cand1, cand2);
        if (staNgrams[key.size].count(key)) return;

        requireTauCalls++;
        int cand1Idx = candidateSet[cand1], cand2Idx = candidateSet[cand2];
        if (postingList[cand1Idx].size() > postingList[cand2Idx].size()) swap(cand1Idx, cand2Idx);
        
        Real cnt = 0;
        for (auto &x1 : postingList[cand1Idx]){
            if (!postingList[cand2Idx].count(x1.first)) continue;
            cnt += x1.second * postingList[cand2Idx][x1.first];
        }
        key.updateMultiplicity();
        ldaNgrams[key.size][key] = cnt / ngramCounts[key.size];
        lda2staSingleAdd(staNgrams, key, ldaNgrams[key.size][key], inputData->K, inputData->alpha, false, hasParam(PARAM_TAG_SKIP_REDUCTION));
    }

    void getTopics(vector<Ngram> anchors){
        unordered_set<Ngram> anchorsSet(anchors.begin(), anchors.end());
        unordered_map<Ngram, Real> newCnts;
        Ngram single;
        for (auto &anchor : anchors){
            int anchorIdx = candidateSet[anchor];
            auto ngramTmpV = anchor.toMultiset();
            for (auto &kv : postingList[anchorIdx]){
                auto &doc = inputData->corpus[kv.first];
                for (int i = 0; i < doc.size(); i++){
                    single.clear(); single.add(doc[i].first);
                    if (anchorsSet.count(single) && (single < anchor || single == anchor)) continue;

                    ngramTmpV->insert(doc[i].first);
                    Ngram tmpNgram(ngramTmpV->begin(), ngramTmpV->end());
                    ngramTmpV->erase(ngramTmpV->find(doc[i].first));

                    if (!newCnts.count(tmpNgram)) tmpNgram.updateMultiplicity();
                    newCnts[tmpNgram] += kv.second * doc[i].second;
                }
            }
        }
        for (auto &kv : newCnts){
            ldaNgrams[kv.first.size][kv.first] = kv.second / ngramCounts[kv.first.size];
            lda2staSingleAdd(staNgrams, kv.first, ldaNgrams[kv.first.size][kv.first], inputData->K, inputData->alpha, false, hasParam(PARAM_TAG_SKIP_REDUCTION));
        }
        return Anchor::getTopics(anchors);
    }

    public:

    Real tau(const Ngram &cand1, const Ngram &cand2){
        requireTau(cand1, cand2);
        return Anchor::tau(cand1, cand2);
    }
    Real tauD(const Ngram &cand1, const Ngram &cand2){
        requireTau(cand1, cand2);
        return Anchor::tauD(cand1, cand2);
    }

    void compute() {
        int ngramLen = inputData->c+1;
        debl("# Getting LDA ngrams (up to "<<ngramLen<<"-grams)");
        if (ngramLen > 2) die("Specialized version only for C=1, revert to generic one for higher C");
        getNgramsInit(ldaNgrams, ngramLen);

        Timer timer;
        
        timer.reset();
        getNgrams_Rho(ldaNgrams);
        timer.printTime("Rho - LDA getNgrams");
        timer.reset();
        function<bool(const Ngram&)> rhoFilter = [&](const Ngram &x){ 
            if (x.size != 2) return false;
            auto m = x.toVec();
            bool eq = (m->at(0) == m->at(1));
            delete m;
            return eq;
        };
        lda2sta(staNgrams, ldaNgrams, inputData->K, inputData->alpha, true, hasParam(PARAM_TAG_SKIP_REDUCTION), rhoFilter);

        timer.printTime("Rho - lda2sta");

        vector<pair<Ngram, Real>> candidateAnchors;
        topics.clear();

        timer.reset();
        getCandidatesAnchor(candidateAnchors);
        timer.printTime("getCandidatesAnchor");
        if (candidateAnchors.size() < inputData->K)
            debl("Got "<<candidateAnchors.size()<<" anchors, K = "<<inputData->K<<". That's a problem.");
        
        timer.reset();
        candidateSet.clear();
        for (int i = 0; i < candidateAnchors.size(); i++) candidateSet[candidateAnchors[i].first] = i;
        postingList.clear(); postingList.resize(candidateAnchors.size());
        Ngram tmpNgram;
        for (int j = 0; j < inputData->corpus.size(); j++){
            auto &doc = inputData->corpus[j];
            for (int i = 0; i < doc.size(); i++){
                tmpNgram.clear(); tmpNgram.add(doc[i].first);
                if (!candidateSet.count(tmpNgram)) continue;
                postingList[candidateSet[tmpNgram]][j] = doc[i].second;
            }
        }
        timer.printTime("postingList");
        
        timer.reset();
        requireTauCalls = 0;
        auto anchors = getAnchors(candidateAnchors);
        debVar(requireTauCalls);
        timer.printTime("getAnchors");
        
        timer.reset();
        getTopics(anchors);
        timer.printTime("getTopics");
    }
};