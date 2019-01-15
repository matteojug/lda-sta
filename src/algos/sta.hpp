#pragma once
#include "algo.hpp"
#include "../reduction.hpp"

class STABasedAlgo : virtual public Algo {

    protected:

    mt19937 randomGen;
    string staDumpPath;
    
    int addedNgrams;
    enum { ngramModePrefix, ngramModeAll, ngramModeSample };
    
    inline void addNgrams(NgramsVec &ngrams, Ngram &current, int count = 1){
        if (ngrams[current.size].find(current) == ngrams[current.size].end())
            current.updateMultiplicity();
        ngrams[current.size][current] += count;
        addedNgrams += count;
    }

    void populateAllNgrams(NgramsVec &ngrams, vector<int> &doc, int ngramSize, Ngram &current, int lastIndex = -1, int currentCount = 1){
        if (current.size) addNgrams(ngrams, current, currentCount);
        if (current.size == ngramSize) return;
        for (int i = lastIndex+1; i < doc.size(); i++){
            Ngram tmp(current);
            tmp.add(doc[i]);
            populateAllNgrams(ngrams, doc, ngramSize, tmp, i, lastIndex != -1 && doc[lastIndex] == doc[i] ? currentCount*2 : currentCount);
        }
    }
    
    void sampleNgrams(NgramsVec &ngrams, vector<int> &doc, int ngramSize, int sampleCount, vector<Ngram> &ngramSampleBox){
        Ngram emptyNgram;
        vector<int> sampledDoc(ngramSize);
        for (int i = 0; i < min(sampleCount, (int)ngramSampleBox.size()); i++){
            int nextId = i + randomGen() % (ngramSampleBox.size()-i);
            swap(ngramSampleBox[i], ngramSampleBox[nextId]);

            auto t = ngramSampleBox[i].toVec();
            for (int j = 0; j < t->size(); j++) sampledDoc[j] = doc[t->at(j)];
            delete t;
            emptyNgram.clear();
            populateAllNgrams(ngrams, sampledDoc, ngramSize, emptyNgram);
        }
    }

    void getNgramsInit(NgramsVec &ngrams, int ngramSize){
        ngrams.clear();
        ngrams.resize(ngramSize+1);
        addedNgrams = 0;
    }

    // Specialized version for c=1
    void getNgramsC1(NgramsVec &ngrams){
        debl("Using getNgramsC1");
        Ngram single, pair;
        for (auto &doc : inputData->corpus){
            for (int i = 0; i < doc.size(); i++){
                single.clear(); single.add(doc[i].first);
                addNgrams(ngrams, single, doc[i].second);
                if (doc[i].second > 1){
                    pair = single; pair.add(doc[i].first);
                    addNgrams(ngrams, pair, doc[i].second*(doc[i].second-1));
                }
                for (int j = i+1; j < doc.size(); j++){
                    pair = single; pair.add(doc[j].first);
                    addNgrams(ngrams, pair, doc[i].second*doc[j].second);
                }
            }
        }
    }
    void getNgramsGeneric(NgramsVec &ngrams, int ngramSize, int ngramMode){
        debl("Using getNgramsGeneric");
        vector<int> flatBow;
        flatBow.reserve(inputData->maxDocLen+1);
        vector<Ngram> ngramSampleBox;
        Ngram emptyNgram;

        if (ngramMode == ngramModePrefix || ngramMode == ngramModeSample){
            function<void(Ngram&,int)> populateSampleBox;
            int docLen = inputData->maxDocLen;
            populateSampleBox = [&](Ngram &current, int lastIndex){
                if (current.size == ngramSize){
                    ngramSampleBox.push_back(current);
                    return;
                }
                for (int i = lastIndex+1; i < docLen; i++){
                    Ngram tmp(current);
                    tmp.add(i);
                    populateSampleBox(tmp, i);
                }
            };
            emptyNgram.clear();
            populateSampleBox(emptyNgram, -1);
            debl("ngramSampleBox size: "<<ngramSampleBox.size());
        }
        for (auto &doc : inputData->corpus){
            flatBow.clear();
            for (auto &x : doc)
                for (int j = 0; j < x.second; j++)
                    flatBow.push_back(x.first);
            switch (ngramMode){
                case ngramModePrefix:
                case ngramModeSample:
                    sampleNgrams(ngrams, flatBow, ngramSize, ngarmSampleCount, ngramSampleBox);
                    break;
                case ngramModeAll:
                    emptyNgram.clear();
                    populateAllNgrams(ngrams, flatBow, ngramSize, emptyNgram);
                    break;
            }
        }
    }
    // Populates the NgramsVec with ngrams up to size ngramCount, normalizing the distributions
    void getNgrams(NgramsVec &ngrams, int ngramSize){
        int ngramMode;
        auto mode = getParamString(PARAM_TAG_NGRAM_MODE, PARAM_NGRAM_MODE_PREFIX);
        if (mode == PARAM_NGRAM_MODE_PREFIX){
            ngramMode = ngramModePrefix;
            ngarmSampleCount = 1;
        }
        else if (mode == PARAM_NGRAM_MODE_ALL) ngramMode = ngramModeAll;
        else if (mode == PARAM_NGRAM_MODE_SAMPLE) ngramMode = ngramModeSample;
        else die("Unknown ngramMode value");

        bool forceGeneric = getParamInt(PARAM_TAG_FORCE_GENERIC, 0);

        getNgramsInit(ngrams, ngramSize);

        if (inputData->c == 1 && ngramMode == ngramModeAll && !forceGeneric)
            getNgramsC1(ngrams);
        else
            getNgramsGeneric(ngrams, ngramSize, ngramMode);

        debl("Added "<<addedNgrams<<" n-grams");
        for (int i = 1; i < ngrams.size(); i++){
            Real total = 0;
            for (auto &x : ngrams[i])
                total += x.second * x.first.multiplicity;
            debl(i<<"-grams count:"<<total<<", unique:"<<ngrams[i].size());
            Real totCheck = 0;
            for (auto &x : ngrams[i])
                x.second /= total,
                totCheck += x.second * x.first.multiplicity;
            debVar(totCheck);
        }
    }

    public:

    NgramsVec staNgrams, ldaNgrams;
    vector<Real> staPrenormalizationSum;
    int ngarmSampleCount;
    
    virtual void computeFromSTA(){};
    
    virtual void parseInputDataParams(){
        Algo::parseInputDataParams();
        ngarmSampleCount = getParamReal(PARAM_TAG_NGRAM_MODE_SAMPLE_COUNT, 0);
        randomGen = mt19937(getParamInt(PARAM_TAG_RANDOM_SEED, RANDOM_SEED));
        staDumpPath = getParamString(PARAM_TAG_STADUMP_PATH, "anchor_ngrams.txt");
    }

    virtual void compute() {
        int ngramLen = inputData->c+1;
        debl("# Getting LDA ngrams (up to "<<ngramLen<<"-grams)");
        
        Timer timer;
        timer.reset();
        getNgrams(ldaNgrams, ngramLen);
        timer.printTime("LDA getNgrams");

        debl("# Getting STA");
        timer.reset();
        lda2sta(staNgrams, ldaNgrams, inputData->K, inputData->alpha);
        timer.printTime("lda2sta");

        timer.reset();
        staPrenormalizationSum.assign(ngramLen+1, 0);
        for (int i = 1; i <= ngramLen; i++){
            for (auto &kv : staNgrams[i]){
                staPrenormalizationSum[i] += kv.second * kv.first.multiplicity;
            }
            // for (auto &kv : staNgrams[i]) kv.second /= staPrenormalizationSum[i];
            debl("[?] S_"<<i<<" pre normalization sum = "<<staPrenormalizationSum[i]);
        }
        timer.printTime("normalization");
        if (inputData->algId == ALGO_STA_DUMP) dumpNgrams(staNgrams, staDumpPath);

        topics.clear();
        timer.reset();
        computeFromSTA();
        timer.printTime("computeFromSTA");
    }
};