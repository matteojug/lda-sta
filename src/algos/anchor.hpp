#pragma once
#include "sta.hpp"

class Anchor : virtual public STABasedAlgo {
    protected:
    
    unordered_map<Ngram,Topic> anchorTopicCache;
    inline const Topic& generateTopicCached(const Ngram &anchor){
        if (!anchorTopicCache.count(anchor)){
            anchorTopicCache[anchor] = generateTopic(anchor, staNgrams[anchor.size+1]);
        }
        return anchorTopicCache[anchor];
    }
    
    virtual Topic generateTopic(const Ngram &anchor, const Ngrams &ngrams){
        Topic topic(inputData->vocabSize);
        multiset<int>* ngramTmpV = anchor.toMultiset();
        Real sum = 0;
        for (int i = 0; i < inputData->vocabSize; i++){
            ngramTmpV->insert(i);
            Ngram tmpNgram(ngramTmpV->begin(), ngramTmpV->end());
            ngramTmpV->erase(ngramTmpV->find(i));
            auto it = ngrams.find(tmpNgram);
            topic[i] = (it != ngrams.end() ? it->second : 0);
            sum += topic[i];
        }
        for (auto &topicVal : topic){
            if (sum == 0.0)
                topicVal = 1.0/inputData->vocabSize;
            else
                topicVal /= sum;
        }
        delete ngramTmpV;
        return topic;
    }
    
    Real anchorDistributionError(const vector<Ngram> &anchors, int ngramSize){
        Topic distr(inputData->vocabSize);
        for (auto &anchor : anchors){
            const Topic& topic = generateTopicCached(anchor);
            for (int i = 0; i < distr.size(); i++)
                distr[i] += topic[i] / anchors.size();
        }
        Real error = 0, computed;
        for (auto &ngram : staNgrams[ngramSize]){
            computed = 1.0;
            ngram.first.forEachReversed([&](int w){
                computed *= distr[w];
            });
            Real pairError = ngram.first.multiplicity * abs(ngram.second - computed);
            error += pairError;
        }
        return error;
    }

    virtual void filterCandidatesAnchor(vector<pair<Ngram, Real>> &candidatesAnchor){
        if (candidateSize >= 0){
            debl("# Trimming candidate anchors to "<<candidateSize);
            candidatesAnchor.resize(candidateSize);
            return;
        }
        Real minRho = (1-delta)/pow(1+delta,2) * pow(1-eps, 2);
        debl("# Finding candidate anchors, minRho = "<<minRho);
        int cutOffB = lower_bound(candidatesAnchor.begin(), candidatesAnchor.end(), minRho, [](pair<Ngram, Real> x, Real v){return x.second >= v;}) - candidatesAnchor.begin();
        candidatesAnchor.resize(cutOffB);
    }

    virtual void getCandidatesAnchor(vector<pair<Ngram, Real>> &candidatesAnchor){
        for (auto &kv : staNgrams[inputData->c])
            candidatesAnchor.push_back(make_pair(kv.first, rho(kv.first)));
        
        sort(candidatesAnchor.begin(), candidatesAnchor.end(), [](pair<Ngram, Real> a, pair<Ngram, Real> b){
            return a.second > b.second;
        });
        filterCandidatesAnchor(candidatesAnchor);
        debl("[?] staRhoCandidates size: "<<candidatesAnchor.size());
    }

    virtual vector<Ngram> getAnchors(vector<pair<Ngram, Real>> &candidatesAnchor){
        if (distanceFunction == dfTau || distanceFunction == dfTaud){
            if (thresholdDist < 0)
                thresholdDist = pow(1-eps,2)/pow(1+delta,2) - delta * pow(p/inputData->K,2)/(inputData->K*pow(1+delta,2));
        }

        debl("# Finding anchors:");
        deb("[?] candidatesAnchor.size: "<<candidatesAnchor.size()<<endl);
        deb("[?] distanceFunction: "<<distanceFunctionStr<<endl);
        deb("[?] thresholdDist: "<<thresholdDist<<endl);

        vector<int> anchorsIndexes;
        for (int i = 0; i < candidatesAnchor.size() && anchorsIndexes.size() < inputData->K; i++){
            auto cand = candidatesAnchor[i];
            deb("[?] Candidate: "); cand.first.print(inputData->vocab, false); debl("\t => "<<cand.second);
            
            bool validAnchor = true;
            for (auto &anchor : anchorsIndexes){
                deb("\tvs "); candidatesAnchor[anchor].first.print(inputData->vocab);
                Real dist;
                switch (distanceFunction){
                    case dfTau:
                        dist = tau(cand.first, candidatesAnchor[anchor].first);
                        break;
                    case dfTaud:
                        dist = tauD(cand.first, candidatesAnchor[anchor].first);
                        break;
                    case dfLinf:
                        dist = inducedTopicDistance(cand.first, candidatesAnchor[anchor].first, distanceLinf<Real>);
                        break;
                    case dfInnerprod:
                        dist = inducedTopicDistance(cand.first, candidatesAnchor[anchor].first, innerProduct<Real>);
                        break;
                    default:
                        die("Unknown distanceFunction: "<<distanceFunctionStr);
                }
                deb("\t\tdist = "<<dist<<" [ thr= "<<thresholdDist<<"] ");
                switch (distanceFunction){
                    case dfTau:
                    case dfTaud:
                        if (dist > thresholdDist)
                            validAnchor = false;
                        break;
                    case dfLinf:
                    case dfInnerprod:
                        if (dist < thresholdDist)
                            validAnchor = false;
                        break;
                }
                if (!validAnchor){
                    debl("failed");
                    break;
                }
                debl("");
            }
            
            if (validAnchor){
                deb("[+] New anchor: "); cand.first.print(inputData->vocab,false); debl(" rho="<<cand.second);
                anchorsIndexes.push_back(i);
            }
        }

        vector<Ngram> anchors;
        for (auto &anchor : anchorsIndexes)
            anchors.push_back(candidatesAnchor[anchor].first);
        return anchors;
    }

    virtual void getTopics(vector<Ngram> anchors){
        debl("Found "<<anchors.size()<<" anchors:");
        for (auto &anchor : anchors){
            #ifdef DEBUG
            deb("\t"); anchor.print(inputData->vocab, false); debl(" [rho="<<rho(anchor)<<"]");
            #endif
            topics.push_back(generateTopicCached(anchor));
        }
    }

    public:
    
    Real delta, eps, p, thresholdDist;
    int candidateSize = -1;
    
    enum distanceFunctionAlias { dfTau, dfLinf, dfTaud, dfInnerprod };
    distanceFunctionAlias distanceFunction;
    string distanceFunctionStr;

    virtual void parseInputDataParams(){
        STABasedAlgo::parseInputDataParams();
        delta = getParamReal(PARAM_TAG_DELTA, 0);
        eps = getParamReal(PARAM_TAG_EPS, 0);
        p = getParamReal(PARAM_TAG_P, 0);
        thresholdDist = getParamReal(PARAM_TAG_THRESHOLD_DIST, -1);
        distanceFunctionStr = getParamString(PARAM_TAG_DISTANCE, PARAM_DISTANCE_TAUD);
        candidateSize = getParamInt(PARAM_TAG_CANDIDATE_SIZE, -1);

        if (distanceFunctionStr == PARAM_DISTANCE_TAU) distanceFunction = dfTau;
        else if (distanceFunctionStr == PARAM_DISTANCE_TAUD) distanceFunction = dfTaud;
        else if (distanceFunctionStr == PARAM_DISTANCE_LINF) distanceFunction = dfLinf;
        else if (distanceFunctionStr == PARAM_DISTANCE_INNERPROD) distanceFunction = dfInnerprod;
        else die("Unknown distance function");
    }

    Real rho(const Ngram &cand){
        Real rho = 1;
        auto ngramTmpV = cand.toMultiset();
        auto ngramVec = cand.toVec();
        for (auto &word : *ngramVec){
            if (ngramTmpV->count(word) != 1){ // Anchors shouldn't contains the same word more than once
                return 0;
            }
            ngramTmpV->insert(word);
            Ngram tmpNgram(ngramTmpV->begin(), ngramTmpV->end());
            ngramTmpV->erase(ngramTmpV->find(word));
            auto it = staNgrams[tmpNgram.size].find(tmpNgram);
            if (it != staNgrams[tmpNgram.size].end())
                rho *= it->second;
            else {
                return 0;
            }
        }
        delete ngramVec;
        delete ngramTmpV;
        rho /= inputData->K * pow(staNgrams[cand.size][cand], inputData->c+1);
        if (rho != 0.0) rho = min(rho, 1.0/rho);
        return rho;
    }
    virtual Real tau(const Ngram &cand1, const Ngram &cand2){
        assert(cand1.size == cand2.size);
        assert(cand1.size == 1);
        Ngram tmpNgram = Ngram::join(cand1, cand2);
        Real tau;
        auto it = staNgrams[tmpNgram.size].find(tmpNgram);
        if (it != staNgrams[tmpNgram.size].end())
            tau = it->second / (inputData->K * staNgrams[cand1.size][cand1] * staNgrams[cand2.size][cand2]);
        else
            tau = 0;
        return tau;
    }
    virtual Real tauD(const Ngram &cand1, const Ngram &cand2){
        assert(cand1.size == cand2.size);
        assert(cand1.size == 1);
        Ngram tmpNgram = Ngram::join(cand1, cand2);
        Real taud;
        auto it = ldaNgrams.at(tmpNgram.size).find(tmpNgram);
        if (it != ldaNgrams.at(tmpNgram.size).end())
            taud = it->second / (ldaNgrams[cand1.size][cand1] * ldaNgrams[cand2.size][cand2]);
        else
            taud = 0;
        return taud;
    }
    Real inducedTopicDistance(const Ngram &cand1, const Ngram &cand2, const function<Real(Topic,Topic)>& distanceMeasure){
        const Topic& topic1 = generateTopicCached(cand1);
        const Topic& topic2 = generateTopicCached(cand2);
        return distanceMeasure(topic1, topic2);
    }

    virtual void computeFromSTA() {
        vector<pair<Ngram, Real>> candidateAnchors;
        getCandidatesAnchor(candidateAnchors);
        if (candidateAnchors.size() < inputData->K)
            debl("Got "<<candidateAnchors.size()<<" anchors, K = "<<inputData->K<<". That's a problem.");
        auto anchors = getAnchors(candidateAnchors);
        getTopics(anchors);
    }
};