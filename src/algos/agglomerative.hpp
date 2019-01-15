#pragma once
#include "anchor.hpp"
#include "anchor_fast.hpp"

typedef set<int> Cluster;
namespace Linkage {
    typedef function<Real(Real distA, Real distB, int sizeA, int sizeB)> Policy;

    Policy average = [](Real distA, Real distB, int sizeA, int sizeB){
        return (distA * sizeA + distB * sizeB) / (sizeA + sizeB);
    };
    Policy complete = [](Real distA, Real distB, int sizeA, int sizeB){
        return max(distA, distB);
    };
    Policy single = [](Real distA, Real distB, int sizeA, int sizeB){
        return min(distA, distB);
    };
}

vector<Cluster> agglomerativeCluster(int k, unordered_map<int, unordered_map<int, Real>> &distanceMatrix, Linkage::Policy &linkagePolicy){
    priority_queue<pair<Real, pair<int,int>>> pq;
    unordered_map<int, Cluster> clusters;
    int nextClusterId = 0;
    for (auto &k : distanceMatrix){
        clusters[k.first] = {k.first};
        for (auto &v : k.second)
            pq.push(make_pair(-v.second, make_pair(k.first, v.first)));
        nextClusterId = max(nextClusterId, k.first);
    }
    nextClusterId++;
    while (clusters.size() > k && !pq.empty()){
        auto x = pq.top(); pq.pop();
        int clusterA = x.second.first, clusterB = x.second.second;
        if (!(clusters.count(clusterA) && clusters.count(clusterB))) continue;

        distanceMatrix[nextClusterId] = {};
        for (auto &otherKV : clusters){
            int other = otherKV.first;
            if (other == clusterA || other == clusterB) continue;
            auto dA = (distanceMatrix[clusterA].count(other) ? distanceMatrix[clusterA][other] : distanceMatrix[other][clusterA]);
            auto dB = (distanceMatrix[clusterB].count(other) ? distanceMatrix[clusterB][other] : distanceMatrix[other][clusterB]);
            distanceMatrix[nextClusterId][other] = linkagePolicy(dA, dB, clusters[clusterA].size(), clusters[clusterB].size());
            pq.push(make_pair(-distanceMatrix[nextClusterId][other], make_pair(nextClusterId, other)));
        }
        distanceMatrix.erase(clusterA);
        distanceMatrix.erase(clusterB);

        clusters[nextClusterId] = {};
        clusters[nextClusterId].insert(clusters[clusterA].begin(), clusters[clusterA].end());
        clusters[nextClusterId].insert(clusters[clusterB].begin(), clusters[clusterB].end());
        clusters.erase(clusterA);
        clusters.erase(clusterB);
        nextClusterId++;
    }
    vector<Cluster> ret;
    for (auto &k : clusters)
        ret.push_back(k.second);
    return ret;
} 

class AnchorAgglomerativeCluster : virtual public Anchor {
    
    enum linkagePolicyAlias { lpAverage, lpSingle, lpComplete };
    linkagePolicyAlias linkagePolicy;
    string linkagePolicyStr;

    public:

    virtual void parseInputDataParams(){
        Anchor::parseInputDataParams();
        linkagePolicyStr = getParamString(PARAM_TAG_AC_LINKAGE, PARAM_AC_LINKAGE_AVERAGE);

        if (linkagePolicyStr == PARAM_AC_LINKAGE_AVERAGE) linkagePolicy = lpAverage;
        else if (linkagePolicyStr == PARAM_AC_LINKAGE_SINGLE) linkagePolicy = lpSingle;
        else if (linkagePolicyStr == PARAM_AC_LINKAGE_COMPLETE) linkagePolicy = lpComplete;
        else die("Unknown linkage policy");
    }
    
    virtual vector<Ngram> getAnchors(vector<pair<Ngram, Real>> &candidatesAnchor){
        debl("Got "<<candidatesAnchor.size()<<" candidates");

        function<Real(int,int)> getDist;
        switch (distanceFunction){
            case dfTau:
                getDist = [&](int cand1, int cand2){
                    return (cand1 == cand2 ? 0 : -tau(candidatesAnchor[cand1].first, candidatesAnchor[cand2].first));
                };
                break;
            case dfTaud:
                getDist = [&](int cand1, int cand2){
                    return (cand1 == cand2 ? 0 : -tauD(candidatesAnchor[cand1].first, candidatesAnchor[cand2].first));
                };
                break;
            case dfLinf:
                getDist = [&](int cand1, int cand2){
                    return inducedTopicDistance(candidatesAnchor[cand1].first, candidatesAnchor[cand2].first, distanceLinf<Real>);
                };
                break;
            default:
                die("Unknown distanceFunction: "<<distanceFunctionStr);
        }

        unordered_map<int, unordered_map<int, Real>> distanceMatrix;
        for (int i = 0; i < candidatesAnchor.size(); i++){
            distanceMatrix[i] = {};
            for (int j = i+1; j < candidatesAnchor.size(); j++){
                distanceMatrix[i][j] = getDist(i,j);
            }
        }

        Linkage::Policy linkagePolicyFn;
        switch (linkagePolicy){
            case lpAverage:
                linkagePolicyFn = Linkage::average;
                break;
            case lpComplete:
                linkagePolicyFn = Linkage::complete;
                break;
            case lpSingle:
                linkagePolicyFn = Linkage::single;
                break;
        }
        auto clusters = agglomerativeCluster(inputData->K, distanceMatrix, linkagePolicyFn);
        vector<Ngram> anchors;
        for (auto &cluster : clusters){
            int representative = -1;
            for (auto &e : cluster)
                if (representative == -1 || candidatesAnchor[e].second > candidatesAnchor[representative].second)
                    representative = e;
            anchors.push_back(candidatesAnchor[representative].first);
        }
        return anchors;
    }
};


class AnchorFastAgglomerativeCluster : public AnchorAgglomerativeCluster, AnchorFast {
    public:
    virtual vector<Ngram> getAnchors(vector<pair<Ngram, Real>> &candidatesAnchor){
        // To avoid the requireTau calls for each pair
        vector<int> anchorOrder(candidatesAnchor.size());
        for (int i = 0; i < anchorOrder.size(); i++) anchorOrder[i] = i;
        sort(anchorOrder.begin(), anchorOrder.end(), [&](int a, int b){ return postingList[a].size() < postingList[b].size(); });
        for (int i = 0; i < anchorOrder.size(); i++){
            for (int j = i+1; j < anchorOrder.size(); j++){
                Ngram key = Ngram::join(candidatesAnchor[anchorOrder[i]].first, candidatesAnchor[anchorOrder[j]].first);
                if (staNgrams[key.size].count(key)) continue;
                Real cnt = 0;
                for (auto &x1 : postingList[anchorOrder[i]]){
                    if (!postingList[anchorOrder[j]].count(x1.first)) continue;
                    cnt += x1.second * postingList[anchorOrder[j]][x1.first];
                }
                key.updateMultiplicity();
                ldaNgrams[key.size][key] = cnt / ngramCounts[key.size];
                lda2staSingleAdd(staNgrams, key, ldaNgrams[key.size][key], inputData->K, inputData->alpha, true, hasParam(PARAM_TAG_SKIP_REDUCTION));                
            }
        }
        return AnchorAgglomerativeCluster::getAnchors(candidatesAnchor);
    }
};