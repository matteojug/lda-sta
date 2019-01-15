#pragma once
#include "../definitions.hpp"
#include "../utils.hpp"

// Algo is mutable
class Algo {
    public:
    vector<Topic> topics;
    InputData *inputData;
    
    void setInputData(InputData *inputData){
        this->inputData = inputData;
        parseInputDataParams();
    }
    virtual void parseInputDataParams(){}
    virtual void compute(){}

    void outputTopics(){
        debl("Found "<<topics.size()<<" topics");
        for (auto &topic : topics){
            for (auto &v : topic)
                cout<<v<<" ";
            cout<<endl;
        }
    }

    void printTopics(){
        #ifdef DEBUG
        deb("Topics ["<<topics.size()<<"]"<<endl);
        for (int i = 0; i < topics.size(); i++){
            auto topK = getTopKIdx(topics[i], 10);
            deb("["<<i<<"]");
            for (auto &vocabId : topK)
                deb(" "<<inputData->vocab[vocabId]<<"("<<vocabId<<")");
            deb(endl);
        }
        #endif
    }

    bool hasParam(const string &tag){
        return inputData->params.count(tag);
    }
    void requireParam(const string &tag){
        if (!hasParam(tag)) die("Required param "<<tag<<" not found in input data");
    }
    string getParamString(const string &tag){
        requireParam(tag);
        return inputData->params[tag];
    }
    inline string getParamString(const string &tag, const string &def){
        return !hasParam(tag) ? def : getParamString(tag);
    }
    Real getParamReal(const string &tag){
        requireParam(tag);
        return stod(inputData->params[tag]);
    }
    inline Real getParamReal(const string &tag, const Real def){
        return !hasParam(tag) ? def : getParamReal(tag);
    }
    int getParamInt(const string &tag){
        requireParam(tag);
        return stoi(inputData->params[tag]);
    }
    inline int getParamInt(const string &tag, const int &def){
        return !hasParam(tag) ? def : getParamInt(tag);
    }

    virtual ~Algo(){ }
};