#pragma once

#include "stdc++.h"
using namespace std;

#include "consts.hpp"

typedef double Real;
typedef unsigned long long ULL;

typedef vector<pair<int,int>> Bow; // word, count
typedef vector<Bow> Corpus;
typedef vector<char*> Vocab;
typedef vector<Real> Topic;

#define die(x) do { cerr<<x<<endl; exit(1); } while (0)

#ifdef DEBUG
    #define deb(x) do { cerr<<x; } while (0)
#else
    #define deb(x)
#endif
#ifdef DEBUG_TIME
    #define debTime(x) do { cerr<<x; } while (0)
#else
    #define debTime(x)
#endif
#define debl(x) deb(x<<endl)
#define debVar(x) deb(__FILE__":"<<__LINE__<<" @ "#x" = "<<x<<endl)
#define deblTime(x) debTime(x<<endl)

#define gammaCacheMaxSize 1000
unordered_map<Real, Real> gammaCache;
inline Real _gamma(Real x){ // Python was messing things up
    if (gammaCache.count(x) == 0){
        if (gammaCache.size() > gammaCacheMaxSize) return tgamma(x);
        return gammaCache[x] = tgamma(x);
    }
    return gammaCache[x];
}
inline Real gamma(Real x){ return _gamma(x); }
inline Real gamma(int x){ return _gamma((Real)x); }

unordered_map<int, int> factorialCache;
inline int factorialNaive(int x){
    int res = 1;
    while (x) res *= x--;
    return res;
}
inline int factorial(int x){
    assert(x < 15);
    if (factorialCache.count(x) == 0){
        return factorialCache[x] = factorialNaive(x);
    }
    return factorialCache[x];
}
// #define factorial(X) gamma((X)+1)

struct InputData {
    int algId;
    int K, c, maxDocLen, vocabSize;
    Real alpha;
    Corpus corpus;
    Vocab vocab;
    unordered_map<string, string> params;

    void printInfo(){
        debl(" == InputData == ");
        debl("AlgId:"<<algId);
        debl("Params ("<<params.size()<<"):");
        for (auto kv : params){
            debl("\t"<<kv.first<<" = "<<kv.second);
        }
        debl("K: "<<K<<", c: "<<c<<", alpha: "<<alpha);
        debl("CorpusSize: "<<corpus.size());
        debl("VocabSize: "<<vocab.size());
        debl("MaxDocLen: "<<maxDocLen);
        debl("VocabSize(exp): "<<vocabSize);
    }
};

thread_local int MAX_VOCAB = 1000; // Placeholder value

// A Ngram is a tuple backed by a buffer (unsigned long long).
// Can hold at most log_{MAX_VOCAB}(ULL_max) ~ 6-grams, TODO vec fallback for longer
struct Ngram {
    int size; // The ngram cardinality
    ULL buffer; // condensed ngram
    int multiplicity = 0; // how many ngrams this represents
    Ngram(){
        clear();
    }
    // Fill the Ngram using the specified iterator
    template<class Iter>
    Ngram(Iter begin, Iter end){
        clear();
        for (auto it = begin; it != end; it++) add(*it);
    }

	Ngram(const Ngram& other) : size(other.size), buffer(other.buffer), multiplicity(other.multiplicity){ }
	Ngram(const Ngram&& other) : size(other.size), buffer(other.buffer), multiplicity(other.multiplicity){ }
	Ngram& operator=(const Ngram& other){
        size = other.size; buffer = other.buffer; multiplicity = other.multiplicity;
        return *this;
    }
	Ngram& operator=(const Ngram&& other){
        size = other.size; buffer = other.buffer; multiplicity = other.multiplicity;
        return *this;
    }

    inline void clear(){
        buffer = 0;
        size = 0;
    }
    // Append an item
    inline void add(int gram){
        buffer *= MAX_VOCAB, buffer += gram;
        size++;
    }
    // Sort the ngram
    void normalize(){
        auto vec = toVec();
        sort(vec->begin(), vec->end());
        clear();
        for (auto &x : *vec) add(x);
        delete vec;
    }
    
    inline int computeMultiplicity() const {
        auto tmp = buffer;
        int multiplicity = factorial(size), sameCount = 1;
        for (int i = 0, x, lastX; i < size; i++){
            x = tmp % MAX_VOCAB, tmp /= MAX_VOCAB;
            if (i > 0){
                if (x == lastX)
                    sameCount++;
                else
                    multiplicity /= factorial(sameCount), sameCount = 1;
            }
            lastX = x;
        }
        multiplicity /= factorial(sameCount);
        return multiplicity;
    }
    inline int updateMultiplicity(){
        return multiplicity = computeMultiplicity();
    }
    
    inline void forEachReversed(const function<void(int)> &f) const {
        auto tmp = buffer;
        for (int i = 0; i < size; i++)
            f(tmp % MAX_VOCAB), tmp /= MAX_VOCAB;
    }
    inline vector<int>* toVec() const {
        vector<int> *ret = new vector<int>();
        forEachReversed([&](int x){ ret->push_back(x); });
        for (int i = 0; i < size/2; i++) swap(ret->at(i), ret->at(size-i-1));
        return ret;
    }
    inline multiset<int>* toMultiset() const {
        multiset<int> *ret = new multiset<int>();
        forEachReversed([&](int x){ ret->insert(x); });
        return ret;
    }

    bool operator<(const Ngram &other) const {
        return size < other.size || (size == other.size && buffer < other.buffer);
    }
    bool operator==(const Ngram &other) const {
        return size == other.size && buffer == other.buffer;
    }
    
    static inline Ngram join(const Ngram &a, const Ngram &b){
        auto ngramMSet = a.toMultiset();
        b.forEachReversed([&](int word){
            ngramMSet->insert(word);
        });
        Ngram ret(ngramMSet->begin(), ngramMSet->end());
        delete ngramMSet;
        return ret;
    }

    void print(bool withEndl = true) const {
        #ifdef DEBUG
        auto v = toVec();
        deb(size<<"-gram(");
        bool first = true;
        for (auto &x : *v){
            if (!first)
                deb(",");
            deb(x);
            first = false;
        }
        deb(")");
        if (withEndl) deb(endl);
        delete v;
        #endif
    }
    void print(Vocab &vocab, bool withEndl = true) const {
        #ifdef DEBUG
        if (vocab.empty()) return print(withEndl);
        auto v = toVec();
        deb(size<<"-gram(");
        bool first = true;
        for (auto &x : *v){
            if (!first)
                deb(",");
            deb(vocab[x]<<"["<<x<<"]");
            first = false;
        }
        deb(")");
        if (withEndl) deb(endl);
        delete v;
        #endif
    }
};

namespace std {
    template <> struct hash<Ngram>{
        std::size_t operator()(const Ngram &ngram) const {
            return ngram.size * 31 + ngram.buffer;
        }
    };
    template <class T1, class T2> struct hash<pair<T1,T2>>{
        std::size_t operator () (const std::pair<T1,T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 * 31 + h2;  
        }
    };
}

typedef unordered_map<Ngram,Real> Ngrams; // Maps Ngrams to real values (count or probability); the value is the count/probability of *each* of the ngram.multiplicity-many that it represents
typedef vector<Ngrams> NgramsVec; // Keeps multiple Ngrams, partitioned by the cardinality. The zero dimension is left empty
