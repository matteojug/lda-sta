#include <Python.h>
#include "src/anchor.cpp"

using namespace std;

#define _fail(e) { cerr<<__FILE__":"<<__LINE__<<" @ "<<e<<endl; \
                        return NULL; }

extern "C" {

static PyObject* anchor(PyObject *self, PyObject *args){
    PyObject *corpus, *vocab = NULL, *algParams = NULL;
    int K, c, algId = ALGO_LAZY_CLUSTER;
    double alpha;

    if (!PyArg_ParseTuple(args, "Oidi|iOO", &corpus, &K, &alpha, &c, &algId, &algParams, &vocab))
        return NULL;

    InputData *inputData = new InputData();
    inputData->algId = algId;
    inputData->K = K;
    inputData->alpha = alpha;
    inputData->c = c;
    inputData->maxDocLen = 0;

    int len;
    PyObject *dictKey, *dictValue;
    Py_ssize_t dictPos;

    len = PySequence_Size(corpus);
    inputData->corpus.resize(len);
    inputData->vocabSize = 0;
    for (int i = 0; i < len; i++) {
        auto doc = PySequence_GetItem(corpus, i);
        dictPos = 0;
        int docLen = 0;
        while (PyDict_Next(doc, &dictPos, &dictKey, &dictValue)){
            int tokenId = PyInt_AsLong(dictKey),
                    tokenCount = PyInt_AsLong(dictValue);
            if (PyErr_Occurred()) _fail("Wrong data type in corpus");
            inputData->corpus[i].push_back(make_pair(tokenId, tokenCount));
            inputData->vocabSize = max(inputData->vocabSize, tokenId+1);
            docLen += tokenCount;
        }
        inputData->maxDocLen = max(inputData->maxDocLen, docLen);
        sort(inputData->corpus[i].begin(), inputData->corpus[i].end());
    }
    
    if (vocab){
        len = PySequence_Size(vocab);
        for (int i = 0; i < len; i++) {
            auto item = PySequence_GetItem(vocab, i);
            char *s = PyString_AsString(item);
            if (PyErr_Occurred()) _fail("Wrong data type in vocab"); 
            inputData->vocab.push_back(s);
        }
        inputData->vocabSize = inputData->vocab.size();
    }

    if (algParams){
        dictPos = 0;
        while (PyDict_Next(algParams, &dictPos, &dictKey, &dictValue)){
            string sk = PyString_AsString(dictKey);
            if (PyErr_Occurred()) _fail("Wrong data type in algParams"); 
            string sv = PyString_AsString(dictValue);
            if (PyErr_Occurred()) _fail("Wrong data type in algParams"); 
            inputData->params[sk] = sv;
        }
    }

    Algo *algo = anchor(inputData);

    PyObject *PTopicList = PyList_New(algo->topics.size());
    for (int i = 0; i < algo->topics.size(); i++){
        PyObject *PTopic = PyList_New(algo->topics[i].size());
        for (int j = 0; j < algo->topics[i].size(); j++)
            PyList_SET_ITEM(PTopic, j, PyFloat_FromDouble(algo->topics[i][j]));
        PyList_SET_ITEM(PTopicList, i, PTopic);
    }
    
    delete inputData;
    delete algo;

    return PTopicList;
}

}

static PyMethodDef LdaStaMethods[] = {
    {"anchor",  anchor, METH_VARARGS,
"Reconstruct the topics from a corpus\n"
"\n"
"Args:\n"
"    corpus: The corpus as a list of documents, each being a dict representing the bag of words of the document as integer key/value {wordId:count}.\n"
"    K: The number of topics to reconstruct.\n"
"    alpha: The dirichlet parameter (usually 1/K).\n"
"    c: The anchor cardinality (with c=1 anchors are single words, c=2 pairs, ecc..).\n"
"    algId: The algorithm to be used (Default: ALGO_LAZY_CLUSTER).\n"
"    algParams: A dict containing the parameters for the algorithm as string key/value (Default: {}).\n"
"    vocab: The vocabulary as list of words, useless in non debug mode (Default: None).\n"
"\n"
"Returns:\n"
"    A list of topics, each being a probability distribution as a list of real values (one for each word of the vocabulary)\n"
"\n"
"Notes:\n"
"The complexity is exponential in c, using values higher than 1 is not recommended unless with very small vocabulary size; higher values require also to specify a distance function to be used.\n"
"The algorithms available are:\n"
"    ALGO_STA_DUMP: dump the STA (Single Topic Allocation) probabilities;\n"
"    ALGO_GREEDY: greedy algorithm that picks the best candidates that don't collide with already picked ones;\n"
"    ALGO_CLUSTER: perform agglomerive clustering on the candidates;\n"
"    ALGO_LAZY_GREEDY/ALGO_LAZY_CLUSTER: lazy implementations where the LDA/STA probabilities are computed on demands (require c=1);\n"
"The recommended ones are ALGO_LAZY_CLUSTER and, if c>1, ALGO_CLUSTER.\n"
"As algParams, the following parameters can be used (the values provided must be casted to string):\n"
"    delta, eps: required, refer to the paper;\n"
"    p: required if using the greedy algorithms, refer to the paper;\n"
"    ngram: (all|sample|prefix, default: all) specify what ngrams are used to compute the LDA n-grams distribution;\n"
"    dist: (taud|tau|linf|innerprod, default: taud) the distance function, linf and innerprod refers to the resp. metrics applied to the topics induced by the anchors;\n"
"    linkage: (average|single|complete|innerprod, default: average) the linkage policy used by the clustering algorithm;\n"
"    candidate_size: force the size of the candidate anchors pool (ignoring the rho cutoff);\n"
"    threshold_dist: required if using the greedy algorithm and a distance function other than tau, overwrite the threshold that define two anchors colliding;\n"
"    ngram_sample_count: if ngram=sample, the numberd of samples to take from each document;\n"
"    random_seed: if ngram=sample, random seed to use when required;\n"
"    stadump_path: if algo=STA_DUMP, path where to save the STA distribution;\n"
"\n"
    },
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initlda_sta(void){
    PyObject* module = Py_InitModule("lda_sta", LdaStaMethods);
    PyObject_SetAttrString(module, "ALGO_STA_DUMP", PyInt_FromLong(ALGO_STA_DUMP));
    PyObject_SetAttrString(module, "ALGO_GREEDY", PyInt_FromLong(ALGO_GREEDY));
    PyObject_SetAttrString(module, "ALGO_LAZY_GREEDY", PyInt_FromLong(ALGO_LAZY_GREEDY));
    PyObject_SetAttrString(module, "ALGO_CLUSTER", PyInt_FromLong(ALGO_CLUSTER));
    PyObject_SetAttrString(module, "ALGO_LAZY_CLUSTER", PyInt_FromLong(ALGO_LAZY_CLUSTER));
}
