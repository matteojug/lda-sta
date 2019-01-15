#define EPS 1e-12 // Minimum value to be considered as non zero
#define RANDOM_SEED 42 // Random seed if not specified

// #define DEBUG
// #define DEBUG_TIME

// algoId values
#define ALGO_STA_DUMP -1
#define ALGO_GREEDY 100
#define ALGO_LAZY_GREEDY 106
#define ALGO_CLUSTER 105
#define ALGO_LAZY_CLUSTER 107

// Various options tags
#define PARAM_TAG_DELTA "delta"
#define PARAM_TAG_EPS "eps"
#define PARAM_TAG_P "p"
#define PARAM_TAG_THRESHOLD_DIST "threshold_dist"
#define PARAM_TAG_NGRAM_MODE_SAMPLE_COUNT "ngram_sample_count"
#define PARAM_TAG_CANDIDATE_SIZE "candidate_size"
#define PARAM_TAG_RANDOM_SEED "random_seed"
#define PARAM_TAG_FORCE_GENERIC "force_generic"
#define PARAM_TAG_STADUMP_PATH "stadump_path"

// How to pick ngrams from the doc to estimade LDA distribution
#define PARAM_TAG_NGRAM_MODE "ngram"
#define PARAM_NGRAM_MODE_PREFIX "prefix"
#define PARAM_NGRAM_MODE_ALL "all"
#define PARAM_NGRAM_MODE_SAMPLE "sample"

// Define the distance measure
#define PARAM_TAG_DISTANCE "dist"
#define PARAM_DISTANCE_TAU "tau"
#define PARAM_DISTANCE_LINF "linf"
#define PARAM_DISTANCE_TAUD "taud"
#define PARAM_DISTANCE_INNERPROD "innerprod"

// Linkage policy for agglomerative clustering
#define PARAM_TAG_AC_LINKAGE "linkage"
#define PARAM_AC_LINKAGE_SINGLE "single"
#define PARAM_AC_LINKAGE_COMPLETE "complete"
#define PARAM_AC_LINKAGE_AVERAGE "average"