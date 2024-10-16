#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "hls_stream.h"
#include "dcl.h"

void compute_CONV_layer(
    int layer_num,
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    std::array<FM_TYPE, NUM_AGGRS> next_message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    node_feature_t* node_feature_in,
    WT_TYPE node_embedding_h_atom_embedding_list_weight_in[9][ND_FEATURE_TOTAL][EMB_DIM],
    FM_TYPE* result,
    int num_of_nodes
);

void h_node_passthrough(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    int num_of_nodes
);

#endif