#ifndef __FINALIZE_H__
#define __FINALIZE_H__

#include "dcl.h"
#include "hls_stream.h"

void finalize(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    WT_TYPE PNA_graph_mlp_1_weights[PNA_GRAPH_MLP_1_OUT][PNA_EMB_DIM],
    WT_TYPE PNA_graph_mlp_1_bias[PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_mlp_2_weights[PNA_GRAPH_MLP_2_OUT][PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_mlp_2_bias[PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_mlp_3_weights[NUM_TASK][PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_mlp_3_bias[NUM_TASK],
    WT_TYPE graph_pred_PNA_graph_mlp_3_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_mlp_3_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
);

#endif