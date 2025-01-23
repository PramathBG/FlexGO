#ifndef __FINALIZE_H__
#define __FINALIZE_H__

#include "dcl.h"
#include "hls_stream.h"

void finalize(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    // WT_TYPE PNA_graph_DGN_MLP_1_weights[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    // WT_TYPE PNA_graph_DGN_MLP_1_bias[DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    // WT_TYPE PNA_graph_DGN_MLP_2_weights[DGN_MLP_PNA_GRAPH_MLP_2_OUT][DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    // WT_TYPE PNA_graph_DGN_MLP_2_bias[DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    // WT_TYPE PNA_graph_DGN_MLP_3_weights[NUM_TASK][DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    // WT_TYPE PNA_graph_DGN_MLP_3_bias[NUM_TASK],
    WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
);

#endif