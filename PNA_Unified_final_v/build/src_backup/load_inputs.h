#ifndef __LOAD_INPUTS_H__
#define __LOAD_INPUTS_H__

#include "dcl.h"
#include "hls_stream.h"
#include "hls_math.h"

void load_weights(
    WT_TYPE GCN_convs_GIN_node_mlp_1_weight_in[NUM_LAYERS][GIN_MLP_1_OUT][EMB_DIM],
    WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_in[NUM_LAYERS][GIN_MLP_1_OUT],
    WT_TYPE GIN_node_mlp_2_weights_in[NUM_LAYERS][EMB_DIM][GIN_MLP_1_OUT],
    WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE PNA_node_conv_weight_in[PNA_NUM_LAYERS][PNA_EMB_DIM][NUM_SCALERS][NUM_AGGRS][PNA_EMB_DIM],
    WT_TYPE edge_embedding_weight_in[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    WT_TYPE bn_weight_PNA_graph_mlp_1_weight_in[PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE bn_bias_PNA_graph_mlp_1_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE bn_mean_PNA_graph_mlp_2_weight_in[PNA_GRAPH_MLP_2_OUT][EMB_DIM],
    WT_TYPE bn_sqrt_var_PNA_graph_mlp_2_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_mlp_3_weight_in[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_mlp_3_bias_in[NUM_TASK],
    WT_TYPE avg_deg_in
);

void load_graph(
    edge_t* edge_list_in,
    edge_attr_t* edge_attr_in,
    int num_of_nodes,
    int num_of_edges
);

void load_input_node_embeddings(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    node_feature_t* node_feature,
    WT_TYPE node_embedding_weight[ND_FEATURE_TOTAL][EMB_DIM],
    std::array<FM_TYPE, NUM_AGGRS> messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int num_of_nodes
);

void reset_messages(
    std::array<FM_TYPE, NUM_AGGRS> messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int num_of_nodes
);

#endif