#ifndef __LOAD_INPUTS_H__
#define __LOAD_INPUTS_H__

#include "dcl.h"
#include "hls_stream.h"
#include "hls_math.h"

void load_weights(
    //WT_TYPE GCN_convs_GIN_node_mlp_1_weight_in[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM],
    WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_in[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT],
    //WT_TYPE GIN_node_mlp_2_weight_in[NUM_LAYERS][EMB_DIM][DGN_LIN_GIN_MLP_1_OUT],
    WT_TYPE layers_posttrans_fully_connected_0_linear_weight_in[4][EMB_DIM][2 * EMB_DIM],
    WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE PNA_node_conv_weight_in[NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM],
    //WT_TYPE edge_embedding_weight_in[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    WT_TYPE bn_weight_PNA_graph_DGN_MLP_1_weight_in[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE bn_bias_PNA_graph_DGN_MLP_1_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE bn_mean_PNA_graph_DGN_MLP_2_weight_in[DGN_MLP_PNA_GRAPH_MLP_2_OUT][EMB_DIM],
    WT_TYPE bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_weight_in[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_bias_in[NUM_TASK],
    WT_TYPE avg_deg_in
);

void load_graph(
    edge_t* edge_list_in,
    //edge_attr_t* edge_attr_in,
    node_eigen_t* node_eigen_in,
    int num_of_nodes,
    int num_of_edges
);

void load_input_node_embeddings(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    node_feature_t* node_feature,
    WT_TYPE node_embedding_h_atom_embedding_list_weight_in[ND_FEATURE][ND_FEATURE_TOTAL][EMB_DIM],
    std::array<FM_TYPE, NUM_AGGRS> messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int num_of_nodes
);

void reset_messages(
    std::array<FM_TYPE, NUM_AGGRS> messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int num_of_nodes
);

#endif