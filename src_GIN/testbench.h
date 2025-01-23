#ifndef __TESTBENCH_H__
#define __TESTBENCH_H__

#include "../src_GIN/dcl.h"
#include "../src_GIN/util.h"

constexpr int NUM_GRAPHS = 1; // 4113;

extern WT_TYPE GCN_convs_GIN_node_mlp_1_weight_fixed[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM];
extern WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT];
extern WT_TYPE GIN_node_mlp_2_weight_fixed[NUM_LAYERS][EMB_DIM][DGN_LIN_GIN_MLP_1_OUT];
extern WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[NUM_LAYERS][EMB_DIM];
extern WT_TYPE PNA_node_conv_weight_fixed[DGN_PNA_NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM];
extern WT_TYPE layers_posttrans_fully_connected_0_linear_weight_fixed [4][EMB_DIM][2 * EMB_DIM];
extern WT_TYPE bn_weight_PNA_graph_DGN_MLP_1_weight_fixed[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM];
extern WT_TYPE bn_bias_PNA_graph_DGN_MLP_1_bias_fixed[NUM_LAYERS][EMB_DIM];
extern WT_TYPE bn_mean_PNA_graph_DGN_MLP_2_weight_fixed[DGN_MLP_PNA_GRAPH_MLP_2_OUT][EMB_DIM];
extern WT_TYPE bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed[NUM_LAYERS][EMB_DIM];
extern WT_TYPE node_embedding_h_atom_embedding_list_weight_fixed[ND_FEATURE][ND_FEATURE_TOTAL][EMB_DIM];
extern WT_TYPE edge_embedding_weight_fixed[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
extern WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[NUM_TASK][EMB_DIM];
extern WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[NUM_TASK];
extern WT_TYPE GIN_node_mlp_eps_PNA_avg_deg_fixed[NUM_LAYERS];
extern WT_TYPE node_embedding_h_atom_embedding_list_weight_fixed_DGN[ND_FEATURE][ND_FEATURE_TOTAL][EMB_DIM];

void read_instruction();
void load_weights(int GNN_instruction);
void fetch_one_graph(
    int g,
    char* graph_name,
    node_feature_t* node_feature,
    edge_t* edge_list,
    edge_attr_t* edge_attr,
    node_eigen_t* node_eigen,
    int num_of_nodes,
    int num_of_edges
);

#endif
