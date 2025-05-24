#include "dcl.h"

Instruction instruction; //instruction to decide which GNN has to be inferred

// #region Global Variables
// #Sub-region - Layer Weights of all models
WT_TYPE GCN_convs_GIN_node_mlp_1_weights[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM];
std::array<std::array<WT_TYPE, NUM_AGGRS>, NUM_SCALERS> PNA_node_conv_weights[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM];

// #Sub-region - Layer Bias of all models and Convolution root embedding weights for GCN
WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT];
WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias[NUM_LAYERS][EMB_DIM];
WT_TYPE convs_root_emb_weight_slice[EMB_DIM];

// #Sub-region - Batch Normalization Weights for GCN
WT_TYPE GCN_bn_weights[NUM_LAYERS - 1][EMB_DIM];
WT_TYPE GCN_bn_bias[NUM_LAYERS - 1][EMB_DIM];
WT_TYPE GCN_bn_mean[NUM_LAYERS - 1][EMB_DIM];
WT_TYPE GCN_bn_sqrt_var[NUM_LAYERS - 1][EMB_DIM];
WT_TYPE GCN_bn_weight_final[EMB_DIM];
WT_TYPE GCN_bn_bias_final[EMB_DIM];
WT_TYPE GCN_bn_mean_final[EMB_DIM];
WT_TYPE GCN_bn_sqrt_var_final[EMB_DIM];

// #Sub-region - Edge Embedding Weights for GCN and GIN
WT_TYPE edge_embedding_weights[EDGE_PARALLEL][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];

// #Sub-region - Graph Level Task Weights and Biases of GCN and GIN, Graph MLP3 Parameters for PNA
WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM];
WT_TYPE graph_pred_bias[NUM_TASK];

int degree_table[MAX_NODE];
int degree_table_finalize[MAX_NODE];
int degree_tables[EDGE_PARALLEL][MAX_NODE][2];
int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
edge_attr_t edge_attrs[EDGE_PARALLEL][MAX_EDGE];
int num_of_edges_per_pe[EDGE_PARALLEL];
WT_TYPE DGN_eig_w_GCN_norms[EDGE_PARALLEL][MAX_EDGE];

// #Sub-region - Message Buffers
std::array<FM_TYPE, NUM_AGGRS> messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM];
std::array<FM_TYPE, NUM_AGGRS> messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM];

FM_TYPE h_node[MAX_NODE][EMB_DIM];
