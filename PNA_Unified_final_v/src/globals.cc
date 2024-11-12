#include "dcl.h"

Instruction instruction; // instruction to decide which GNN has to be inferred

// #region Global Variables
// #Sub-region - Layer Weights of all models
WT_TYPE GCN_convs_GIN_node_mlp_1_weight[NUM_LAYERS][GIN_MLP_1_OUT][EMB_DIM];
WT_TYPE GIN_node_mlp_2_weights[NUM_LAYERS][EMB_DIM][GIN_MLP_1_OUT];
std::array<std::array<WT_TYPE, NUM_AGGRS>, NUM_SCALERS> PNA_node_conv_weights[NUM_LAYERS][GIN_MLP_1_OUT][EMB_DIM];

// #Sub-region - Layer Bias of all models and Convolution root embedding weights for GCN
WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias[NUM_LAYERS][GIN_MLP_1_OUT];
WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_bias[NUM_LAYERS][EMB_DIM];
WT_TYPE convs_root_emb_weight_slice[EMB_DIM];

// #Sub-region - Batch Normalization Weights for GCN and Graph MLP1 adnd MLP2 Parameters of PNA
WT_TYPE bn_weight_PNA_graph_mlp_1_weights[NUM_LAYERS - 1][EMB_DIM];
WT_TYPE bn_bias_PNA_graph_mlp_1_bias[NUM_LAYERS - 1][EMB_DIM];
WT_TYPE bn_mean_PNA_graph_mlp_2_weights[NUM_LAYERS - 1][EMB_DIM];
WT_TYPE bn_sqrt_var_PNA_graph_mlp_2_bias[NUM_LAYERS - 1][EMB_DIM];
WT_TYPE bn_weight_final[EMB_DIM];
WT_TYPE bn_bias_final[EMB_DIM];
WT_TYPE bn_mean_final[EMB_DIM];
WT_TYPE bn_sqrt_var_final[EMB_DIM];

// #Sub-region - GIN Epsilon values and PNA Average Degree
WT_TYPE GIN_node_mlp_eps_PNA_avg_deg[NUM_LAYERS];

// #Sub-region - Edge Embedding Weights for GCN and GIN
WT_TYPE edge_embedding_weights[EDGE_PARALLEL][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];

// #Sub-region - Graph Level Task Weights and Biases of GCN and GIN, Graph MLP3 Parameters for PNA
WT_TYPE graph_pred_PNA_graph_mlp_3_weights[NUM_TASK][EMB_DIM];
WT_TYPE graph_pred_PNA_graph_mlp_3_bias[NUM_TASK];


int degree_table[MAX_NODE];
int degree_table_finalize[MAX_NODE];
int degree_tables[EDGE_PARALLEL][MAX_NODE][2];
int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
WT_TYPE GCN_norms[EDGE_PARALLEL][MAX_EDGE];
edge_attr_t edge_attrs[EDGE_PARALLEL][MAX_EDGE];
int num_of_edges_per_pe[EDGE_PARALLEL];
FM_TYPE PNA_log_degrees[MAX_NODE];

std::array<FM_TYPE, NUM_AGGRS> messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
std::array<FM_TYPE, NUM_AGGRS> messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];

FM_TYPE h_node[MAX_NODE][EMB_DIM];

WT_TYPE PNA_graph_mlp_1_weights[PNA_GRAPH_MLP_1_OUT][PNA_EMB_DIM];
WT_TYPE PNA_graph_mlp_1_bias[PNA_GRAPH_MLP_1_OUT];
WT_TYPE PNA_graph_mlp_2_weights[PNA_GRAPH_MLP_2_OUT][PNA_GRAPH_MLP_1_OUT];
WT_TYPE PNA_graph_mlp_2_bias[PNA_GRAPH_MLP_2_OUT];
WT_TYPE PNA_graph_mlp_3_weights[NUM_TASK][PNA_GRAPH_MLP_2_OUT];
WT_TYPE PNA_graph_mlp_3_bias[NUM_TASK];


int max_EMB_DIM;
int max_iter;
int max_NUM_LAYERS;
