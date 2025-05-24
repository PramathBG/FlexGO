#include "dcl.h"

Instruction instruction; //instruction to decide which GNN has to be inferred

// #region Global Variables
// #Sub-region - Layer Weights of all models
std::array<std::array<WT_TYPE, NUM_AGGRS>, NUM_SCALERS> PNA_node_conv_weights[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM];

// #Sub-region - Layer Bias of all models and Convolution root embedding weights for GCN
WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT];

// #Sub-region - GIN Epsilon values and PNA Average Degree
WT_TYPE GIN_node_mlp_eps_PNA_avg_deg[NUM_LAYERS];

int degree_table[MAX_NODE];
int degree_tables[EDGE_PARALLEL][MAX_NODE][2];
int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
int num_of_edges_per_pe[EDGE_PARALLEL];
FM_TYPE DGN_abssums_PNA_log_degrees[MAX_NODE];

// #Sub-region - Message Buffers
std::array<FM_TYPE, NUM_AGGRS> messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM];
std::array<FM_TYPE, NUM_AGGRS> messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM];

FM_TYPE h_node[MAX_NODE][EMB_DIM];

// #Sub-region - PNA Graph MLP Parameters
WT_TYPE PNA_graph_DGN_MLP_1_weights[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM];
WT_TYPE PNA_graph_DGN_MLP_1_bias[DGN_MLP_PNA_GRAPH_MLP_1_OUT];
WT_TYPE PNA_graph_DGN_MLP_2_weights[DGN_MLP_PNA_GRAPH_MLP_2_OUT][DGN_MLP_PNA_GRAPH_MLP_1_OUT];
WT_TYPE PNA_graph_DGN_MLP_2_bias[DGN_MLP_PNA_GRAPH_MLP_2_OUT];
WT_TYPE PNA_graph_DGN_MLP_3_weights[NUM_TASK][DGN_MLP_PNA_GRAPH_MLP_2_OUT];
WT_TYPE PNA_graph_DGN_MLP_3_bias[NUM_TASK];
