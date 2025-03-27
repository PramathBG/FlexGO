#ifndef __DCL_H__
#define __DCL_H__

//#include <gmp.h>
//#define __gmp_const const 

#include "util.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ap_fixed.h>
#include <array>

// #region model parameters common to all the Graph Neural Networks
constexpr int MAX_EDGE = 500;
constexpr int MAX_NODE = 500;
constexpr int ND_FEATURE = 9;
constexpr int ND_FEATURE_TOTAL = 174;
constexpr int EDGE_ATTR = 3;
constexpr int ED_FEATURE_PER_LAYER = 13;
constexpr int EMB_DIM = 32; //DGN_LOUT is the same as EMB_DIM
constexpr int NUM_LAYERS = 5;
constexpr int NUM_TASK = 1;
// #endregion

// #region model parameters - GIN and DGN specific parameters
constexpr int DGN_LIN_GIN_MLP_1_OUT = 64;
// #endregion

//#region model parameters - PNA and DGN specific parameters 
constexpr int DGN_PNA_NUM_LAYERS = 4;
constexpr int DGN_MLP_PNA_GRAPH_MLP_1_OUT = 16;
constexpr int DGN_MLP_PNA_GRAPH_MLP_2_OUT = 8;

typedef enum{
    AGGR_MEAN,
    AGGR_MIN,
    AGGR_MAX,
    AGGR_STD,
    NUM_AGGRS
} aggregator_t;

typedef enum{
    SCALER_NONE,
    SCALER_T,
    SCALER_SCALE,
    NUM_SCALERS
} scaler_t;
// #endregion

// #region Hardware Parameters
constexpr int LOAD_IN_EMB_PARALLEL = 2;
constexpr int NODE_PARALLEL = 2;
constexpr int SCATTER_PARALLEL = 4;
constexpr int APPLY_PARALLEL = 2;
constexpr int EDGE_PARALLEL = 2;
constexpr int MLP_PARALLEL = 2;
//#endregion

// #region Analysis Parameters
// actual min/avg/max from MolHIV dataset
// constexpr int ANALYSIS_NUM_GRAPHS = 4113;
// constexpr int ANALYSIS_MIN_NODES = 6;
// constexpr int ANALYSIS_AVG_NODES = 25;
// constexpr int ANALYSIS_MAX_NODES = 183;
// constexpr int ANALYSIS_MIN_EDGES = 12;
// constexpr int ANALYSIS_AVG_EDGES = 56;
// constexpr int ANALYSIS_MAX_EDGES = 378;
// #endregion

//#region Analysis Parameters
constexpr int ANALYSIS_NUM_GRAPHS = 4113;
constexpr int ANALYSIS_MIN_NODES = 19;
constexpr int ANALYSIS_MAX_NODES = 19;
constexpr int ANALYSIS_AVG_NODES = 19;
constexpr int ANALYSIS_MIN_EDGES = 40;
constexpr int ANALYSIS_MAX_EDGES = 40;
constexpr int ANALYSIS_AVG_EDGES = 40;
// #endregion

// #region Data Types

typedef ap_fixed<16, 6> FM_TYPE;
typedef ap_fixed<16, 6> WT_TYPE;
typedef std::array<FM_TYPE, APPLY_PARALLEL> ne_out_t;
typedef std::array<FM_TYPE, SCATTER_PARALLEL> mp_in_t;
typedef std::array<FM_TYPE, SCATTER_PARALLEL> me_out_t;
typedef std::array<FM_TYPE, APPLY_PARALLEL> mlp_xfer_t;
typedef std::array<int, ND_FEATURE> node_feature_t;
typedef std::array<int, EDGE_ATTR> edge_attr_t;
typedef std::array<FM_TYPE, NUM_TASK> result_t;
typedef std::array<WT_TYPE, 4> node_eigen_t; 

typedef struct{
    int u;
    int v;
}edge_t;

typedef enum {
    GCN = 0,
    GIN = 1,
    PNA = 2,
    DGN = 3
} Instruction;


extern Instruction instruction; // instruction to decide which GNN has to be inferred

// #region Global Variables
// #Sub-region - Layer Weights of all models
//extern WT_TYPE GCN_convs_GIN_node_mlp_1_weights[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM];
//extern WT_TYPE GIN_node_mlp_2_weights[NUM_LAYERS][EMB_DIM][DGN_LIN_GIN_MLP_1_OUT];
extern std::array<std::array<WT_TYPE, NUM_AGGRS>, NUM_SCALERS> PNA_node_conv_weights[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM];
extern WT_TYPE layers_posttrans_fully_connected_0_linear_weights[4][EMB_DIM][2][EMB_DIM];

// #Sub-region - Layer Bias of all models and Convolution root embedding weights for GCN
extern WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT];
extern WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias[NUM_LAYERS][EMB_DIM];
//extern WT_TYPE convs_root_emb_weight_slice[EMB_DIM];

// #Sub-region - Batch Normalization Weights for GCN
//extern WT_TYPE GCN_bn_weights[NUM_LAYERS - 1][EMB_DIM];
//extern WT_TYPE GCN_bn_bias[NUM_LAYERS - 1][EMB_DIM];
//extern WT_TYPE GCN_bn_mean[NUM_LAYERS - 1][EMB_DIM];
//extern WT_TYPE GCN_bn_sqrt_var[NUM_LAYERS - 1][EMB_DIM];
//extern WT_TYPE GCN_bn_weight_final[EMB_DIM];
//extern WT_TYPE GCN_bn_bias_final[EMB_DIM];
//extern WT_TYPE GCN_bn_mean_final[EMB_DIM];
//extern WT_TYPE GCN_bn_sqrt_var_final[EMB_DIM];

// #Sub-region - GIN Epsilon values and PNA Average Degree
extern WT_TYPE GIN_node_mlp_eps_PNA_avg_deg[NUM_LAYERS];

// #Sub-region - Edge Embedding Weights for GCN and GIN
//extern WT_TYPE edge_embedding_weights[EDGE_PARALLEL][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];

// #Sub-region - Graph Level Task Weights and Biases of GCN and GIN, Graph MLP3 Parameters for PNA
//extern WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM];
//extern WT_TYPE graph_pred_bias[NUM_TASK];


extern int degree_table[MAX_NODE];
//extern int degree_table_finalize[MAX_NODE];
extern int degree_tables[EDGE_PARALLEL][MAX_NODE][2];
extern int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
//extern int neighbor_table[MAX_EDGE]
//extern edge_attr_t edge_attrs[EDGE_PARALLEL][MAX_EDGE];
extern int num_of_edges_per_pe[EDGE_PARALLEL];
extern WT_TYPE DGN_eig_w_GCN_norms[EDGE_PARALLEL][MAX_EDGE];
extern FM_TYPE DGN_abssums_PNA_log_degrees[MAX_NODE];
extern WT_TYPE DGN_eigw_sums[MAX_NODE];


// #Sub-region - Message Buffers
extern std::array<FM_TYPE, NUM_AGGRS> messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM];
extern std::array<FM_TYPE, NUM_AGGRS> messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM];

extern FM_TYPE h_node[MAX_NODE][EMB_DIM];


// #Sub-region - PNA Graph MLP Parameters
extern WT_TYPE PNA_graph_DGN_MLP_1_weights[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM];
extern WT_TYPE PNA_graph_DGN_MLP_1_bias[DGN_MLP_PNA_GRAPH_MLP_1_OUT];
extern WT_TYPE PNA_graph_DGN_MLP_2_weights[DGN_MLP_PNA_GRAPH_MLP_2_OUT][DGN_MLP_PNA_GRAPH_MLP_1_OUT];
extern WT_TYPE PNA_graph_DGN_MLP_2_bias[DGN_MLP_PNA_GRAPH_MLP_2_OUT];
extern WT_TYPE PNA_graph_DGN_MLP_3_weights[NUM_TASK][DGN_MLP_PNA_GRAPH_MLP_2_OUT];
extern WT_TYPE PNA_graph_DGN_MLP_3_bias[NUM_TASK];

//extern int max_EMB_DIM;
//extern int max_iter;
//extern int max_NUM_LAYERS;

//Top Function
extern "C"{

void GNN_compute_graphs(
    Instruction instrcution_in,
    int num_graphs,
    int* nums_of_nodes,
    int* nums_of_edges,
    int* reload_weights,
    FM_TYPE out[][NUM_TASK],
    node_feature_t* node_feature_in,
    node_eigen_t* node_eigen_in,
    edge_t* edge_list_in,
    //edge_attr_t* edge_attr_in,
    WT_TYPE node_embedding_h_atom_embedding_list_weight_in[][ND_FEATURE][ND_FEATURE_TOTAL][EMB_DIM],
    //WT_TYPE edge_embedding_weight_in[][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    //WT_TYPE GCN_convs_GIN_node_mlp_1_weight_in[][NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM],
    WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_in[][NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT],
    //WT_TYPE GIN_node_mlp_2_weight_in[][NUM_LAYERS][EMB_DIM][DGN_LIN_GIN_MLP_1_OUT],
    WT_TYPE layers_posttrans_fully_connected_0_linear_weight_in[][4][EMB_DIM][2 * EMB_DIM],
    WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_in [][NUM_LAYERS][EMB_DIM],
    WT_TYPE PNA_node_conv_weight_in[][NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM],
    WT_TYPE bn_weight_PNA_graph_DGN_MLP_1_weight_in[][DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE bn_bias_PNA_graph_DGN_MLP_1_bias_in[][NUM_LAYERS][EMB_DIM],
    WT_TYPE bn_mean_PNA_graph_DGN_MLP_2_weight_in[][DGN_MLP_PNA_GRAPH_MLP_2_OUT][EMB_DIM],
    WT_TYPE bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_in[][NUM_LAYERS][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_weight_in[][NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_bias_in[][NUM_TASK],
    WT_TYPE avg_deg_in[][NUM_LAYERS]
);
}

//void GCN_finalize(
//    int num_of_nodes,
//    FM_TYPE* result
//);
#endif


