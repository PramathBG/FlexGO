#ifndef __HOST_H__
#define __HOST_H__

#include "dcl.h"
#include <vector>
#include "xcl2.hpp"
#include "../../common/includes/dataset/dataset.hpp"

constexpr int NUM_TRIALS = 1;


template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

extern aligned_vector<WT_TYPE> node_embedding_h_atom_embedding_list_weight_fixed;
extern aligned_vector<WT_TYPE> edge_embedding_weight_fixed;

extern aligned_vector<WT_TYPE> GCN_convs_GIN_node_mlp_1_weight_fixed;
extern aligned_vector<WT_TYPE> GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed;
extern aligned_vector<WT_TYPE> GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed;
extern aligned_vector<WT_TYPE> GIN_node_mlp_2_weight_fixed;
extern aligned_vector<WT_TYPE> PNA_node_conv_weight_fixed;
extern aligned_vector<WT_TYPE> layers_posttrans_fully_connected_0_linear_weight_fixed;

extern aligned_vector<WT_TYPE> bn_weight_PNA_graph_DGN_MLP_1_weight_fixed;
extern aligned_vector<WT_TYPE> bn_bias_PNA_graph_DGN_MLP_1_bias_fixed;
extern aligned_vector<WT_TYPE> bn_mean_PNA_graph_DGN_MLP_2_weight_fixed;
extern aligned_vector<WT_TYPE> bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed;
extern aligned_vector<WT_TYPE> graph_pred_PNA_graph_DGN_MLP_3_weight_fixed;
extern aligned_vector<WT_TYPE> graph_pred_PNA_graph_DGN_MLP_3_bias_fixed;

extern aligned_vector<WT_TYPE> GIN_node_mlp_eps_PNA_avg_deg_fixed;

void read_instruction();
void GNN_compute_one_graph();
void load_weights(int GNN_instruction);
void fetch_one_graph(
    int g,
    char* graph_name,
    aligned_vector<node_feature_t>& node_feature, 
    aligned_vector<edge_t>& edge_list,
    aligned_vector<edge_attr_t>& edge_attr,
    aligned_vector<node_eigen_t>& node_eigen,
    int num_of_nodes, 
    int num_of_edges
);

#endif