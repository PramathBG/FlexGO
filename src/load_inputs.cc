#include "load_inputs.h"
#include "message_passing.h"
#include <iostream>
#include <cstring>
#include "ap_fixed.h"

using std::array;

static const int nd_feature_offsets[ND_FEATURE] = {
    // Defined by get_atom_feature_dims() in https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
    0,                                  // possible_atomic_num_list:       119 dims
    119,                                // possible_chirality_list:          5 dims
    119 + 5,                            // possible_degree_list:            12 dims
    119 + 5 + 12,                       // possible_formal_charge_list:     12 dims
    119 + 5 + 12 + 12,                  // possible_numH_list:              10 dims
    119 + 5 + 12 + 12 + 10,             // possible_number_radical_e_list:   6 dims
    119 + 5 + 12 + 12 + 10 + 6,         // possible_hybridization_list:      6 dims
    119 + 5 + 12 + 12 + 10 + 6 + 6,     // possible_is_aromatic_list:        2 dims
    119 + 5 + 12 + 12 + 10 + 6 + 6 + 2  // possible_is_in_ring_list:         2 dims
                                        // ND_FEATURE_TOTAL (dcl.h):       174 dims
};

static const int nd_feature_table[ND_FEATURE] = {119, 4, 12, 12, 10, 6, 6, 2, 2};

void load_weights(
    WT_TYPE GCN_convs_GIN_node_mlp_1_weight_in[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM],
    WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_in[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT],
    WT_TYPE GIN_node_mlp_2_weight_in[NUM_LAYERS][EMB_DIM][DGN_LIN_GIN_MLP_1_OUT],
    WT_TYPE layers_posttrans_fully_connected_0_linear_weight_in[4][EMB_DIM][2 * EMB_DIM],
    WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE PNA_node_conv_weight_in[DGN_PNA_NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM],
    WT_TYPE edge_embedding_weight_in[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    WT_TYPE bn_weight_PNA_graph_DGN_MLP_1_weight_in[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE bn_bias_PNA_graph_DGN_MLP_1_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE bn_mean_PNA_graph_DGN_MLP_2_weight_in[DGN_MLP_PNA_GRAPH_MLP_2_OUT][EMB_DIM],
    WT_TYPE bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_weight_in[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_bias_in[NUM_TASK],
    WT_TYPE avg_deg_in
)
{
#pragma HLS INLINE off

    GIN_node_mlp_eps_PNA_avg_deg[0] = avg_deg_in;
    std::memcpy(GCN_convs_GIN_node_mlp_1_weights, GCN_convs_GIN_node_mlp_1_weight_in, sizeof(WT_TYPE) * NUM_LAYERS * DGN_LIN_GIN_MLP_1_OUT * EMB_DIM);
    std::memcpy(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias, GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_in, sizeof(WT_TYPE) * NUM_LAYERS * DGN_LIN_GIN_MLP_1_OUT);
    std::memcpy(GIN_node_mlp_2_weights, GIN_node_mlp_2_weight_in, sizeof(WT_TYPE) * NUM_LAYERS * EMB_DIM * DGN_LIN_GIN_MLP_1_OUT);
    std::memcpy(layers_posttrans_fully_connected_0_linear_weights, layers_posttrans_fully_connected_0_linear_weight_in, sizeof(WT_TYPE) * 4 * EMB_DIM * 2 * EMB_DIM);
    std::memcpy(GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias, GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_in, sizeof(WT_TYPE) * NUM_LAYERS * EMB_DIM);

    load_layer_2D_params: for(int l = 0; l < DGN_MLP_PNA_GRAPH_MLP_1_OUT; l++)
    {
        PNA_graph_DGN_MLP_1_bias[l] = bn_bias_PNA_graph_DGN_MLP_1_bias_in[0][l];
        if(l < DGN_MLP_PNA_GRAPH_MLP_2_OUT)
        {
            PNA_graph_DGN_MLP_2_bias[l] = bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_in[0][l];
        }

        load_layer_2D_params_dim: for(int dim = 0; dim < EMB_DIM; dim++)
        {
#pragma HLS PIPELINE II=6
            if (l < NUM_LAYERS)
            {
                if(l != NUM_LAYERS - 1)
                {
                    GCN_bn_weights[l][dim] = bn_weight_PNA_graph_DGN_MLP_1_weight_in[l][dim];
                    GCN_bn_bias[l][dim] = bn_bias_PNA_graph_DGN_MLP_1_bias_in[l][dim];
                    GCN_bn_mean[l][dim] = bn_mean_PNA_graph_DGN_MLP_2_weight_in[l][dim];
                    GCN_bn_sqrt_var[l][dim] = hls::sqrt(bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_in[l][dim] + ap_fixed_epsilon<WT_TYPE>());
                }
                else
                {
                    GCN_bn_weight_final[dim] = bn_weight_PNA_graph_DGN_MLP_1_weight_in[l][dim];
                    GCN_bn_bias_final[dim] = bn_bias_PNA_graph_DGN_MLP_1_bias_in[l][dim];
                    GCN_bn_mean_final[dim] = bn_mean_PNA_graph_DGN_MLP_2_weight_in[l][dim];
                    GCN_bn_sqrt_var_final[dim] = hls::sqrt(bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_in[l][dim] + ap_fixed_epsilon<WT_TYPE>());
                    convs_root_emb_weight_slice[dim] = GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_in[l][dim];
                }
            }           
            PNA_graph_DGN_MLP_1_weights[l][dim] = bn_weight_PNA_graph_DGN_MLP_1_weight_in[l][dim];
            if(l < DGN_MLP_PNA_GRAPH_MLP_2_OUT && dim < DGN_MLP_PNA_GRAPH_MLP_1_OUT)
            {
                PNA_graph_DGN_MLP_2_weights[l][dim] = bn_mean_PNA_graph_DGN_MLP_2_weight_in[l][dim];
            }
        }
    }

    load_edge_emb_weights: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_edge_emb_weights_feat: for (int i = 0; i < ED_FEATURE_PER_LAYER; i++)
        {
#pragma HLS PIPELINE off
            load_edge_emb_weights_dim: for (int dim = 0; dim < EMB_DIM; dim++)
            {
                WT_TYPE tmp = edge_embedding_weight_in[l][i][dim];
                for (int pe_id = 0; pe_id < EDGE_PARALLEL; pe_id++)
                {
#pragma HLS UNROLL
                    edge_embedding_weights[pe_id][l][i][dim] = tmp;
                }
            }
        }
    }

    for (int layer = 0; layer < DGN_PNA_NUM_LAYERS; layer++)
    {
        for (int dim_out = 0; dim_out < EMB_DIM; dim_out++)
        {   
            for (int scaler = 0; scaler < NUM_SCALERS; scaler++)
            {
                for (int aggr = 0; aggr < NUM_AGGRS; aggr++)
                {
#pragma HLS PIPELINE off
                    for (int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                    {   
                        PNA_node_conv_weights[layer][dim_out][dim_in][scaler][aggr] = PNA_node_conv_weight_in[layer][dim_out][scaler][aggr][dim_in];
                    }
                }
            }    
        }
    }

    int max_dim = (instruction == PNA || instruction == DGN) ? DGN_MLP_PNA_GRAPH_MLP_2_OUT : EMB_DIM;
    load_graph_pred_weights: for (int t = 0; t < NUM_TASK; t++)
    {
        graph_pred_bias[t] = graph_pred_PNA_graph_DGN_MLP_3_bias_in[t];
        PNA_graph_DGN_MLP_3_bias[t] = graph_pred_PNA_graph_DGN_MLP_3_bias_in[t];

        load_graph_pred_weights_dim: for (int dim_in = 0; dim_in < EMB_DIM; dim_in++)
        {
            WT_TYPE graph_pred_PNA_graph_mlp_3_weight_dim = graph_pred_PNA_graph_DGN_MLP_3_weight_in[t][dim_in];
            graph_pred_weights[t][dim_in] = graph_pred_PNA_graph_mlp_3_weight_dim;
            if((instruction == PNA || instruction == DGN) && dim_in < max_dim)
            {
                PNA_graph_DGN_MLP_3_weights[t][dim_in] = graph_pred_PNA_graph_mlp_3_weight_dim;
            }
            
        }
    }
}

void load_graph(
    edge_t* edge_list_in,
    edge_attr_t* edge_attr_in,
    node_eigen_t* node_eigen_in,
    int num_of_nodes,
    int num_of_edges
)
{
#pragma HLS INLINE off

    WT_TYPE degree_inv_sqrt_PNA_out_degree_table[MAX_NODE];
    int neighbor_table_offsets[MAX_NODE];
    int neighbor_tables_offsets[EDGE_PARALLEL][MAX_NODE];

#pragma HLS ARRAY_PARTITION variable=degree_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=degree_tables complete dim=3
#pragma HLS ARRAY_PARTITION variable=neighbor_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=neighbor_tables_offsets complete dim=1
#pragma HLS ARRAY_PARTITION variable=edge_attrs complete dim=1
#pragma HLS ARRAY_PARTITION variable=num_of_edges_per_pe complete dim=1
#pragma HLS ARRAY_PARTITION variable=DGN_eig_w_GCN_norms complete dim=1 

    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES

        degree_table[i] = 0; //in_degree_table in PNA
        degree_table_finalize[i] = 0;
        degree_inv_sqrt_PNA_out_degree_table[i] = 0;
        DGN_abssums_PNA_log_degrees[i] = 0;
        DGN_eigw_sums[i] = 0;

        for (int j = 0; j < EDGE_PARALLEL; j++)
        {
#pragma HLS UNROLL
            degree_tables[j][i][0] = 0;
        }
    }

    for (int i = 0; i < num_of_edges; i++)
    {
        // TODO: can we make this II=1?
#pragma HLS PIPELINE II=3
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_EDGES max=ANALYSIS_MAX_EDGES avg=ANALYSIS_AVG_EDGES
        edge_t edge = edge_list_in[i];
        int u = edge.u;
        int v = edge.v;
        int pe_id = v % EDGE_PARALLEL;
        int node = (instruction == PNA) ? v : u;
        degree_table[node]++;
        degree_table_finalize[node]++;
        if(instruction == GCN || instruction == GIN)
        {
            degree_inv_sqrt_PNA_out_degree_table[u] = hls::recip(hls::sqrt(WT_TYPE(degree_table[u] + 1)));
        }
        if(instruction == PNA)
        {
            degree_inv_sqrt_PNA_out_degree_table[u] = degree_inv_sqrt_PNA_out_degree_table[u] + 1;
        }
        degree_tables[pe_id][u][0]++;
    }

    int acc = 0;
    for (int i = 0; i < EDGE_PARALLEL; i++)
    {
#pragma HLS UNROLL
        num_of_edges_per_pe[i] = 0;
    }

    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES
        int degree = degree_table[i];
        if(instruction == PNA)
        {
            degree = (int)degree_inv_sqrt_PNA_out_degree_table[i];
            DGN_abssums_PNA_log_degrees[i] = (hls::log(FM_TYPE(degree + 1)));
        }
        neighbor_table_offsets[i] = acc;
        acc += degree;
        for (int j = 0; j < EDGE_PARALLEL; j++)
        {
#pragma HLS UNROLL
            int degree_j = degree_tables[j][i][0];
            neighbor_tables_offsets[j][i] = num_of_edges_per_pe[j];
            degree_tables[j][i][1] = num_of_edges_per_pe[j];
            num_of_edges_per_pe[j] += degree_j;
        }
    }

    for (int i = 0; i < num_of_edges; i++)
    {
        // TODO: can we make this II=1?
#pragma HLS PIPELINE II=6
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_EDGES max=ANALYSIS_MAX_EDGES avg=ANALYSIS_AVG_EDGES
        edge_t edge = edge_list_in[i];
        int u = edge.u;
        int v = edge.v;
        int pe_id = v % EDGE_PARALLEL;
        int e = neighbor_table_offsets[u];
        int e_pe = neighbor_tables_offsets[pe_id][u];
        neighbor_table_offsets[u] = e + 1;
        neighbor_tables[pe_id][e_pe] = v / EDGE_PARALLEL;
        neighbor_tables_offsets[pe_id][u] = e_pe + 1;
        edge_attrs[pe_id][e_pe] = edge_attr_in[i];
        if(instruction == GCN)
        {
            DGN_eig_w_GCN_norms[pe_id][e_pe] = degree_inv_sqrt_PNA_out_degree_table[u] * degree_inv_sqrt_PNA_out_degree_table[v];
        }
        WT_TYPE u_eigen = node_eigen_in[u][1];
        WT_TYPE v_eigen = node_eigen_in[v][1];
        WT_TYPE diff_eigen = u_eigen - v_eigen;
        if(instruction == DGN)
        {
            DGN_eig_w_GCN_norms[pe_id][e_pe] = diff_eigen;
            DGN_abssums_PNA_log_degrees[v] += hls::abs(diff_eigen);
        }
        DGN_eigw_sums[v] += diff_eigen;
    }
}

void load_input_node_embeddings(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    node_feature_t* node_feature,
    WT_TYPE node_embedding_h_atom_embedding_list_weight_in[ND_FEATURE][ND_FEATURE_TOTAL][EMB_DIM],
    std::array<FM_TYPE, NUM_AGGRS> messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=nd_feature_offsets complete dim=1
#pragma HLS BIND_STORAGE variable=node_embedding_h_atom_embedding_list_weight_in type=RAM_1WNR
#pragma HLS ARRAY_PARTITION variable=node_embedding_h_atom_embedding_list_weight_in cyclic factor=ND_FEATURE dim=1

    for (int nd = 0; nd < num_of_nodes; nd++)
    {
#pragma HLS PIPELINE II=ceildiv(EMB_DIM, APPLY_PARALLEL)
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES

        array<WT_TYPE, EMB_DIM> weights[ND_FEATURE];
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1

        node_feature_t node_feature_nd = node_feature[nd];
        for (int nf = 0; nf < ND_FEATURE; nf++)
        {
#pragma HLS UNROLL
            int nd_f = nd_feature_offsets[nf] + node_feature_nd[nf];
            weights[nf] = *((array<WT_TYPE, EMB_DIM>*)node_embedding_h_atom_embedding_list_weight_in[0][nd_f]);

            if(instruction == DGN)
            {
                nd_f = node_feature_nd[nf];
                weights[nf] = *((array<WT_TYPE, EMB_DIM>*)node_embedding_h_atom_embedding_list_weight_in[nf][nd_f]);
            }
        }

        for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
        {
            ne_out_t embedding;
            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
            {
                int dim = dim_base + dim_offset;
                FM_TYPE h_node_nd_dim = 0;
                if(dim < EMB_DIM)
                {
                    for (int nf = 0; nf < ND_FEATURE; nf++)
                    {
                        h_node_nd_dim += weights[nf][dim];
                    }
                    h_node[nd][dim] = h_node_nd_dim;
                    embedding[dim_offset] = h_node_nd_dim;
                    
                    // in preparation for the next round of message passing
                    reset_message(messages[nd % EDGE_PARALLEL][nd / EDGE_PARALLEL], dim);
                }
            }
            if(instruction == GIN || instruction == PNA || instruction == DGN)
            {
                embeddings[nd % NODE_PARALLEL] << embedding;
            }
        }
    }
}

void reset_messages(
    std::array<FM_TYPE, NUM_AGGRS> messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int num_of_nodes
)
{
#pragma HLS INLINE off

    int num_iters = ceildiv(num_of_nodes, EDGE_PARALLEL);
    for (int i = 0; i < num_iters; i++)
    {
        for (int dim_base = 0; dim_base < EMB_DIM; dim_base += SCATTER_PARALLEL)
        {
#pragma HLS PIPELINE II=1
            for (int nd_offset = 0; nd_offset < EDGE_PARALLEL; nd_offset++)
            {
                for (int dim_offset = 0; dim_offset < SCATTER_PARALLEL; dim_offset++)
                {
                    int dim = dim_base + dim_offset;
                    if (dim < EMB_DIM) reset_message(messages[nd_offset][i], dim);
                }
            }
        }
    }
}