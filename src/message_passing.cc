#include "message_passing.h"
#include <iostream>

static const int ed_feature_offsets[EDGE_ATTR] = {0, 5, 11};

// #region Internal Functions Declarations
static void filter(
    int pe_id,
    hls::stream<mp_in_t> unfiltered_embeddings_per_node[NODE_PARALLEL],
    hls::stream<int>& degrees,
    hls::stream<mp_in_t>& filtered_embeddings_per_node,
    int num_of_nodes
);

static void scatter(
    int pe_id,
    int layer_num,
    hls::stream<int>& degrees,
    hls::stream<mp_in_t>& embeddings_per_node,
    std::array<FM_TYPE, NUM_AGGRS> message[ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM]
);
// #endregion

void message_passing_pe(
    int pe_id,
    hls::stream<mp_in_t> node_embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int layer_num,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<int> degrees("degrees");
#pragma HLS STREAM variable=degrees depth=20

    hls::stream<mp_in_t> filtered_embeddings_per_node("filtered_embeddings_per_node");
#pragma HLS STREAM variable=filtered_embeddings_per_node depth=(20 * ceildiv(EMB_DIM, SCATTER_PARALLEL))

    filter(pe_id, node_embeddings, degrees, filtered_embeddings_per_node, num_of_nodes);
    scatter(pe_id, layer_num, degrees, filtered_embeddings_per_node, message);
}

static void filter(
    int pe_id,
    hls::stream<mp_in_t> unfiltered_embeddings_per_node[NODE_PARALLEL],
    hls::stream<int>& degrees,
    hls::stream<mp_in_t>& filtered_embeddings_per_node,
    int num_of_nodes
)
{
#pragma HLS INLINE off

    for(int nd = 0; nd < num_of_nodes; nd++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES
#pragma HLS ARRAY_PARTITION variable=degree_tables cyclic factor=ceildiv(EMB_DIM, SCATTER_PARALLEL) dim=2
        for (int i = 0; i < ceildiv(EMB_DIM, SCATTER_PARALLEL); i++)
        {
#pragma HLS PIPELINE II=1

            mp_in_t embedding;
#pragma HLS AGGREGATE variable=embedding
            unfiltered_embeddings_per_node[nd % NODE_PARALLEL] >> embedding;

            int degree = degree_tables[pe_id][nd][0];
            if (degree != 0)
            {
                if (i == 0)
                {
#pragma HLS OCCURRENCE cycle=ceildiv(EMB_DIM, SCATTER_PARALLEL)
                    degrees << degree;
                }
                filtered_embeddings_per_node << embedding;
            }
        }
    }
}

static void scatter(
    int pe_id,
    int layer_num,
    hls::stream<int>& degrees,
    hls::stream<mp_in_t>& embeddings_per_node,
    std::array<FM_TYPE, NUM_AGGRS> message[ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM]
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ed_feature_offsets complete dim=1
#pragma HLS ARRAY_PARTITION variable=edge_embedding_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=edge_embedding_weights cyclic factor=SCATTER_PARALLEL dim=4
#pragma HLS BIND_STORAGE variable=edge_embedding_weights type=ram_1wnr
#pragma HLS AGGREGATE variable=edge_attrs
#pragma HLS AGGREGATE variable=message

    mp_in_t mp_ins[ceildiv(EMB_DIM, SCATTER_PARALLEL)];
    int e_start = 0;
    int e_end = 0;
    int num_of_edges = num_of_edges_per_pe[pe_id];

    for(int e = 0; e < num_of_edges; e++)
    {
#pragma HLS LOOP_TRIPCOUNT min=0 max=ANALYSIS_MAX_EDGES avg=ceildiv(ANALYSIS_AVG_EDGES, EDGE_PARALLEL)

        int v = neighbor_tables[pe_id][e];
        edge_attr_t attrs = edge_attrs[pe_id][e];
        WT_TYPE GCN_norm = DGN_eig_w_GCN_norms[pe_id][e];
        WT_TYPE DGN_eigen_w_e = DGN_eig_w_GCN_norms[pe_id][e];
        if(instruction == GIN)
            GCN_norm = (FM_TYPE)1;

        for (int i = 0, dim_base = 0; i < ceildiv(EMB_DIM, SCATTER_PARALLEL); i++, dim_base += SCATTER_PARALLEL)
        {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=message inter true distance=ceildiv(EMB_DIM, SCATTER_PARALLEL)
#pragma HLS DEPENDENCE variable=message inter false distance=1

            if (e >= e_end)
            {
                int degree;
                degrees >> degree;
                e_start = e;
                e_end = e + degree;
            }

            mp_in_t node_embedding;
#pragma HLS AGGREGATE variable=node_embedding

            if (e == e_start)
            {
                embeddings_per_node >> node_embedding;
                mp_ins[i] = node_embedding;
            }
            else
            {
                node_embedding = mp_ins[i];
            }

            for(int dim_offset = 0; dim_offset < SCATTER_PARALLEL; dim_offset++)
            {
#pragma HLS UNROLL

                int dim = dim_base + dim_offset;
                if(dim < EMB_DIM)
                {
                    FM_TYPE message_tmp [2][NUM_AGGRS];

                    for(int aggr = 0; aggr < NUM_AGGRS; aggr++)
                    {
                        message_tmp[0][aggr] = message[v][0][dim][aggr];
                        message_tmp[1][aggr] = message[v][1][dim][aggr];
                    }

                    if(instruction == GCN || instruction == GIN)
                    {
                        FM_TYPE edge_embed = (FM_TYPE)0;
                        edge_embed_loop: for(int ef = 0; ef < EDGE_ATTR; ef++)
                        {
#pragma HLS UNROLL
                            int e_ef = ed_feature_offsets[ef] + attrs[ef];
                            int layer_idx = layer_num;

                            if(instruction == GCN)
                            {
                                layer_idx = layer_num - 1;
                            }
                            edge_embed += edge_embedding_weights[pe_id][layer_idx][e_ef][dim];
                        }

                        FM_TYPE total_embed = edge_embed + node_embedding[dim_offset];
                        message_tmp[0][0] += GCN_norm * ap_fixed_relu(total_embed);
                    }
                    else if(instruction == PNA)
                    {
                        FM_TYPE embedding_dim = node_embedding[dim_offset];
                        message_tmp[0][AGGR_MEAN] += embedding_dim;
                        message_tmp[0][AGGR_STD] += FM_TYPE(embedding_dim * embedding_dim);
                        if (embedding_dim < message_tmp[0][AGGR_MIN])
                            message_tmp[0][AGGR_MIN] = embedding_dim;
                        if (embedding_dim > message_tmp[0][AGGR_MAX])
                            message_tmp[0][AGGR_MAX] = embedding_dim;
                    }
                    else if(instruction == DGN)
                    {
                        FM_TYPE embedding_dim = node_embedding[dim_offset];
                        message_tmp[0][0] += embedding_dim;
                        message_tmp[1][0] += embedding_dim * DGN_eigen_w_e;
                    }
                    for(int aggr = 0; aggr < NUM_AGGRS; aggr++)
                    {
                        message[v][0][dim][aggr] = message_tmp[0][aggr];
                        message[v][1][dim][aggr] = message_tmp[1][aggr];
                    }
                }
            }
        }
    }
}

void reset_message(std::array<FM_TYPE, NUM_AGGRS> message[2][EMB_DIM], int dim)
{
    message[0][dim][AGGR_MEAN] = 0;
    message[0][dim][AGGR_STD] = 0;
    message[0][dim][AGGR_MIN] = ap_fixed_max<FM_TYPE>();
    message[0][dim][AGGR_MAX] = ap_fixed_min<FM_TYPE>();
    message[1][dim][AGGR_MEAN] = 0;
    message[1][dim][AGGR_STD] = 0;
    message[1][dim][AGGR_MIN] = ap_fixed_max<FM_TYPE>();
    message[1][dim][AGGR_MAX] = ap_fixed_min<FM_TYPE>();
}