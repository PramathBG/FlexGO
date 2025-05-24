#include "finalize.h"
#include "linear.h"
#include <iostream>

using std::array;

// #region Internal Functions Declarations 
void global_mean_pooling(
    hls::stream<ne_out_t>& GCN_h_graph,
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int num_of_nodes
);
void check_linear(
    hls::stream<ne_out_t>& GCN_h_graph,
    WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias[NUM_TASK],
    FM_TYPE* result

);
// #endregion

void finalize(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ne_out_t> GCN_h_graph; 
#pragma HLS STREAM variable=GCN_h_graph depth=ceildiv(EMB_DIM, APPLY_PARALLEL)

    global_mean_pooling(
        GCN_h_graph,
        embeddings,
        message,
        num_of_nodes
    );

    check_linear(GCN_h_graph,
    graph_pred_weights,
    graph_pred_bias,
    result);
}
void global_mean_pooling(
    hls::stream<ne_out_t>& GCN_h_graph,
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int num_of_nodes
)
{
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable=convs_root_emb_weight_slice cyclic factor=APPLY_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=degree_table_finalize cyclic factor=NODE_PARALLEL
#pragma HLS ARRAY_PARTITION variable=GCN_bn_sqrt_var_final cyclic factor=APPLY_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=GCN_bn_weight_final cyclic factor=APPLY_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=GCN_bn_mean_final cyclic factor=APPLY_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=GCN_bn_bias_final cyclic factor=APPLY_PARALLEL dim=1

    FM_TYPE sums[EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=sums cyclic factor=APPLY_PARALLEL dim=1

    int num_iters = ceildiv(num_of_nodes, NODE_PARALLEL) - 1;
    int tail_nodes = (((num_of_nodes - 1) % NODE_PARALLEL)) + 1;

        num_iters = ceildiv(num_of_nodes, NODE_PARALLEL);
        ne_out_t embedding_slice[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=embedding_slice complete dim=1

        GCN_post_mp_then_global_mean_pooling_main: for(int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
        {
#pragma HLS PIPELINE II=1
            for(int i = 0; i < num_iters; i++)
            {
#pragma HLS LOOP_TRIPCOUNT min=ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL) max=ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL) avg=ceildiv(ANALYSIS_AVG_NODES, NODE_PARALLEL)
                int nd_base = (i * NODE_PARALLEL);
                for(int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
                {
#pragma HLS UNROLL
                    if(i == num_iters - 1 && nd_offset == tail_nodes)
                    {
                        break;
                    }
                    embeddings[nd_offset] >> embedding_slice[nd_offset];
                }

                if(i == 0)
                {
                    for(int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
                    {
#pragma HLS UNROLL      
                        sums[dim_offset] = 0;
                    }
                }

                for(int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
                {
#pragma HLS UNROLL
                    int dim = dim_base + dim_offset;
                    WT_TYPE convs_root_emb_weight_dim;
                    WT_TYPE bn_sqrt_var_dim;
                    WT_TYPE bn_weight_dim;
                    WT_TYPE bn_mean_dim;
                    WT_TYPE bn_bias_dim;
                    if (dim < EMB_DIM)
                    {
                        convs_root_emb_weight_dim = convs_root_emb_weight_slice[dim];
                        bn_sqrt_var_dim = GCN_bn_sqrt_var_final[dim];
                        bn_weight_dim = GCN_bn_weight_final[dim];
                        bn_mean_dim = GCN_bn_mean_final[dim];
                        bn_bias_dim = GCN_bn_bias_final[dim];
                    }
                    for(int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
                    {
                        int nd = nd_base + nd_offset;
                        if(nd < num_of_nodes)
                        {
                            FM_TYPE activation = message[nd % EDGE_PARALLEL][nd / EDGE_PARALLEL][0][dim][0];
                            activation += ap_fixed_relu<FM_TYPE>(embedding_slice[nd_offset][dim_offset] + convs_root_emb_weight_dim) / (degree_table_finalize[nd] + 1);
                            activation = (activation - bn_mean_dim) / bn_sqrt_var_dim * bn_weight_dim + bn_bias_dim;
                            sums[dim_offset] += activation;
                        }
                        
                    }
                }

                if(i == num_iters - 1)
                {
                    ne_out_t h_graph_vals;
#pragma HLS AGGREGATE variable=h_graph_vals

                    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
                    {
#pragma HLS UNROLL
                        h_graph_vals[dim_offset] = sums[dim_offset] / num_of_nodes;
                    }
                    GCN_h_graph << h_graph_vals;
                }
            }
        }
}

void check_linear(
    hls::stream<ne_out_t>& GCN_h_graph,
    WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias[NUM_TASK],
    FM_TYPE* result

)
{   
    FM_TYPE result_tmp[NUM_TASK];
    linear_input_stationary<EMB_DIM, NUM_TASK, APPLY_PARALLEL, false>(
                GCN_h_graph,
                graph_pred_weights,
                graph_pred_bias,
                result_tmp
    );

    for (int task = 0; task < NUM_TASK; task++)
    {
        result[task] = result_tmp[task];
    }
}