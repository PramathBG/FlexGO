#include "finalize.h"
#include "linear.h"
#include <iostream>

using std::array;

// #region Internal Functions Declarations 
void global_mean_pooling(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    FM_TYPE PNA_DGN_h_graph[EMB_DIM],
    int num_of_nodes
);
void check_linear(
    FM_TYPE PNA_DGN_h_graph[EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_weights[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_bias[DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_weights[DGN_MLP_PNA_GRAPH_MLP_2_OUT][DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_bias[DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_weights[NUM_TASK][DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_bias[NUM_TASK],
    FM_TYPE* result

);
void linear_PNA_DGN(
    FM_TYPE PNA_DGN_h_graph[EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_weights[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_bias[DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_weights[DGN_MLP_PNA_GRAPH_MLP_2_OUT][DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_bias[DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_weights[NUM_TASK][DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_bias[NUM_TASK],
    FM_TYPE* result
);
// #endregion

void finalize(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_weights[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_bias[DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_weights[DGN_MLP_PNA_GRAPH_MLP_2_OUT][DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_bias[DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_weights[NUM_TASK][DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    FM_TYPE PNA_DGN_h_graph[EMB_DIM];

    global_mean_pooling(
        embeddings,
        message,
        PNA_DGN_h_graph,
        num_of_nodes
    );

    check_linear(
        PNA_DGN_h_graph,
        PNA_graph_DGN_MLP_1_weights, 
        PNA_graph_DGN_MLP_1_bias,
        PNA_graph_DGN_MLP_2_weights,
        PNA_graph_DGN_MLP_2_bias,
        PNA_graph_DGN_MLP_3_weights,
        PNA_graph_DGN_MLP_3_bias,
        result
    );
}
void global_mean_pooling(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    FM_TYPE PNA_DGN_h_graph[EMB_DIM],
    int num_of_nodes
)
{
#pragma HLS INLINE off

    FM_TYPE sums[EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=sums cyclic factor=APPLY_PARALLEL dim=1

    int num_iters = ceildiv(num_of_nodes, NODE_PARALLEL) - 1;
    int tail_nodes = (((num_of_nodes - 1) % NODE_PARALLEL)) + 1;

        global_mean_pooling_main: for (int i = 0; i < num_iters; i++)
        {
#pragma HLS LOOP_TRIPCOUNT min=(ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL) - 1) max=(ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL) - 1) avg=(ceildiv(ANALYSIS_AVG_NODES, NODE_PARALLEL) - 1)
            for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
            {
#pragma HLS PIPELINE II=1
   
                ne_out_t embeddings_slice[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=embeddings_slice complete dim=1

                for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
                {
#pragma HLS UNROLL
                    embeddings[nd_offset] >> embeddings_slice[nd_offset];
                }

                for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
                {
#pragma HLS UNROLL
                    int dim = dim_base + dim_offset;
                    FM_TYPE h_graph_el = 0;

                    for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
                    {
#pragma HLS UNROLL
                        h_graph_el += embeddings_slice[nd_offset][dim_offset];
                    }

                    if (i != 0) h_graph_el += sums[dim];
                    sums[dim] = h_graph_el;
                }
            }
        }

        global_mean_pooling_tail: for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
        {
#pragma HLS PIPELINE II=1
   
            ne_out_t embeddings_slice[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=embeddings_slice complete dim=1

            for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
            {
#pragma HLS UNROLL
                if (nd_offset == tail_nodes) break;
                embeddings[nd_offset] >> embeddings_slice[nd_offset];
            }

            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
            {
#pragma HLS UNROLL
                int dim = dim_base + dim_offset;
                FM_TYPE h_graph_el = 0;

                for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
                {
#pragma HLS UNROLL
                    if (nd_offset == tail_nodes) break;
                    h_graph_el += embeddings_slice[nd_offset][dim_offset];
                }

                if (num_iters != 0) 
                {
                    h_graph_el += sums[dim];
                }
                PNA_DGN_h_graph[dim] = h_graph_el / num_of_nodes;
            }
        }
}

void check_linear(
    FM_TYPE PNA_DGN_h_graph[EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_weights[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_bias[DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_weights[DGN_MLP_PNA_GRAPH_MLP_2_OUT][DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_bias[DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_weights[NUM_TASK][DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_bias[NUM_TASK],
    FM_TYPE* result

)
{   
    FM_TYPE result_tmp[NUM_TASK];
    linear_PNA_DGN(
        PNA_DGN_h_graph,
        PNA_graph_DGN_MLP_1_weights,
        PNA_graph_DGN_MLP_1_bias,
        PNA_graph_DGN_MLP_2_weights,
        PNA_graph_DGN_MLP_2_bias,
        PNA_graph_DGN_MLP_3_weights,
        PNA_graph_DGN_MLP_3_bias,
        result_tmp
    );

    for (int task = 0; task < NUM_TASK; task++)
    {
        result[task] = result_tmp[task];
    }
}

void linear_PNA_DGN(
    FM_TYPE PNA_DGN_h_graph[EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_weights[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE PNA_graph_DGN_MLP_1_bias[DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_weights[DGN_MLP_PNA_GRAPH_MLP_2_OUT][DGN_MLP_PNA_GRAPH_MLP_1_OUT],
    WT_TYPE PNA_graph_DGN_MLP_2_bias[DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_weights[NUM_TASK][DGN_MLP_PNA_GRAPH_MLP_2_OUT],
    WT_TYPE PNA_graph_DGN_MLP_3_bias[NUM_TASK],
    FM_TYPE* result
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<mlp_xfer_t> PNA_DGN_graph_mlp_1_out ("PNA_DGN_graph_mlp_1_out");
#pragma HLS STREAM variable=PNA_DGN_graph_mlp_1_out depth=ceildiv(DGN_MLP_PNA_GRAPH_MLP_1_OUT, APPLY_PARALLEL)

    FM_TYPE PNA_DGN_graph_mlp_2_out[DGN_MLP_PNA_GRAPH_MLP_2_OUT];

    linear_output_stationary<EMB_DIM, DGN_MLP_PNA_GRAPH_MLP_1_OUT, APPLY_PARALLEL, true>(
        PNA_DGN_h_graph,
        PNA_graph_DGN_MLP_1_weights,
        PNA_graph_DGN_MLP_1_bias,
        PNA_DGN_graph_mlp_1_out
    );

    linear_input_stationary<DGN_MLP_PNA_GRAPH_MLP_1_OUT, DGN_MLP_PNA_GRAPH_MLP_2_OUT, APPLY_PARALLEL, true>(
        PNA_DGN_graph_mlp_1_out,
        PNA_graph_DGN_MLP_2_weights,
        PNA_graph_DGN_MLP_2_bias,
        PNA_DGN_graph_mlp_2_out
    );

    linear<DGN_MLP_PNA_GRAPH_MLP_2_OUT, NUM_TASK, NUM_TASK, false>(
        PNA_DGN_graph_mlp_2_out,
        PNA_graph_DGN_MLP_3_weights,
        PNA_graph_DGN_MLP_3_bias,
        result
    );
}