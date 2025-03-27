#include "node_embedding.h"
#include <iostream>
#include "message_passing.h"
#include "hls_math.h"

// #region Internal Function Declarations
static void accumulate(
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    FM_TYPE accs[NODE_PARALLEL][DGN_LIN_GIN_MLP_1_OUT],
    // FM_TYPE h_node_buf[NODE_PARALLEL][EMB_DIM],
    // WT_TYPE eps,
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
);

static void output(
    FM_TYPE accs[NODE_PARALLEL][DGN_LIN_GIN_MLP_1_OUT],
    // FM_TYPE h_node_buf[NODE_PARALLEL][EMB_DIM],
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
);
// #endregion

void node_embedding_multi_pe(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int layer_num,
    int num_of_nodes
)
{
#pragma HLS INLINE off 
#pragma HLS ARRAY_PARTITION variable=degree_table cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias cyclic factor=APPLY_PARALLEL dim=2

    FM_TYPE accs_ping[NODE_PARALLEL][DGN_LIN_GIN_MLP_1_OUT];
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=2
    FM_TYPE accs_pong[NODE_PARALLEL][DGN_LIN_GIN_MLP_1_OUT];
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=2
//     FM_TYPE h_node_ping[NODE_PARALLEL][EMB_DIM];
// #pragma HLS ARRAY_PARTITION variable=h_node_ping complete dim=1
// #pragma HLS ARRAY_PARTITION variable=h_node_ping cyclic factor=APPLY_PARALLEL dim=2
//     FM_TYPE h_node_pong[NODE_PARALLEL][EMB_DIM];
// #pragma HLS ARRAY_PARTITION variable=h_node_pong complete dim=1
// #pragma HLS ARRAY_PARTITION variable=h_node_pong cyclic factor=APPLY_PARALLEL dim=2

    // WT_TYPE eps = GIN_node_mlp_eps_PNA_avg_deg[layer_num];

    int num_iters = ceildiv(num_of_nodes, NODE_PARALLEL) + 1;
    for (
        int i = 0, acc_v_base = 0, out_v_base = -NODE_PARALLEL;
        i < num_iters;
        i++, acc_v_base += NODE_PARALLEL, out_v_base += NODE_PARALLEL
    )   
    {
        #pragma HLS LOOP_TRIPCOUNT min=(ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL) + 1) max=(ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL) + 1) avg=(ceildiv(ANALYSIS_AVG_NODES, NODE_PARALLEL) + 1)
        for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
        {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=message inter false
            if(i != 0)
            {
                output(
                    (i % 2 == 0) ? accs_pong : accs_ping,
                    // (i % 2 == 0) ? h_node_pong : h_node_ping,
                    embeddings,
                    layer_num,
                    out_v_base,
                    dim_base,
                    num_of_nodes
                );
            }
            if(i != num_iters - 1)
            {
                accumulate(
                    message,
                    (i % 2 == 0) ? accs_ping : accs_pong,
                    // (i % 2 == 0) ? h_node_ping : h_node_pong,
                    // eps,
                    layer_num,
                    acc_v_base,
                    dim_base,
                    num_of_nodes
                );
            }
        }
    }
}

static void accumulate(
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    FM_TYPE accs[NODE_PARALLEL][DGN_LIN_GIN_MLP_1_OUT],
    // FM_TYPE h_node_buf[NODE_PARALLEL][EMB_DIM],
    // WT_TYPE eps,
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
)
{
#pragma HLS INLINE 

#pragma HLS ARRAY_PARTITION variable=GCN_convs_GIN_node_mlp_1_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=GCN_convs_GIN_node_mlp_1_weights cyclic factor=APPLY_PARALLEL dim=3
#pragma HLS ARRAY_PARTITION variable=GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias complete dim=2

#pragma HLS ARRAY_PARTITION variable=GCN_bn_sqrt_var cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=GCN_bn_weights cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=GCN_bn_mean cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=GCN_bn_bias cyclic factor=APPLY_PARALLEL dim=2

#pragma HLS ARRAY_PARTITION variable=PNA_node_conv_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=PNA_node_conv_weights cyclic factor=APPLY_PARALLEL dim=3
#pragma HLS AGGREGATE variable=PNA_node_conv_weights
// #pragma HLS ARRAY_PARTITION variable=DGN_abssums_PNA_log_degrees cyclic factor=NODE_PARALLEL dim=1
// #pragma HLS AGGREGATE variable=DGN_abssums_PNA_log_degrees

// #pragma HLS ARRAY_PARTITION variable=layers_posttrans_fully_connected_0_linear_weights complete dim=2
// #pragma HLS ARRAY_PARTITION variable=layers_posttrans_fully_connected_0_linear_weights complete dim=3
// #pragma HLS ARRAY_PARTITION variable=layers_posttrans_fully_connected_0_linear_weights cyclic factor=APPLY_PARALLEL dim=4

#pragma HLS ARRAY_PARTITION variable=GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias complete dim=2

// #pragma HLS ARRAY_PARTITION variable=DGN_eigw_sums cyclic factor=NODE_PARALLEL dim=1

    // int max_dim_out = EMB_DIM;

    // if(instruction == GIN)
    //     max_dim_out = DGN_LIN_GIN_MLP_1_OUT;
    
    // WT_TYPE PNA_avg_deg_reg = GIN_node_mlp_eps_PNA_avg_deg[0];

    for(int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
#pragma HLS UNROLL
        int dim_in = dim_base + dim_offset;

        int in_degree;

        //GCN Parameters
        FM_TYPE GCN_h_node_els[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=GCN_h_node_els complete dim=1

        //GIN Parameters
        FM_TYPE GIN_h_node_els[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=GIN_h_node_els complete dim=1

        //PNA Parameters
//         FM_TYPE PNA_sum[NODE_PARALLEL];
// #pragma HLS ARRAY_PARTITION variable=PNA_sum complete dim=1

//        FM_TYPE PNA_sum_squares[NODE_PARALLEL];
// #pragma HLS ARRAY_PARTITION variable=PNA_sum_squares complete dim=1

        FM_TYPE PNA_min[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_min complete dim=1

        FM_TYPE PNA_max[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_max complete dim=1

        FM_TYPE PNA_stddev[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_stddev complete dim=1

        FM_TYPE PNA_T[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_T complete dim=1

        FM_TYPE PNA_scale[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_scale complete dim=1

        //GCN, GIN, DGN activations and PNA mean
        FM_TYPE GCN_GIN_DGN_activations_PNA_mean[2][NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=GCN_GIN_DGN_activations_PNA_mean complete dim=1
#pragma HLS ARRAY_PARTITION variable=GCN_GIN_DGN_activations_PNA_mean complete dim=2

        for(int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
        {
#pragma HLS UNROLL
#pragma HLS DEPENDENCE variable=message inter false
            int v = v_base + v_offset;
            int message_bank;
            int message_bank_v;
            std::array<FM_TYPE, NUM_AGGRS> message_dim_1;
            std::array<FM_TYPE, NUM_AGGRS> message_dim_2;
            if(v < num_of_nodes)  
            {   
                message_bank = v % EDGE_PARALLEL;
                message_bank_v = v / EDGE_PARALLEL;
                message_dim_1 = message[message_bank][message_bank_v][0][dim_in];
                message_dim_2 = message[message_bank][message_bank_v][1][dim_in];

                //clear message table for the next round of message passing
                reset_message(message[message_bank][message_bank_v], dim_in);

                FM_TYPE h_node_v_dim = h_node[v][dim_in];
                // h_node_buf[v_offset][dim_in] = h_node_v_dim;
                // GCN_h_node_els[v_offset] = (instruction == GCN) ? h_node_v_dim : (FM_TYPE)0;
		GCN_h_node_els[v_offset] = h_node_v_dim;
                // GIN_h_node_els[v_offset] = (instruction == GIN) ? h_node_v_dim : (FM_TYPE)0;
                GIN_h_node_els[v_offset] = (FM_TYPE)0;
                
                in_degree = (degree_table[v] == 0) ? 1 : degree_table[v];
                // WT_TYPE DGN_eigw_sums_v = DGN_eigw_sums[v];
                // WT_TYPE DGN_abssums_PNA_log_degrees_v = DGN_abssums_PNA_log_degrees[v];
                
                if(layer_num == 0)
                {
                    GCN_GIN_DGN_activations_PNA_mean[0][v_offset] = GCN_h_node_els[v_offset] + message_dim_1[0] + GIN_h_node_els[v_offset];
                }
                else 
                {
                    WT_TYPE bn_sqrt_var_dim = GCN_bn_sqrt_var[layer_num - 1][dim_in];
                    WT_TYPE bn_weight_dim = GCN_bn_weights[layer_num - 1][dim_in];
                    WT_TYPE bn_mean_dim = GCN_bn_mean[layer_num - 1][dim_in];
                    WT_TYPE bn_bias_dim = GCN_bn_bias[layer_num - 1][dim_in];
                    WT_TYPE convs_root_emb_weight_dim = GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias[layer_num - 1][dim_in];

                    // if(instruction == GIN)
                    //     convs_root_emb_weight_dim = (WT_TYPE)0;
                    
                    GCN_GIN_DGN_activations_PNA_mean[0][v_offset] = message_dim_1[0] + ap_fixed_relu<FM_TYPE>(GCN_h_node_els[v_offset] + convs_root_emb_weight_dim) / (in_degree + 1) + GIN_h_node_els[v_offset];
                    // if(instruction == GCN)
                    // {
                        GCN_GIN_DGN_activations_PNA_mean[0][v_offset] = (GCN_GIN_DGN_activations_PNA_mean[0][v_offset] - bn_mean_dim) / bn_sqrt_var_dim * bn_weight_dim + bn_bias_dim;
                        GCN_GIN_DGN_activations_PNA_mean[0][v_offset] = ap_fixed_relu<FM_TYPE>(GCN_GIN_DGN_activations_PNA_mean[0][v_offset]);
                    // }
                }

                // PNA_sum[v_offset] = message_dim_1[AGGR_MEAN];
                // PNA_sum_squares[v_offset] = message_dim_1[AGGR_STD];
                PNA_min[v_offset] = message_dim_1[AGGR_MIN];
                PNA_max[v_offset] = message_dim_1[AGGR_MAX];
                PNA_stddev[v_offset] = (FM_TYPE)0;
                PNA_T[v_offset] = (FM_TYPE)0;
                PNA_scale[v_offset] = (FM_TYPE)0;

//                if(instruction == PNA)
//                {   
//                    GCN_GIN_DGN_activations_PNA_mean[0][v_offset] = PNA_sum[v_offset] / in_degree;
//                    PNA_stddev[v_offset] =  hls::sqrt(ap_fixed_relu<FM_TYPE>(
//                        FM_TYPE(PNA_sum_squares[v_offset] / in_degree) - FM_TYPE(GCN_GIN_DGN_activations_PNA_mean[0][v_offset] * GCN_GIN_DGN_activations_PNA_mean[0][v_offset])));    
//
//                    PNA_T[v_offset] = DGN_abssums_PNA_log_degrees_v / PNA_avg_deg_reg;
//                    PNA_scale[v_offset] = PNA_avg_deg_reg / DGN_abssums_PNA_log_degrees_v;
//                    if(PNA_scale[v_offset] == 0) PNA_scale[v_offset] = 1; 
//                } 

//                if(instruction == DGN)
//                {
//                    DGN_abssums_PNA_log_degrees_v = (DGN_abssums_PNA_log_degrees_v == 0) ? ap_fixed_epsilon<WT_TYPE>() : DGN_abssums_PNA_log_degrees_v;
//                    GCN_GIN_DGN_activations_PNA_mean[0][v_offset] = message_dim_1[0] / in_degree;
//                    GCN_GIN_DGN_activations_PNA_mean[1][v_offset] = hls::abs(FM_TYPE((message_dim_2[0] - DGN_eigw_sums_v * h_node_v_dim) / DGN_abssums_PNA_log_degrees_v));
//                    PNA_stddev[v_offset] = GCN_GIN_DGN_activations_PNA_mean[1][v_offset];
//                }
            }
        }

        // for (int dim_out = 0; dim_out < DGN_LIN_GIN_MLP_1_OUT; dim_out++)
        for (int dim_out = 0; dim_out < EMB_DIM; dim_out++)
        {
#pragma HLS UNROLL

            // if(dim_out < max_dim_out)
            // {
                WT_TYPE GCN_GIN_DGN_weight_dim_0;
                WT_TYPE GCN_GIN_DGN_weight_dim_1;
                GCN_GIN_DGN_weight_dim_0 = GCN_convs_GIN_node_mlp_1_weights[layer_num][dim_out][dim_in];
                GCN_GIN_DGN_weight_dim_1 = (WT_TYPE)0;
                WT_TYPE GCN_GIN_DGN_bias = GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias[layer_num][dim_out];

                // if(instruction == DGN)
                // {
                //     GCN_GIN_DGN_weight_dim_0 = layers_posttrans_fully_connected_0_linear_weights[layer_num][dim_out][0][dim_in];
                //     GCN_GIN_DGN_weight_dim_1 = layers_posttrans_fully_connected_0_linear_weights[layer_num][dim_out][1][dim_in];
                //     GCN_GIN_DGN_bias = GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias[layer_num][dim_out];
                // }

                std::array<std::array<WT_TYPE, NUM_AGGRS>, NUM_SCALERS> PNA_weights = PNA_node_conv_weights[layer_num][dim_out][dim_in];           
#pragma HLS AGGREGATE variable=PNA_weights

                PNA_weights[SCALER_NONE][AGGR_MEAN] = GCN_GIN_DGN_weight_dim_0;
                PNA_weights[SCALER_NONE][AGGR_STD] = GCN_GIN_DGN_weight_dim_1;

                // if(instruction == PNA)
                // {
                //     PNA_weights = PNA_node_conv_weights[layer_num][dim_out][dim_in];
                // }

                for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
                {
#pragma HLS UNROLL
                    FM_TYPE addend;
                    addend = FM_TYPE(
                        FM_TYPE(
                            FM_TYPE(
                                FM_TYPE(GCN_GIN_DGN_activations_PNA_mean[0][v_offset] * PNA_weights[SCALER_NONE][AGGR_MEAN])
                                + FM_TYPE(PNA_stddev[v_offset] * PNA_weights[SCALER_NONE][AGGR_STD])
                            ) + FM_TYPE(
                                FM_TYPE(PNA_min[v_offset] * PNA_weights[SCALER_NONE][AGGR_MIN])
                                + FM_TYPE(PNA_max[v_offset] * PNA_weights[SCALER_NONE][AGGR_MAX])
                            )
                        ) + FM_TYPE(
                            FM_TYPE(FM_TYPE(
                                FM_TYPE(
                                    FM_TYPE(GCN_GIN_DGN_activations_PNA_mean[0][v_offset] * PNA_weights[SCALER_T][AGGR_MEAN])
                                    + FM_TYPE(PNA_stddev[v_offset] * PNA_weights[SCALER_T][AGGR_STD])
                                ) + FM_TYPE(
                                    FM_TYPE(PNA_min[v_offset] * PNA_weights[SCALER_T][AGGR_MIN])
                                    + FM_TYPE(PNA_max[v_offset] * PNA_weights[SCALER_T][AGGR_MAX])
                                )
                            ) * PNA_T[v_offset]) + FM_TYPE(FM_TYPE(
                                FM_TYPE(
                                    FM_TYPE(GCN_GIN_DGN_activations_PNA_mean[0][v_offset] * PNA_weights[SCALER_SCALE][AGGR_MEAN])
                                    + FM_TYPE(PNA_stddev[v_offset] * PNA_weights[SCALER_SCALE][AGGR_STD])
                                ) + FM_TYPE(
                                    FM_TYPE(PNA_min[v_offset] * PNA_weights[SCALER_SCALE][AGGR_MIN])
                                    + FM_TYPE(PNA_max[v_offset] * PNA_weights[SCALER_SCALE][AGGR_MAX])
                                )
                            ) * PNA_scale[v_offset])
                        )
                    );
                    FM_TYPE addend_slice_reg = addend;
                    FM_TYPE accs_v_offset_dim_out = accs[v_offset][dim_out];
                    FM_TYPE augend = (dim_in == 0) ? GCN_GIN_DGN_bias : accs_v_offset_dim_out;
                    FM_TYPE accs_v_offset_dim_out_final = addend_slice_reg + augend;
                    accs[v_offset][dim_out] = accs_v_offset_dim_out_final;
                }
            // }
        }
    }
}

static void output(
    FM_TYPE accs[NODE_PARALLEL][DGN_LIN_GIN_MLP_1_OUT],
    // FM_TYPE h_node_buf[NODE_PARALLEL][EMB_DIM],
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
)
{
#pragma HLS INLINE
// #pragma HLS ARRAY_PARTITION variable=GIN_node_mlp_2_weights cyclic factor=APPLY_PARALLEL dim=2
// #pragma HLS ARRAY_PARTITION variable=GIN_node_mlp_2_weights complete dim=3

    ne_out_t outputs[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1
#pragma HLS AGGREGATE variable=outputs

   FM_TYPE GCN_PNA_DGN_emb_result[MAX_NODE][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=GCN_PNA_DGN_emb_result cyclic dim=1 factor=NODE_PARALLEL
#pragma HLS ARRAY_PARTITION variable=GCN_PNA_DGN_emb_result cyclic dim=2 factor=APPLY_PARALLEL

//     FM_TYPE GIN_emb_result[MAX_NODE][EMB_DIM];
// #pragma HLS ARRAY_PARTITION variable=GIN_emb_result cyclic dim=1 factor=NODE_PARALLEL
// #pragma HLS ARRAY_PARTITION variable=GIN_emb_result cyclic dim=2 factor=APPLY_PARALLEL

    for(int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
#pragma HLS UNROLL 
        int dim = dim_base + dim_offset;
        
        //GIN Parameters
        // FM_TYPE GIN_bias;
        // FM_TYPE GIN_results[NODE_PARALLEL];

//#pragma HLS ARRAY_PARTITION variable=GIN_results complete dim=1

        for(int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
        {
#pragma HLS UNROLL
            int v = v_base + v_offset;
            FM_TYPE result;
            if(v < num_of_nodes)
            {
                FM_TYPE acc = accs[v_offset][dim];
                FM_TYPE relu_acc = (hls::signbit(acc)) ? FM_TYPE(0.0) : acc;
                result = accs[v_offset][dim];
                // if(instruction == PNA || instruction == DGN)
                //     result = h_node_buf[v_offset][dim] + relu_acc;

                GCN_PNA_DGN_emb_result[v][dim] = result;
                outputs[v_offset][dim_offset] = result;
            }
            // GIN_bias = GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias[layer_num][dim];
            // GIN_results[v_offset] = GIN_bias;
        }
    

//        if(instruction == GIN)
//        {
//            for(int dim_in = 0; dim_in < DGN_LIN_GIN_MLP_1_OUT; dim_in++)
//            {
//#pragma HLS UNROLL 
//                WT_TYPE weight = GIN_node_mlp_2_weights[layer_num][dim][dim_in];
//
//                for(int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
//                {
//#pragma HLS UNROLL 
//                    FM_TYPE activation = accs[v_offset][dim_in];
//                    GIN_results[v_offset] += ap_fixed_relu(activation) * weight;                                      
//                }
//            }
//
//            for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
//            {
//#pragma HLS UNROLL
//                int v = v_base + v_offset;
//                FM_TYPE result = GIN_results[v_offset];
//
//                if (layer_num != NUM_LAYERS - 1) result = ap_fixed_relu(result);
//
//                outputs[v_offset][dim_offset] = result;
//            
//                if (v < num_of_nodes) GIN_emb_result[v][dim] = result;
//            }
//        }     
    
        for(int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
        {
#pragma HLS UNROLL
            int v = v_base + v_offset;
            FM_TYPE h_node_v_dim;
            if(v < num_of_nodes)
            {
                h_node_v_dim = GCN_PNA_DGN_emb_result[v][dim];
                // if(instruction == GIN)
                // {
                //     h_node_v_dim = GIN_emb_result[v][dim];
                // }
                h_node[v][dim] = h_node_v_dim;
            }
        }
    }

    for(int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
    {
#pragma HLS UNROLL
        int v = v_base + v_offset;
        if(v < num_of_nodes)
        {
            embeddings[v_offset] << outputs[v_offset];
        }
    }
}

