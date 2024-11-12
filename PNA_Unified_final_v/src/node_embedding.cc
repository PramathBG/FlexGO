
#include "node_embedding.h"
#include <iostream>
#include "message_passing.h"
#include "hls_math.h"

// #region Internal Function Declarations
static void accumulate(
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE accs[NODE_PARALLEL][GIN_MLP_1_OUT],
    FM_TYPE h_node_buf[NODE_PARALLEL][EMB_DIM],
    WT_TYPE eps,
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
);

static void output(
    FM_TYPE accs[NODE_PARALLEL][GIN_MLP_1_OUT],
    FM_TYPE h_node_buf[NODE_PARALLEL][EMB_DIM],
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
);

void node_embedding_multi_pe(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int layer_num,
    int num_of_nodes
)
{
#pragma HLS INLINE off 
#pragma HLS ARRAY_PARTITION variable=degree_table cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=GCN_convs_root_emb_weight_GIN_node_mlp_2_bias cyclic factor=APPLY_PARALLEL dim=2

    FM_TYPE accs_ping[NODE_PARALLEL][GIN_MLP_1_OUT];
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=2
    FM_TYPE accs_pong[NODE_PARALLEL][GIN_MLP_1_OUT];
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=2
    FM_TYPE h_node_ping[NODE_PARALLEL][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=h_node_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=h_node_ping cyclic factor=APPLY_PARALLEL dim=2
    FM_TYPE h_node_pong[NODE_PARALLEL][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=h_node_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=h_node_pong cyclic factor=APPLY_PARALLEL dim=2


    WT_TYPE eps = GIN_node_mlp_eps_PNA_avg_deg[layer_num];

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
                    (i % 2 == 0) ? h_node_pong : h_node_ping,
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
                    (i % 2 == 0) ? h_node_ping : h_node_pong,
                    eps,
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
    std::array<FM_TYPE, NUM_AGGRS> message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE accs[NODE_PARALLEL][GIN_MLP_1_OUT],
    FM_TYPE h_node_buf[NODE_PARALLEL][EMB_DIM],
    WT_TYPE eps,
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
)
{
#pragma HLS INLINE 
#pragma HLS ARRAY_PARTITION variable=GCN_convs_GIN_node_mlp_1_weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=GCN_convs_GIN_node_mlp_1_weight cyclic factor=APPLY_PARALLEL dim=3
#pragma HLS ARRAY_PARTITION variable=GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias complete dim=2

#pragma HLS ARRAY_PARTITION variable=bn_sqrt_var_PNA_graph_mlp_2_bias cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=bn_weight_PNA_graph_mlp_1_weights cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=bn_mean_PNA_graph_mlp_2_weights cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=bn_bias_PNA_graph_mlp_1_bias cyclic factor=APPLY_PARALLEL dim=2
//#pragma HLS ARRAY_PARTITION variable=GCN_convs_root_emb_weight_GIN_node_mlp_2_bias cyclic factor=APPLY_PARALLEL dim=2

#pragma HLS ARRAY_PARTITION variable=PNA_node_conv_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=PNA_node_conv_weights cyclic factor=APPLY_PARALLEL dim=3
#pragma HLS AGGREGATE variable=PNA_node_conv_weights
#pragma HLS ARRAY_PARTITION variable=PNA_log_degrees cyclic factor=NODE_PARALLEL dim=1
#pragma HLS AGGREGATE variable=PNA_log_degrees
    int max_dim_out = EMB_DIM;

    if(instruction == GIN)
        max_dim_out = GIN_MLP_1_OUT;
    

    WT_TYPE PNA_avg_deg_reg = GIN_node_mlp_eps_PNA_avg_deg[0];
    for(int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
#pragma HLS UNROLL
        int dim_in = dim_base + dim_offset;

        int in_degree;

        //GCN Parameters
//        FM_TYPE GCN_activations[NODE_PARALLEL];
//#pragma HLS ARRAY_PARTITION variable=GCN_activations complete dim=1
        FM_TYPE GCN_h_node_els[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=GCN_h_node_els complete dim=1

        //GIN Parameters
        FM_TYPE GIN_h_node_els[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=GIN_h_node_els complete dim=1
//        FM_TYPE GIN_activations[NODE_PARALLEL];
//#pragma HLS ARRAY_PARTITION variable=GIN_activations complete dim=1

        //PNA Parameters
        FM_TYPE PNA_sum[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_sum complete dim=1

        FM_TYPE PNA_sum_squares[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_sum_squares complete dim=1

        FM_TYPE PNA_min[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_min complete dim=1

        FM_TYPE PNA_max[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_max complete dim=1

//        FM_TYPE PNA_mean[NODE_PARALLEL];
//#pragma HLS ARRAY_PARTITION variable=PNA_mean complete dim=1

        FM_TYPE PNA_stddev[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_stddev complete dim=1

        FM_TYPE PNA_T[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_T complete dim=1

        FM_TYPE PNA_scale[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=PNA_scale complete dim=1

        FM_TYPE GCN_GIN_activations_PNA_mean[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=GCN_GIN_activations_PNA_mean complete dim=1
        for(int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
        {
#pragma HLS UNROLL
            int v = v_base + v_offset;
            int message_bank;
            int message_bank_v;
            std::array<FM_TYPE, NUM_AGGRS> message_dim;
            if(v < num_of_nodes)  
            {   
                message_bank = v % EDGE_PARALLEL;
                message_bank_v = v / EDGE_PARALLEL;
                message_dim = message[message_bank][message_bank_v][dim_in];
                FM_TYPE h_node_v_dim = h_node[v][dim_in];
                in_degree = degree_table[v];
                GCN_h_node_els[v_offset] = (instruction == GCN) ? h_node_v_dim : (FM_TYPE)0;
                GIN_h_node_els[v_offset] = (instruction == GIN) ? h_node_v_dim : (FM_TYPE)0;
                PNA_stddev[v_offset] = (FM_TYPE)0;
                PNA_T[v_offset] = (FM_TYPE)0;
                PNA_scale[v_offset] = (FM_TYPE)0;
                PNA_min[v_offset] = (FM_TYPE)0;
                PNA_max[v_offset] = (FM_TYPE)0;
                
                if(layer_num == 0)
                {
                    if(instruction == GCN)
                        message_dim[0] = (FM_TYPE)0;
                    GCN_GIN_activations_PNA_mean[v_offset] = GCN_h_node_els[v_offset] + message_dim[0] + GIN_h_node_els[v_offset];
                }
                else 
                {
                    WT_TYPE bn_sqrt_var_dim = bn_sqrt_var_PNA_graph_mlp_2_bias[layer_num - 1][dim_in];
                    WT_TYPE bn_weight_dim = bn_weight_PNA_graph_mlp_1_weights[layer_num - 1][dim_in];
                    WT_TYPE bn_mean_dim = bn_mean_PNA_graph_mlp_2_weights[layer_num - 1][dim_in];
                    WT_TYPE bn_bias_dim = bn_bias_PNA_graph_mlp_1_bias[layer_num - 1][dim_in];
                    WT_TYPE convs_root_emb_weight_dim = GCN_convs_root_emb_weight_GIN_node_mlp_2_bias[layer_num - 1][dim_in];

                    if(instruction == GIN)
                        convs_root_emb_weight_dim = (WT_TYPE)0;
                    
                    GCN_GIN_activations_PNA_mean[v_offset] = message_dim[0] + ap_fixed_relu<FM_TYPE>(GCN_h_node_els[v_offset] + convs_root_emb_weight_dim) / (in_degree + 1) + GIN_h_node_els[v_offset];
                    if(instruction == GCN)
                    {
                        GCN_GIN_activations_PNA_mean[v_offset] = (GCN_GIN_activations_PNA_mean[v_offset] - bn_mean_dim) / bn_sqrt_var_dim * bn_weight_dim + bn_bias_dim;
                        GCN_GIN_activations_PNA_mean[v_offset] = ap_fixed_relu<FM_TYPE>(GCN_GIN_activations_PNA_mean[v_offset]);
                    }
                }
                
                //GIN_activations[v_offset] = message_dim[0] + (1 + eps) * GIN_h_node_els[v_offset];
                //if(instruction == GCN)
                //    PNA_mean[v_offset] = GCN_activations[v_offset];
                //if(instruction == GIN)
                //    PNA_mean[v_offset] = GIN_activations[v_offset];
                
                if(in_degree == 0) in_degree = 1;
                h_node_buf[v_offset][dim_in] = h_node_v_dim;

                PNA_sum[v_offset] = message_dim[AGGR_MEAN];
                PNA_sum_squares[v_offset] = message_dim[AGGR_STD];
                PNA_min[v_offset] = message_dim[AGGR_MIN];
                PNA_max[v_offset] = message_dim[AGGR_MAX];

                if(instruction == PNA)
                {   
                    GCN_GIN_activations_PNA_mean[v_offset] = PNA_sum[v_offset] / in_degree;
                    //cast_type_FM_TYPE cast_val = static_cast<cast_type_FM_TYPE>(ap_fixed_relu<FM_TYPE>(
                    //    FM_TYPE(PNA_sum_squares[v_offset] / in_degree) - FM_TYPE(GCN_GIN_activations_PNA_mean[v_offset] * GCN_GIN_activations_PNA_mean[v_offset]))); 
                    //PNA_stddev[v_offset] = static_cast<FM_TYPE>(hls::sqrt(cast_val));

                    PNA_stddev[v_offset] =  hls::sqrt(ap_fixed_relu<FM_TYPE>(
                        FM_TYPE(PNA_sum_squares[v_offset] / in_degree) - FM_TYPE(GCN_GIN_activations_PNA_mean[v_offset] * GCN_GIN_activations_PNA_mean[v_offset])));    

                    FM_TYPE PNA_log_degrees_reg = PNA_log_degrees[v];
                    PNA_T[v_offset] = PNA_log_degrees_reg / PNA_avg_deg_reg;
                    PNA_scale[v_offset] = PNA_avg_deg_reg / PNA_log_degrees_reg;
                    if(PNA_scale[v_offset] == 0) PNA_scale[v_offset] = 1; 
                }
                //clear message table for the next round of message passing
                reset_message(message[message_bank][message_bank_v], dim_in);
            }
        }    
        
        for (int dim_out = 0; dim_out < GIN_MLP_1_OUT; dim_out++)
        {
#pragma HLS UNROLL

            if(dim_out < max_dim_out)
            {
                for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
                {
#pragma HLS UNROLL
                    FM_TYPE addend;
                    FM_TYPE GCN_GIN_weight = GCN_convs_GIN_node_mlp_1_weight[layer_num][dim_out][dim_in];
                    FM_TYPE bias = GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias[layer_num][dim_out];
                    
                    //addend = GCN_activations[v_offset] * GCN_GIN_weight;
                    //if(instruction == GIN) 
                    //{
                    //    addend = GIN_activations[v_offset] * GCN_GIN_weight;
                    //}
                    //if(instruction == PNA)
                    //{
                        std::array<std::array<WT_TYPE, NUM_AGGRS>, NUM_SCALERS> PNA_weights = PNA_node_conv_weights[layer_num][dim_out][dim_in];           
#pragma HLS AGGREGATE variable=PNA_weights
                        if(instruction == GCN || instruction == GIN)
                        {
                            PNA_weights[SCALER_NONE][AGGR_MEAN] = GCN_GIN_weight;
                            PNA_weights[SCALER_NONE][AGGR_STD] = (WT_TYPE)0;
                            PNA_weights[SCALER_NONE][AGGR_MIN] = (WT_TYPE)0;
                            PNA_weights[SCALER_NONE][AGGR_MAX] = (WT_TYPE)0;

                            PNA_weights[SCALER_SCALE][AGGR_MEAN] = (WT_TYPE)0;
                            PNA_weights[SCALER_SCALE][AGGR_STD] = (WT_TYPE)0;
                            PNA_weights[SCALER_SCALE][AGGR_MIN] = (WT_TYPE)0;
                            PNA_weights[SCALER_SCALE][AGGR_MAX] = (WT_TYPE)0;

                            PNA_weights[SCALER_T][AGGR_MEAN] = (WT_TYPE)0;
                            PNA_weights[SCALER_T][AGGR_STD] = (WT_TYPE)0;
                            PNA_weights[SCALER_T][AGGR_MIN] = (WT_TYPE)0;
                            PNA_weights[SCALER_T][AGGR_MAX] = (WT_TYPE)0;

                        }
                        addend = FM_TYPE(
                            FM_TYPE(
                                FM_TYPE(
                                    FM_TYPE(GCN_GIN_activations_PNA_mean[v_offset] * PNA_weights[SCALER_NONE][AGGR_MEAN])
                                    + FM_TYPE(PNA_stddev[v_offset] * PNA_weights[SCALER_NONE][AGGR_STD])
                                ) + FM_TYPE(
                                    FM_TYPE(PNA_min[v_offset] * PNA_weights[SCALER_NONE][AGGR_MIN])
                                    + FM_TYPE(PNA_max[v_offset] * PNA_weights[SCALER_NONE][AGGR_MAX])
                                )
                            ) + FM_TYPE(
                                FM_TYPE(FM_TYPE(
                                    FM_TYPE(
                                        FM_TYPE(GCN_GIN_activations_PNA_mean[v_offset] * PNA_weights[SCALER_T][AGGR_MEAN])
                                        + FM_TYPE(PNA_stddev[v_offset] * PNA_weights[SCALER_T][AGGR_STD])
                                    ) + FM_TYPE(
                                        FM_TYPE(PNA_min[v_offset] * PNA_weights[SCALER_T][AGGR_MIN])
                                        + FM_TYPE(PNA_max[v_offset] * PNA_weights[SCALER_T][AGGR_MAX])
                                    )
                                ) * PNA_T[v_offset]) + FM_TYPE(FM_TYPE(
                                    FM_TYPE(
                                        FM_TYPE(GCN_GIN_activations_PNA_mean[v_offset] * PNA_weights[SCALER_SCALE][AGGR_MEAN])
                                        + FM_TYPE(PNA_stddev[v_offset] * PNA_weights[SCALER_SCALE][AGGR_STD])
                                    ) + FM_TYPE(
                                        FM_TYPE(PNA_min[v_offset] * PNA_weights[SCALER_SCALE][AGGR_MIN])
                                        + FM_TYPE(PNA_max[v_offset] * PNA_weights[SCALER_SCALE][AGGR_MAX])
                                    )
                                ) * PNA_scale[v_offset])
                            )
                        );
                    //}
                    //accs[v_offset][dim_out] = (addend + ((dim_in == 0) ? bias : accs[v_offset][dim_out]));
                    FM_TYPE addend_slice_reg = addend;
                    FM_TYPE accs_v_offset_dim_out = accs[v_offset][dim_out];
                    FM_TYPE augend = (dim_in == 0) ? bias : accs_v_offset_dim_out;
                    FM_TYPE accs_v_offset_dim_out_final = addend_slice_reg + augend;
                    accs[v_offset][dim_out] = accs_v_offset_dim_out_final;
                }
            }
        }
    }           
}


static void output(
    FM_TYPE accs[NODE_PARALLEL][GIN_MLP_1_OUT],
    FM_TYPE h_node_buf[NODE_PARALLEL][EMB_DIM],
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
)
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=GIN_node_mlp_2_weights cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=GIN_node_mlp_2_weights complete dim=3

    ne_out_t outputs[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1
#pragma HLS AGGREGATE variable=outputs

    FM_TYPE GCN_emb_result[MAX_NODE][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=GCN_emb_result cyclic dim=1 factor=NODE_PARALLEL
#pragma HLS ARRAY_PARTITION variable=GCN_emb_result cyclic dim=2 factor=APPLY_PARALLEL

    FM_TYPE GIN_emb_result[MAX_NODE][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=GIN_emb_result cyclic dim=1 factor=NODE_PARALLEL
#pragma HLS ARRAY_PARTITION variable=GIN_emb_result cyclic dim=2 factor=APPLY_PARALLEL

    FM_TYPE PNA_emb_result[MAX_NODE][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=PNA_emb_result cyclic dim=1 factor=NODE_PARALLEL
#pragma HLS ARRAY_PARTITION variable=PNA_emb_result cyclic dim=2 factor=APPLY_PARALLEL

    FM_TYPE h_node_tmp[MAX_NODE][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=h_node_tmp cyclic dim=1 factor=NODE_PARALLEL
#pragma HLS ARRAY_PARTITION variable=h_node_tmp cyclic dim=2 factor=APPLY_PARALLEL

    for(int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
#pragma HLS UNROLL 
        int dim = dim_base + dim_offset;
        
        //GIN Parameters
        FM_TYPE GIN_bias;
        FM_TYPE GIN_results[NODE_PARALLEL];

#pragma HLS ARRAY_PARTITION variable=GIN_results complete dim=1

        for(int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
        {
#pragma HLS UNROLL
            int v = v_base + v_offset;
            FM_TYPE result;
            if(v < num_of_nodes)
            {
                FM_TYPE acc = accs[v_offset][dim];
                FM_TYPE relu_acc = (hls::signbit(acc)) ? FM_TYPE(0.0) : acc;
                //FM_TYPE relu_acc = (acc < 0) ? FM_TYPE(0.0) : acc;
                result = accs[v_offset][dim];
                if(instruction == PNA)
                    result = h_node_buf[v_offset][dim] + relu_acc;
                GCN_emb_result[v][dim] = result;
                PNA_emb_result[v][dim] = result;
                outputs[v_offset][dim_offset] = result;
            }
            GIN_bias = GCN_convs_root_emb_weight_GIN_node_mlp_2_bias[layer_num][dim];
            GIN_results[v_offset] = GIN_bias;
        }
    

        if(instruction == GIN)
        {
            for(int dim_in = 0; dim_in < GIN_MLP_1_OUT; dim_in++)
            {
#pragma HLS UNROLL 
                WT_TYPE weight = GIN_node_mlp_2_weights[layer_num][dim][dim_in];

                for(int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
                {
#pragma HLS UNROLL 
                    FM_TYPE activation = accs[v_offset][dim_in];
                    GIN_results[v_offset] += ap_fixed_relu(activation) * weight;                                      
                }
            }

            for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
            {
#pragma HLS UNROLL
                int v = v_base + v_offset;
                FM_TYPE result = GIN_results[v_offset];

                if (layer_num != NUM_LAYERS - 1) result = ap_fixed_relu(result);

                outputs[v_offset][dim_offset] = result;
            
                if (v < num_of_nodes) GIN_emb_result[v][dim] = result;
            }
        }     
    
        for(int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
        {
#pragma HLS UNROLL
            int v = v_base + v_offset;
            if(v < num_of_nodes)
            {
                if(instruction == GCN)
                    h_node_tmp[v][dim] = GCN_emb_result[v][dim];
                
                if(instruction == GIN)
                    h_node_tmp[v][dim] = GIN_emb_result[v][dim];
                
                if(instruction == PNA)
                    h_node_tmp[v][dim] = PNA_emb_result[v][dim];
            
                FM_TYPE h_node_v_dim = h_node_tmp[v][dim];
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
