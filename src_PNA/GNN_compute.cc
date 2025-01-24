#include "dcl.h"
#include "load_inputs.h"
#include "conv_layer.h"
#include "finalize.h"
#include <iostream>
#include "util.h"

extern "C"{

void GNN_compute_graphs(
    //Instruction instrcution_in,
    int num_graphs,
    int* nums_of_nodes,
    int* nums_of_edges,
    int* reload_weights,
    FM_TYPE out[][NUM_TASK],
    node_feature_t* node_feature_in,
    //node_eigen_t* node_eigen_in,
    edge_t* edge_list_in,
    //edge_attr_t* edge_attr_in,
    WT_TYPE node_embedding_h_atom_embedding_list_weight_in[][ND_FEATURE][ND_FEATURE_TOTAL][EMB_DIM],
    //WT_TYPE edge_embedding_weight_in[][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    //WT_TYPE GCN_convs_GIN_node_mlp_1_weight_in[][NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM],
    WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_in[][NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT],
    //WT_TYPE GIN_node_mlp_2_weight_in[][NUM_LAYERS][EMB_DIM][DGN_LIN_GIN_MLP_1_OUT],
    //WT_TYPE layers_posttrans_fully_connected_0_linear_weight_in[][4][EMB_DIM][2 * EMB_DIM],
    //WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_in [][NUM_LAYERS][EMB_DIM],
    WT_TYPE PNA_node_conv_weight_in[][DGN_PNA_NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM],
    WT_TYPE bn_weight_PNA_graph_DGN_MLP_1_weight_in[][DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE bn_bias_PNA_graph_DGN_MLP_1_bias_in[][NUM_LAYERS][EMB_DIM],
    WT_TYPE bn_mean_PNA_graph_DGN_MLP_2_weight_in[][DGN_MLP_PNA_GRAPH_MLP_2_OUT][EMB_DIM],
    WT_TYPE bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_in[][NUM_LAYERS][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_weight_in[][NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_bias_in[][NUM_TASK],
    WT_TYPE avg_deg_in[][NUM_LAYERS]
)
{

#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi depth=(1) port=nums_of_nodes offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=nums_of_edges offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=reload_weights offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=out offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(9 * MAX_NODE) port=node_feature_in offset=slave bundle=mem
//#pragma HLS INTERFACE m_axi depth=(MAX_NODE) port=node_eigen_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(MAX_NODE) port=edge_list_in offset=slave bundle=mem
//#pragma HLS INTERFACE m_axi depth=(MAX_NODE) port=edge_attr_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=node_embedding_h_atom_embedding_list_weight_in offset=slave bundle=mem
//#pragma HLS INTERFACE m_axi depth=(1) port=edge_embedding_weight_in offset=slave bundle=mem
//#pragma HLS INTERFACE m_axi depth=(1) port=GCN_convs_GIN_node_mlp_1_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_in offset=slave bundle=mem
//#pragma HLS INTERFACE m_axi depth=(1) port=GIN_node_mlp_2_weight_in offset=slave bundle=mem
//#pragma HLS INTERFACE m_axi depth=(1) port=layers_posttrans_fully_connected_0_linear_weight_in offset=slave bundle=mem
//#pragma HLS INTERFACE m_axi depth=(1) port=GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=PNA_node_conv_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=bn_weight_PNA_graph_DGN_MLP_1_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=bn_bias_PNA_graph_DGN_MLP_1_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=bn_mean_PNA_graph_DGN_MLP_2_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graph_pred_PNA_graph_DGN_MLP_3_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graph_pred_PNA_graph_DGN_MLP_3_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=avg_deg_in offset=slave bundle=mem

//#pragma HLS BIND_STORAGE variable=layers_posttrans_fully_connected_0_linear_weights type=RAM_2P impl=bram
//#pragma HLS BIND_STORAGE variable=GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias type=RAM_1WNR impl=bram
//#pragma HLS BIND_STORAGE variable=GCN_convs_GIN_node_mlp_1_weights type=RAM_1WNR impl=bram

    //instruction = instrcution_in;
    //max_iter = ceildiv(EMB_DIM, SCATTER_PARALLEL);
    //max_NUM_LAYERS = (instruction == PNA || instruction == DGN) ? DGN_PNA_NUM_LAYERS : NUM_LAYERS;

    for(int graph = 0, weights_ndx = -1, nodes_offset = 0, edges_offset = 0; graph < num_graphs; graph++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_NUM_GRAPHS max=ANALYSIS_NUM_GRAPHS avg=ANALYSIS_NUM_GRAPHS
        int num_of_nodes = nums_of_nodes[graph];
        int num_of_edges = nums_of_edges[graph];
        bool reload_weights_graph = reload_weights[graph];

        if(reload_weights_graph)
        {
            weights_ndx++;
            load_weights(
                //GCN_convs_GIN_node_mlp_1_weight_in[weights_ndx],
                GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_in[weights_ndx],
                //GIN_node_mlp_2_weight_in[weights_ndx],
                //layers_posttrans_fully_connected_0_linear_weight_in[weights_ndx],
                //GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_in[weights_ndx],
                PNA_node_conv_weight_in[weights_ndx],
                //edge_embedding_weight_in[weights_ndx],
                bn_weight_PNA_graph_DGN_MLP_1_weight_in[weights_ndx],
                bn_bias_PNA_graph_DGN_MLP_1_bias_in[weights_ndx],
                bn_mean_PNA_graph_DGN_MLP_2_weight_in[weights_ndx],
                bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_in[weights_ndx],
                graph_pred_PNA_graph_DGN_MLP_3_weight_in[weights_ndx],
                graph_pred_PNA_graph_DGN_MLP_3_bias_in[weights_ndx],
                avg_deg_in[weights_ndx][0]
            );
        }
        load_graph(
            &edge_list_in[edges_offset],
            //&edge_attr_in[edges_offset],
            //&node_eigen_in[nodes_offset],
            num_of_nodes,
            num_of_edges
        );

        reset_messages(messages_pong, num_of_nodes);

        for (int i = 0; i <= DGN_PNA_NUM_LAYERS; i++)
        {
            if (i % 2 == 0)
                compute_CONV_layer(
                    i,
                    messages_ping,
                    messages_pong,
                    &node_feature_in[nodes_offset],
                    node_embedding_h_atom_embedding_list_weight_in[weights_ndx],
                    out[graph],
                    num_of_nodes
                );
            else
                compute_CONV_layer(
                    i,
                    messages_pong,
                    messages_ping,
                    &node_feature_in[nodes_offset],
                    node_embedding_h_atom_embedding_list_weight_in[weights_ndx],
                    out[graph],
                    num_of_nodes
                );
        }
        //if(instruction == GCN)
        //    GCN_finalize(num_of_nodes, out[graph]);
        nodes_offset += num_of_nodes;
        edges_offset += num_of_edges;
    }
}
}

//void GCN_finalize(
//    int num_of_nodes,
//    FM_TYPE* result
//)
//{
//#pragma HLS DATAFLOW 
//
//    hls::stream<ne_out_t> GCN_finalize_embeddings[NODE_PARALLEL];
//#pragma HLS STREAM variable=GCN_finalize_embeddings depth=(4 * (ceildiv(EMB_DIM, APPLY_PARALLEL)))
//
//    h_node_passthrough(GCN_finalize_embeddings, num_of_nodes);
//    if(NUM_LAYERS % 2 == 0)
//        finalize(GCN_finalize_embeddings, 
//                messages_pong, 
//                PNA_graph_DGN_MLP_1_weights,
//                PNA_graph_DGN_MLP_1_bias,
//                PNA_graph_DGN_MLP_2_weights,
//                PNA_graph_DGN_MLP_2_bias,
//                PNA_graph_DGN_MLP_3_weights,
//                PNA_graph_DGN_MLP_3_bias, 
//                graph_pred_weights, 
//                graph_pred_bias, 
//                result, 
//                num_of_nodes);
//    else 
//        finalize(GCN_finalize_embeddings, 
//                messages_ping, 
//                PNA_graph_DGN_MLP_1_weights,
//                PNA_graph_DGN_MLP_1_bias,
//                PNA_graph_DGN_MLP_2_weights,
//                PNA_graph_DGN_MLP_2_bias,
//                PNA_graph_DGN_MLP_3_weights,
//                PNA_graph_DGN_MLP_3_bias, 
//                graph_pred_weights, 
//                graph_pred_bias, 
//                result, 
//                num_of_nodes);
//}