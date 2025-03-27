#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "testbench.h"


// static const char* GRAPH_NAME_FORMAT = "g%d";
// static const char* GRAPH_INFO_FORMAT = "g%d_info.txt";
static const char* GRAPH_NAME_FORMAT = "graphs/graph_bin/g%d";
static const char* GRAPH_INFO_FORMAT = "graphs/graph_info/g%d_info.txt";

//static node_feature_t node_feature[MAX_NODE * NUM_GRAPHS];
//static edge_t edge_list[MAX_EDGE * NUM_GRAPHS];
//static edge_attr_t edge_attr[MAX_EDGE * NUM_GRAPHS];
node_feature_t* node_feature = (node_feature_t*)malloc(roundup(MAX_NODE * NUM_GRAPHS * sizeof(node_feature_t), 128lu));
edge_t* edge_list = (edge_t*)malloc(roundup(MAX_EDGE * NUM_GRAPHS * sizeof(edge_t), 128lu));
edge_attr_t* edge_attr = (edge_attr_t*)malloc(roundup(MAX_EDGE * NUM_GRAPHS * sizeof(edge_attr_t), 128lu));
node_eigen_t* node_eigen = (node_eigen_t*)malloc(roundup(MAX_NODE * sizeof(node_eigen_t), 128lu));


const char* GNN_to_infer;
int GNN_instruction;
int nd_feature_table_v1[ND_FEATURE] = {119, 4, 12, 12, 10, 6, 6, 2, 2};
void read_instruction()
{
    std::ifstream in("Instruction.txt", std::ios_base::in);
    while(in >> GNN_instruction)
    {
        //std::cout <<"The instruction is " <<GNN_instruction <<std::endl;
    }
    if(GNN_instruction == 0)
    {
        GNN_to_infer = "GCN";
        //std::cout << "The GNN to be inferred is "<<GNN_to_infer<<std::endl;
    }
    else if(GNN_instruction == 1)
    {
        GNN_to_infer = "GIN";
        //std::cout << "The GNN to be inferred is "<<GNN_to_infer<<std::endl;
    }
    else if(GNN_instruction == 2)
    {
        GNN_to_infer = "PNA";
        //std::cout << "The GNN to be inferred is "<<GNN_to_infer<<std::endl;
    }
    else if(GNN_instruction == 3)
    {
        GNN_to_infer = "DGN";
        //std::cout << "The GNN to be inferred is " << GNN_to_infer << std::endl;
    }
}

int main()
{
    read_instruction();
    if(GNN_instruction == 0)
    {
        //printf("\n******* This is the C testbench for GCN model *******\n");
    }
    else if(GNN_instruction == 1)
    {
        //printf("\n******* This is the C testbench for GIN model *******\n");
    }
    else if(GNN_instruction == 2)
    {
        //printf("\n******* This is the C testbench for PNA model *******\n");
    }
    else if(GNN_instruction == 3)
    {
        //printf("\n******* This is the C testbench for DGN model *******\n");
    }

    load_weights(GNN_instruction);

    //std::cout << "Printing embedding_h_atom_embedding_list_weights" << std::endl;
    //for(int i = 0; i < ND_FEATURE; i++)
    //{
    //    for(int j = 0; j < nd_feature_table_v1[i]; j++)
    //    {
    //        for(int dim = 0; dim < EMB_DIM; dim++)
    //        {
    //            std::cout << node_embedding_h_atom_embedding_list_weight_fixed_DGN[i][j][dim] << std::endl;
    //        }
    //    }
    //}

    FM_TYPE all_results[NUM_GRAPHS][NUM_TASK];
    int nums_of_nodes[NUM_GRAPHS];
    int nums_of_edges[NUM_GRAPHS];
    int reload_weights[NUM_GRAPHS];
    int total_nodes = 0;
    int total_edges = 0;

    for (int g = 1; g <= NUM_GRAPHS; g++) {
        char info_file[128];
        int num_of_nodes;
        int num_of_edges;

        sprintf(info_file, GRAPH_INFO_FORMAT, g);

        FILE* f_info = fopen(info_file, "r");
        if (!f_info)
        {
            fprintf(stderr, "failed to open %s\n", info_file);
            exit(1);
        }
        fscanf(f_info, "%d\n%d", &num_of_nodes, &num_of_edges);
        fclose(f_info);

        nums_of_nodes[g - 1] = num_of_nodes;
        nums_of_edges[g - 1] = num_of_edges;
        reload_weights[g - 1] = g == 1;
        total_nodes += num_of_nodes;
        total_edges += num_of_edges;
    }

    int nodes_offset = 0;
    int edges_offset = 0;

    for (int g = 1; g <= NUM_GRAPHS; g++) {
        int num_of_nodes = nums_of_nodes[g - 1];
        int num_of_edges = nums_of_edges[g - 1];
        char graph_name[128];
        sprintf(graph_name, GRAPH_NAME_FORMAT, g);

        fetch_one_graph(
            g,
            graph_name,
            &node_feature[nodes_offset],
            &edge_list[edges_offset],
            &edge_attr[edges_offset],
            &node_eigen[nodes_offset],
            num_of_nodes,
            num_of_edges
        );

        nodes_offset += num_of_nodes;
        edges_offset += num_of_edges;
    }

    if(GNN_instruction == 0)
    {
        //printf("Computing GCN ...\n");
    }
    else if(GNN_instruction == 1)
    {
        //printf("Computing GIN ...\n");
    }
    else if(GNN_instruction == 2)
    {
        //printf("Computing PNA ...\n");
    }
    else if(GNN_instruction == 3)
    {
        //printf("Computing DGN");
    }

    if(GNN_instruction == 3)
    {
        GNN_compute_graphs(
            static_cast<Instruction>(GNN_instruction),
            NUM_GRAPHS,
            nums_of_nodes,
            nums_of_edges,
            reload_weights,
            all_results,
            node_feature,
            node_eigen,
            edge_list,
            //edge_attr,
            &node_embedding_h_atom_embedding_list_weight_fixed_DGN,
            //&edge_embedding_weight_fixed,
            //&GCN_convs_GIN_node_mlp_1_weight_fixed,
            &GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed,
            //&GIN_node_mlp_2_weight_fixed,
            &layers_posttrans_fully_connected_0_linear_weight_fixed,
            &GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed,
            &PNA_node_conv_weight_fixed,
            &bn_weight_PNA_graph_DGN_MLP_1_weight_fixed,
            &bn_bias_PNA_graph_DGN_MLP_1_bias_fixed,
            &bn_mean_PNA_graph_DGN_MLP_2_weight_fixed,
            &bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed,
            &graph_pred_PNA_graph_DGN_MLP_3_weight_fixed,
            &graph_pred_PNA_graph_DGN_MLP_3_bias_fixed,
            &GIN_node_mlp_eps_PNA_avg_deg_fixed
        );
    }
    else
    {
        GNN_compute_graphs(
            static_cast<Instruction>(GNN_instruction),
            NUM_GRAPHS,
            nums_of_nodes,
            nums_of_edges,
            reload_weights,
            all_results,
            node_feature,
            node_eigen,
            edge_list,
            //edge_attr,
            &node_embedding_h_atom_embedding_list_weight_fixed,
            //&edge_embedding_weight_fixed,
            //&GCN_convs_GIN_node_mlp_1_weight_fixed,
            &GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed,
            //&GIN_node_mlp_2_weight_fixed,
            &layers_posttrans_fully_connected_0_linear_weight_fixed,
            &GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed,
            &PNA_node_conv_weight_fixed,
            &bn_weight_PNA_graph_DGN_MLP_1_weight_fixed,
            &bn_bias_PNA_graph_DGN_MLP_1_bias_fixed,
            &bn_mean_PNA_graph_DGN_MLP_2_weight_fixed,
            &bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed,
            &graph_pred_PNA_graph_DGN_MLP_3_weight_fixed,
            &graph_pred_PNA_graph_DGN_MLP_3_bias_fixed,
            &GIN_node_mlp_eps_PNA_avg_deg_fixed
        );
    }


    FILE* c_output = fopen("C_sim_output.txt", "w+");
    for (int g = 1; g <= NUM_GRAPHS; g++) {
        int num_of_nodes = nums_of_nodes[g - 1];
        int num_of_edges = nums_of_edges[g - 1];
        char graph_name[128];
        sprintf(graph_name, GRAPH_NAME_FORMAT, g);

        //printf("********** Graph %s *************\n", graph_name);
        //printf("# of nodes: %d, # of edges: %d\n", num_of_nodes, num_of_edges);
        for (int t = 0; t < NUM_TASK; t++) {
            printf("%.7f\n", float(all_results[g - 1][t]));
            fprintf(c_output, "g%d: %.8f\n", g, float(all_results[g - 1][t]));
        }
    }
    fclose(c_output);
    return 0;
}