#include "host.h"
#include "xcl2.hpp"
#include <fstream>
#include <string>
#include <iostream>

aligned_vector<WT_TYPE> node_embedding_h_atom_embedding_list_weight_fixed(ND_FEATURE * ND_FEATURE_TOTAL * EMB_DIM);
aligned_vector<WT_TYPE> edge_embedding_weight_fixed(NUM_LAYERS * ED_FEATURE_PER_LAYER * EMB_DIM);
aligned_vector<WT_TYPE> GCN_convs_GIN_node_mlp_1_weight_fixed(NUM_LAYERS * DGN_LIN_GIN_MLP_1_OUT * EMB_DIM);
aligned_vector<WT_TYPE> GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed(NUM_LAYERS * DGN_LIN_GIN_MLP_1_OUT);
aligned_vector<WT_TYPE> GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed(NUM_LAYERS * EMB_DIM);
aligned_vector<WT_TYPE> GIN_node_mlp_2_weight_fixed(NUM_LAYERS * EMB_DIM * DGN_LIN_GIN_MLP_1_OUT);
aligned_vector<WT_TYPE> PNA_node_conv_weight_fixed(NUM_LAYERS * EMB_DIM * NUM_SCALERS * NUM_AGGRS * EMB_DIM);
aligned_vector<WT_TYPE> layers_posttrans_fully_connected_0_linear_weight_fixed(DGN_PNA_NUM_LAYERS * EMB_DIM * 2 * EMB_DIM);
aligned_vector<WT_TYPE> bn_weight_PNA_graph_DGN_MLP_1_weight_fixed(DGN_MLP_PNA_GRAPH_MLP_1_OUT * EMB_DIM);
aligned_vector<WT_TYPE> bn_bias_PNA_graph_DGN_MLP_1_bias_fixed(NUM_LAYERS * EMB_DIM);
aligned_vector<WT_TYPE> bn_mean_PNA_graph_DGN_MLP_2_weight_fixed(DGN_MLP_PNA_GRAPH_MLP_2_OUT * EMB_DIM);
aligned_vector<WT_TYPE> bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed(NUM_LAYERS * EMB_DIM);
aligned_vector<WT_TYPE> graph_pred_PNA_graph_DGN_MLP_3_weight_fixed(NUM_TASK * EMB_DIM);
aligned_vector<WT_TYPE> graph_pred_PNA_graph_DGN_MLP_3_bias_fixed(NUM_TASK);
aligned_vector<WT_TYPE> GIN_node_mlp_eps_PNA_avg_deg_fixed(NUM_LAYERS);

int GNN_instruction;

const char* GNN_to_infer;
static const char* GRAPH_INFO_FORMAT = "../graphs/graph_info/g%d_info.txt";
static const char* GRAPH_NAME_FORMAT = "../graphs/graph_bin/g%d";

void read_instruction()
{
    std::ifstream in("./Instruction.txt", std::ios::in);
    while(in >> GNN_instruction)
    {
        std::cout << "The instruction is " << GNN_instruction << std::endl;
    }
    switch(GNN_instruction)
    {
        case 0 :    GNN_to_infer = "GCN";
                    break;
        
        case 1 :    GNN_to_infer = "GIN";
                    break;
        
        case 2 :    GNN_to_infer = "PNA";
                    break;
        
        case 3:     GNN_to_infer = "DGN";
                    break;
        
        default:    break;
    }
    std::cout << "The GNN to be inferred is " << GNN_to_infer << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_GNN_compute_graphs;
    cl::CommandQueue q;

    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err,
                  q = cl::CommandQueue(
                      context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i
                  << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i
                      << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_GNN_compute_graphs = cl::Kernel(program, "GNN_compute_graphs", &err));
            valid_device = true;
            break;
        }
    }
    if (!valid_device) {
	    std::cout << "Failed to program any device found, exit!\n";
	    exit(EXIT_FAILURE);
    }

    read_instruction();

    switch(GNN_instruction)
    {
        case 0 :    printf("\n******* This is the HLS for GCN model *******\n");
                    break;
        
        case 1 :    printf("\n******* This is the HLS for GIN model *******\n");
                    break;
        
        case 2 :    printf("\n******* This is the HLS for PNA model *******\n");
                    break;
        
        case 3 :    printf("\n******* This is the HLS for DGN model *******\n");
        
        default  :  break;
    }

    load_weights(GNN_instruction);
    printf("\n******* Weights loading done *******\n");

    OCL_CHECK(err, cl::Buffer node_embedding_h_atom_embedding_list_weight_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        ND_FEATURE * ND_FEATURE_TOTAL * EMB_DIM * sizeof(WT_TYPE),
        node_embedding_h_atom_embedding_list_weight_fixed.data(),
        &err));
    
    OCL_CHECK(err, cl::Buffer edge_embedding_weight_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * ED_FEATURE_PER_LAYER * EMB_DIM * sizeof(WT_TYPE),
        edge_embedding_weight_fixed.data(),
        &err));

    OCL_CHECK(err, cl::Buffer GCN_convs_GIN_node_mlp_1_weight_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * DGN_LIN_GIN_MLP_1_OUT * EMB_DIM * sizeof(WT_TYPE),
        GCN_convs_GIN_node_mlp_1_weight_fixed.data(),
        &err));

    OCL_CHECK(err, cl::Buffer GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * DGN_LIN_GIN_MLP_1_OUT * sizeof(WT_TYPE),
        GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed.data(),
        &err));
    
    OCL_CHECK(err, cl::Buffer GIN_node_mlp_2_weight_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * EMB_DIM * DGN_LIN_GIN_MLP_1_OUT * sizeof(WT_TYPE),
        GIN_node_mlp_2_weight_fixed.data(),
        &err));

    OCL_CHECK(err, cl::Buffer GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * EMB_DIM * sizeof(WT_TYPE),
        GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed.data(),
        &err));
    
    OCL_CHECK(err, cl::Buffer PNA_node_conv_weight_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * EMB_DIM * NUM_SCALERS * NUM_AGGRS * EMB_DIM * sizeof(WT_TYPE),
        PNA_node_conv_weight_fixed.data(),
        &err));
    
    OCL_CHECK(err, cl::Buffer layers_posttrans_fully_connected_0_linear_weight_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        DGN_PNA_NUM_LAYERS * EMB_DIM * 2 * EMB_DIM * sizeof(WT_TYPE),
        layers_posttrans_fully_connected_0_linear_weight_fixed.data(),
        &err));

    OCL_CHECK(err, cl::Buffer bn_weight_PNA_graph_DGN_MLP_1_weight_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        DGN_MLP_PNA_GRAPH_MLP_1_OUT * EMB_DIM * sizeof(WT_TYPE),
        bn_weight_PNA_graph_DGN_MLP_1_weight_fixed.data(),
        &err));
    
    OCL_CHECK(err, cl::Buffer bn_bias_PNA_graph_DGN_MLP_1_bias_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * EMB_DIM * sizeof(WT_TYPE),
        bn_bias_PNA_graph_DGN_MLP_1_bias_fixed.data(),
        &err));
    
     OCL_CHECK(err, cl::Buffer bn_mean_PNA_graph_DGN_MLP_2_weight_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        DGN_MLP_PNA_GRAPH_MLP_2_OUT * EMB_DIM * sizeof(WT_TYPE),
        bn_mean_PNA_graph_DGN_MLP_2_weight_fixed.data(),
        &err));
    
    OCL_CHECK(err, cl::Buffer bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * EMB_DIM * sizeof(WT_TYPE),
        bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed.data(),
        &err));
    
    OCL_CHECK(err, cl::Buffer graph_pred_PNA_graph_DGN_MLP_3_weight_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_TASK * EMB_DIM * sizeof(WT_TYPE),
        graph_pred_PNA_graph_DGN_MLP_3_weight_fixed.data(),
        &err));

    OCL_CHECK(err, cl::Buffer graph_pred_PNA_graph_DGN_MLP_3_bias_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_TASK * sizeof(WT_TYPE),
        graph_pred_PNA_graph_DGN_MLP_3_bias_fixed.data(),
        &err));

    OCL_CHECK(err, cl::Buffer GIN_node_mlp_eps_PNA_avg_deg_buf(context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_LAYERS * sizeof(WT_TYPE),
        GIN_node_mlp_eps_PNA_avg_deg_fixed.data(),
        &err));
    
    int num_of_graphs = NUM_GRAPHS;
    aligned_vector<int> nums_of_nodes(NUM_GRAPHS);
    aligned_vector<int> nums_of_edges(NUM_GRAPHS);
    aligned_vector<int> reload_weights(NUM_GRAPHS);
    aligned_vector<FM_TYPE> result(NUM_GRAPHS * NUM_TASK);
    aligned_vector<node_feature_t> node_feature;
    aligned_vector<edge_t> edge_list;
    aligned_vector<edge_attr_t> edge_attr;
    aligned_vector<node_eigen_t> node_eigen;

    for (int g = 1; g <= NUM_GRAPHS; g++)
    {
        char graph_name[128];
        char info_file[128];
        int num_of_nodes;
        int num_of_edges;

        sprintf(info_file, GRAPH_INFO_FORMAT, g);
        sprintf(graph_name, GRAPH_NAME_FORMAT, g);

        FILE* f_info = fopen(info_file, "r");
        fscanf(f_info, "%d\n%d", &num_of_nodes, &num_of_edges);
        fclose(f_info);

        nums_of_nodes[g - 1] = num_of_nodes;
        nums_of_edges[g - 1] = num_of_edges;
        reload_weights[g - 1] = g == 1;

        fetch_one_graph(g, graph_name, node_feature, edge_list, edge_attr, node_eigen, num_of_nodes, num_of_edges);
    }
    printf("\nFinished Loading Graphs.......\n");

     OCL_CHECK(err, cl::Buffer nums_of_nodes_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_GRAPHS * sizeof(int),
        nums_of_nodes.data(),
        &err));

    OCL_CHECK(err, cl::Buffer nums_of_edges_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_GRAPHS * sizeof(int),
        nums_of_edges.data(),
        &err));

    OCL_CHECK(err, cl::Buffer reload_weights_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_GRAPHS * sizeof(int),
        reload_weights.data(),
        &err));

    OCL_CHECK(err, cl::Buffer result_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        NUM_GRAPHS * NUM_TASK * sizeof(FM_TYPE),
        result.data(),
        &err));

    OCL_CHECK(err, cl::Buffer node_feature_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        node_feature.size() * sizeof(node_feature_t),
        node_feature.data(),
        &err));

    OCL_CHECK(err, cl::Buffer edge_list_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        edge_list.size() * sizeof(edge_t),
        edge_list.data(),
        &err));

    OCL_CHECK(err, cl::Buffer edge_attr_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        edge_attr.size() * sizeof(edge_attr_t),
        edge_attr.data(),
        &err));
    
    OCL_CHECK(err, cl::Buffer node_eigen_buf(
        context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        node_eigen.size() * sizeof(node_eigen_t),
        node_eigen.data(),
        &err));
    
    printf("Finished Loading Buffers......\n");
    
    int idx = 0;

    int instruction_in = GNN_instruction;

    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, num_of_graphs));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, nums_of_nodes_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, nums_of_edges_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, reload_weights_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, result_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, node_feature_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, node_eigen_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, edge_list_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, node_embedding_h_atom_embedding_list_weight_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, layers_posttrans_fully_connected_0_linear_weight_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, PNA_node_conv_weight_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, bn_weight_PNA_graph_DGN_MLP_1_weight_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, bn_bias_PNA_graph_DGN_MLP_1_bias_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, bn_mean_PNA_graph_DGN_MLP_2_weight_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, graph_pred_PNA_graph_DGN_MLP_3_weight_buf));
    OCL_CHECK(err, err = krnl_GNN_compute_graphs.setArg(idx++, graph_pred_PNA_graph_DGN_MLP_3_bias_buf));

    if (err != CL_SUCCESS) 
    {
        std::cerr << "Error setting kernel argument 0: " << err << " (" << getCLErrorString(err) << ")" << std::endl;
    }

    for (int i = 0; i < NUM_TRIALS; i++)
    {
       switch(GNN_instruction)
       {
            case 0 : printf("(%d/%d) Computing GCN ...\r", i + 1, NUM_TRIALS);
                     break;

            case 1 : printf("(%d/%d) Computing GIN ...\r", i + 1, NUM_TRIALS);
                     break;
            
            case 2 : printf("(%d/%d) Computing PNA ...\r", i + 1, NUM_TRIALS);
                     break;
            
            case 3  : printf("(%d/%d) Computing DGN ...\r", i + 1, NUM_TRIALS);
                      break;
            
            default : break;

       }
        fflush(stdout);
        OCL_CHECK(err, err = q.enqueueTask(krnl_GNN_compute_graphs));
        if (err != CL_SUCCESS) 
        {
            std::cerr << err << " (" << getCLErrorString(err) << ")" << std::endl;
            std::cerr << "Error 1" << std::endl;
        }
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({result_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
        if (err != CL_SUCCESS) 
        {
            std::cerr << err << " (" << getCLErrorString(err) << ")" << std::endl;
            std::cerr << "Error 2" << std::endl;
        }
        OCL_CHECK(err, err = q.finish());
        if (err != CL_SUCCESS) 
        {
            std::cerr << err << " (" << getCLErrorString(err) << ")" << std::endl;
            std::cerr << "Error 3" << std::endl;
        }
    }
    printf("\n******* Computation done *******\n");

    FILE* c_output = fopen("HLS_output.txt", "w+");
    for (int g = 1; g <= NUM_GRAPHS; g++) {
        int num_of_nodes = nums_of_nodes[g - 1];
        int num_of_edges = nums_of_edges[g - 1];
        char graph_name[128];
        sprintf(graph_name, GRAPH_NAME_FORMAT, g);
        for (int t = 0; t < NUM_TASK; t++) {
            fprintf(c_output, "g%d: %.8f\n", g, float(result[(g - 1) * NUM_TASK + t]));
            printf("g%d: %.8f\n", g, float(result[(g - 1) * NUM_TASK + t]));
        }
    }
    return 0;
}