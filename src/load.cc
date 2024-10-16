#include <stdlib.h>
#include <stdio.h>
#include "testbench.h"

int nd_feature_table[ND_FEATURE] = {119, 5, 12, 12, 10, 6, 6, 2, 2};
int ed_feature_table[EDGE_ATTR] = {5, 6, 2};

//Edge and node embedding parameters
float node_embedding_h_atom_embedding_list_weight_float[9][ND_FEATURE_TOTAL][EMB_DIM];
float node_embedding_weight_float[ND_FEATURE_TOTAL][EMB_DIM];
float GCN_edge_embedding_weight_float[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
float GIN_edge_embedding_weight_float[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
float node_embedding_weight_raw[ND_FEATURE_TOTAL * EMB_DIM];
float node_embedding_weight_raw_PNA[ND_FEATURE_TOTAL * EMB_DIM];
float edge_embedding_weight_raw[NUM_LAYERS][ED_FEATURE_PER_LAYER * EMB_DIM];

//Model parameters - weights and biases
float GCN_convs_GIN_node_mlp_1_weight_float[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM];
float GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT];
float GIN_node_mlp_2_weight_float[NUM_LAYERS][EMB_DIM][DGN_LIN_GIN_MLP_1_OUT];
float layers_posttrans_fully_connected_0_linear_weight_float_in[4][EMB_DIM][2 * EMB_DIM];
float GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[NUM_LAYERS][EMB_DIM];
float PNA_node_conv_weight_float[DGN_PNA_NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM];
float bn_weight_PNA_graph_DGN_MLP_1_weight_float[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM];
float bn_bias_PNA_graph_DGN_MLP_1_bias_float[NUM_LAYERS][EMB_DIM];
float bn_mean_PNA_graph_DGN_MLP_2_weight_float[DGN_MLP_PNA_GRAPH_MLP_2_OUT][EMB_DIM];
float bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[NUM_LAYERS][EMB_DIM];
float graph_pred_PNA_graph_DGN_MLP_3_weight_float[NUM_TASK][EMB_DIM];
float graph_pred_PNA_graph_DGN_MLP_3_bias_float[NUM_TASK];
float GIN_node_mlp_eps_PNA_avg_deg_float[NUM_LAYERS];

float bn_mean_PNA_graph_DGN_MLP_2_weight_float_PNA[DGN_MLP_PNA_GRAPH_MLP_2_OUT][DGN_MLP_PNA_GRAPH_MLP_1_OUT];

WT_TYPE GCN_convs_GIN_node_mlp_1_weight_fixed[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT][EMB_DIM];
WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[NUM_LAYERS][DGN_LIN_GIN_MLP_1_OUT];
WT_TYPE GIN_node_mlp_2_weight_fixed[NUM_LAYERS][EMB_DIM][DGN_LIN_GIN_MLP_1_OUT];
WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[NUM_LAYERS][EMB_DIM];
WT_TYPE PNA_node_conv_weight_fixed[DGN_PNA_NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM];
WT_TYPE layers_posttrans_fully_connected_0_linear_weight_fixed [4][EMB_DIM][2 * EMB_DIM];
WT_TYPE bn_weight_PNA_graph_DGN_MLP_1_weight_fixed[DGN_MLP_PNA_GRAPH_MLP_1_OUT][EMB_DIM];
WT_TYPE bn_bias_PNA_graph_DGN_MLP_1_bias_fixed[NUM_LAYERS][EMB_DIM];
WT_TYPE bn_mean_PNA_graph_DGN_MLP_2_weight_fixed[DGN_MLP_PNA_GRAPH_MLP_2_OUT][EMB_DIM];
WT_TYPE bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed[NUM_LAYERS][EMB_DIM];
WT_TYPE node_embedding_h_atom_embedding_list_weight_fixed[9][ND_FEATURE_TOTAL][EMB_DIM];
WT_TYPE edge_embedding_weight_fixed[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[NUM_TASK][EMB_DIM];
WT_TYPE graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[NUM_TASK];
WT_TYPE GIN_node_mlp_eps_PNA_avg_deg_fixed[NUM_LAYERS];

void load_weights(int GNN_instruction)
{
    if(GNN_instruction == 0)
    {
        //printf("Loading weights for GCN ...\n");
    }
    else if(GNN_instruction == 1)
    {
        //printf("Loading weights for GIN ...\n");
    }
    else if(GNN_instruction == 2)
    {
        //printf("Loading weights for PNA ...\n");
    }

    FILE* f;

    if(GNN_instruction == 0)
    {
        f = fopen("GCN_weights_biases/gcn_ep1_dim32.weights.all.bin", "rb");

        fseek(f, 0*sizeof(float), SEEK_SET);
	    fread(node_embedding_weight_raw, sizeof(float), ND_FEATURE_TOTAL * EMB_DIM, f);

	    fseek(f, 5568*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_weight_float[0], sizeof(float), 1024, f);
	
	    fseek(f, 6592*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[0], sizeof(float), 32, f);

	    fseek(f, 6624*sizeof(float), SEEK_SET);
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[0], sizeof(float), 32, f);

	    fseek(f, 6656*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[0], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);

	    fseek(f, 7072*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_weight_float[1], sizeof(float), 1024, f);

	    fseek(f, 8096*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[1], sizeof(float), 32, f);

	    fseek(f, 8128*sizeof(float), SEEK_SET);
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[1], sizeof(float), 32, f);

	    fseek(f, 8160*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[1], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);

	    fseek(f, 8576*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_weight_float[2], sizeof(float), 1024, f);

	    fseek(f, 9600*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[2], sizeof(float), 32, f);

	    fseek(f, 9632*sizeof(float), SEEK_SET);
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[2], sizeof(float), 32, f);

	    fseek(f, 9664*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[2], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);

	    fseek(f, 10080*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_weight_float[3], sizeof(float), 1024, f);

	    fseek(f, 11104*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[3], sizeof(float), 32, f);

	    fseek(f, 11136*sizeof(float), SEEK_SET);
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[3], sizeof(float), 32, f);

	    fseek(f, 11168*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[3], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);
	
	    fseek(f, 11584*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_weight_float[4], sizeof(float), 1024, f);

	    fseek(f, 12608*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[4], sizeof(float), 32, f);

	    fseek(f, 12640*sizeof(float), SEEK_SET);
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[4], sizeof(float), 32, f);

	    fseek(f, 12672*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[4], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);
	

	    fseek(f, 13088*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_DGN_MLP_1_weight_float[0], sizeof(float), 32, f);

	    fseek(f, 13120*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_DGN_MLP_1_bias_float[0], sizeof(float), 32, f);

	    fseek(f, 13152*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_DGN_MLP_2_weight_float[0], sizeof(float), 32, f);

	    fseek(f, 13184*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[0], sizeof(float), 32, f);


	    fseek(f, 13217*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_DGN_MLP_1_weight_float[1], sizeof(float), 32, f);

	    fseek(f, 13249*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_DGN_MLP_1_bias_float[1], sizeof(float), 32, f);

	    fseek(f, 13281*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_DGN_MLP_2_weight_float[1], sizeof(float), 32, f);

	    fseek(f, 13313*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[1], sizeof(float), 32, f);

		
	    fseek(f, 13346*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_DGN_MLP_1_weight_float[2], sizeof(float), 32, f);

	    fseek(f, 13378*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_DGN_MLP_1_bias_float[2], sizeof(float), 32, f);

	    fseek(f, 13410*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_DGN_MLP_2_weight_float[2], sizeof(float), 32, f);

	    fseek(f, 13442*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[2], sizeof(float), 32, f);


	    fseek(f, 13475*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_DGN_MLP_1_weight_float[3], sizeof(float), 32, f);

	    fseek(f, 13507*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_DGN_MLP_1_bias_float[3], sizeof(float), 32, f);

	    fseek(f, 13539*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_DGN_MLP_2_weight_float[3], sizeof(float), 32, f);

	    fseek(f, 13571*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[3], sizeof(float), 32, f);
	

		fseek(f, 13604*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_DGN_MLP_1_weight_float[4], sizeof(float), 32, f);

	    fseek(f, 13636*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_DGN_MLP_1_bias_float[4], sizeof(float), 32, f);

	    fseek(f, 13668*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_DGN_MLP_2_weight_float[4], sizeof(float), 32, f);

	    fseek(f, 13700*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[4], sizeof(float), 32, f);
	

	    fseek(f, 13733*sizeof(float), SEEK_SET);
	    fread(graph_pred_PNA_graph_DGN_MLP_3_weight_float, sizeof(float), 32, f);

	    fseek(f, 13765*sizeof(float), SEEK_SET);
	    fread(graph_pred_PNA_graph_DGN_MLP_3_bias_float, sizeof(float), 1, f);

	    fclose(f);

        int idx = 0;
	    for(int i = 0; i < ND_FEATURE; i++) 
        {
		    int nd_f = nd_feature_table[i];
		    for(int j = 0; j < nd_f; j++) 
            {
			    for(int dim = 0; dim < EMB_DIM; dim++) 
                {
				node_embedding_weight_float[idx + j][dim] = node_embedding_weight_raw[(idx + j) * EMB_DIM + dim];
                node_embedding_h_atom_embedding_list_weight_float[0][idx + j][dim] = node_embedding_weight_raw[(idx + j) * EMB_DIM + dim];
			    }
		    }
		    idx += nd_f;
	    }

		
	    for(int l = 0; l < NUM_LAYERS; l++) 
        {
	    	idx = 0;
            int	idx2 = 0;
            for(int i = 0; i < EDGE_ATTR; i++) 
            {
                int ed_f = ed_feature_table[i];
                for(int j = 0; j < ed_f; j++) 
                {
                    for(int dim = 0; dim < EMB_DIM; dim++) 
                    {
                        GCN_edge_embedding_weight_float[l][idx + j][dim] = edge_embedding_weight_raw[l][(idx2 + j) * EMB_DIM + dim];
                    }
                }
                idx += ed_f;
                idx2 += ed_f;
            }
        }

        for(int nft = 0; nft < ND_FEATURE_TOTAL; nft++)
        {
            for(int dim = 0; dim < EMB_DIM; dim++)
            {
                node_embedding_h_atom_embedding_list_weight_fixed[0][nft][dim] = (WT_TYPE)node_embedding_h_atom_embedding_list_weight_float[0][nft][dim];
                //std::cout << node_embedding_h_atom_embedding_list_weight_fixed[0][nft][dim] << std::endl;
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int ed_f = 0; ed_f < ED_FEATURE_PER_LAYER; ed_f++)
            {
                for(int dim = 0; dim < EMB_DIM; dim++)
                {
                    edge_embedding_weight_fixed[l][ed_f][dim] = (WT_TYPE)GCN_edge_embedding_weight_float[l][ed_f][dim];
                    //std::cout << edge_embedding_weight_fixed[l][ed_f][dim] << std::endl;
                }
            }
        }
        
        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[l][dim_out] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[l][dim_out];
                //std::cout << GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[l][dim_out] << std::endl;
                for(int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                {
                    GCN_convs_GIN_node_mlp_1_weight_fixed[l][dim_out][dim_in] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_weight_float[l][dim_out][dim_in];
                    //std::cout << GCN_convs_GIN_node_mlp_1_weight_fixed[l][dim_out][dim_in] << std::endl;
                }
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                bn_weight_PNA_graph_DGN_MLP_1_weight_fixed[l][dim_out] = (WT_TYPE)bn_weight_PNA_graph_DGN_MLP_1_weight_float[l][dim_out];
                bn_bias_PNA_graph_DGN_MLP_1_bias_fixed[l][dim_out] = (WT_TYPE)bn_bias_PNA_graph_DGN_MLP_1_bias_float[l][dim_out];
                bn_mean_PNA_graph_DGN_MLP_2_weight_fixed[l][dim_out] = (WT_TYPE)bn_mean_PNA_graph_DGN_MLP_2_weight_float[l][dim_out];
                bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed[l][dim_out] = (WT_TYPE)bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[l][dim_out];
                GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[l][dim_out] = (WT_TYPE)GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[l][dim_out];
            }
        }

        for(int t = 0; t < NUM_TASK; t++)
        {
            graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[t] = graph_pred_PNA_graph_DGN_MLP_3_bias_float[t];
            for(int dim = 0; dim < EMB_DIM; dim++)
            {
                graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[t][dim] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_weight_float[t][dim];
            }
        }
    }
    else if(GNN_instruction == 1)
    {
        f = fopen("GIN_weights_biases/gin_ep1_mlp_1_weights_dim32.bin", "r");
        fread(GCN_convs_GIN_node_mlp_1_weight_float, sizeof(float), NUM_LAYERS * DGN_LIN_GIN_MLP_1_OUT * EMB_DIM, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_mlp_1_bias_dim32.bin", "r");
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float, sizeof(float), NUM_LAYERS * DGN_LIN_GIN_MLP_1_OUT, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_mlp_2_weights_dim32.bin", "r");
        fread(GIN_node_mlp_2_weight_float, sizeof(float), NUM_LAYERS * EMB_DIM * DGN_LIN_GIN_MLP_1_OUT, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_mlp_2_bias_dim32.bin", "r");
        fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float, sizeof(float), NUM_LAYERS * EMB_DIM, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_eps_dim32.bin", "r");
        fread(GIN_node_mlp_eps_PNA_avg_deg_float, sizeof(float), NUM_LAYERS, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_nd_embed_dim32.bin", "r");
        fread(node_embedding_weight_float, sizeof(float), ND_FEATURE_TOTAL * EMB_DIM, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_ed_embed_dim32.bin", "r");
        fread(GIN_edge_embedding_weight_float, sizeof(float), NUM_LAYERS * ED_FEATURE_PER_LAYER * EMB_DIM, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_pred_weights_dim32.bin", "r");
        fread(graph_pred_PNA_graph_DGN_MLP_3_weight_float, sizeof(float), NUM_TASK * EMB_DIM, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_pred_bias_dim32.bin", "r");
        fread(graph_pred_PNA_graph_DGN_MLP_3_bias_float, sizeof(float), NUM_TASK, f);
        fclose(f);

        for(int nft = 0; nft < ND_FEATURE_TOTAL; nft++)
        {
            for(int dim = 0; dim < EMB_DIM; dim++)
            {
                node_embedding_h_atom_embedding_list_weight_fixed[0][nft][dim] = (WT_TYPE)node_embedding_weight_float[nft][dim];
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int ed_f = 0; ed_f < ED_FEATURE_PER_LAYER; ed_f++)
            {
                for(int dim = 0; dim < EMB_DIM; dim++)
                {
                    edge_embedding_weight_fixed[l][ed_f][dim] = (WT_TYPE)GIN_edge_embedding_weight_float[l][ed_f][dim];
                }
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            GIN_node_mlp_eps_PNA_avg_deg_fixed[l] = (WT_TYPE)GIN_node_mlp_eps_PNA_avg_deg_float[l];
            for(int dim_out = 0; dim_out < DGN_LIN_GIN_MLP_1_OUT; dim_out++)
            {
                GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[l][dim_out] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[l][dim_out];
                for(int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                {
                    GCN_convs_GIN_node_mlp_1_weight_fixed[l][dim_out][dim_in] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_weight_float[l][dim_out][dim_in];
                }
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[l][dim_out] = (WT_TYPE)GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[l][dim_out];
                for(int dim_in = 0; dim_in < DGN_LIN_GIN_MLP_1_OUT; dim_in++)
                {
                    GIN_node_mlp_2_weight_fixed[l][dim_out][dim_in] = (WT_TYPE)GIN_node_mlp_2_weight_float[l][dim_out][dim_in];
                }
            }
        }

        for(int t = 0; t < NUM_TASK; t++)
        {
            graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[t] = graph_pred_PNA_graph_DGN_MLP_3_bias_float[t];
            for(int dim = 0; dim < EMB_DIM; dim++)
            {
                graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[t][dim] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_weight_float[t][dim];
            }
        }

    }
    else if(GNN_instruction == 2)
    {
        f = fopen("PNA_weights_biases/pna_ep1_noBN_dim32.weights.all.bin", "rb");

        fseek(f, 0*sizeof(float), SEEK_SET);
        fread(node_embedding_weight_raw_PNA, sizeof(float), ND_FEATURE_TOTAL * EMB_DIM, f);

        fseek(f, 5568*sizeof(float), SEEK_SET);
        fread(PNA_node_conv_weight_float[0], sizeof(float), 12288, f);

        fseek(f, 17856*sizeof(float), SEEK_SET);
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[0], sizeof(float), 32 , f);

        fseek(f, 17888*sizeof(float), SEEK_SET);
        fread(PNA_node_conv_weight_float[1], sizeof(float), 12288 , f);

        fseek(f, 30176*sizeof(float), SEEK_SET);
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[1], sizeof(float), 32, f);

        fseek(f, 30208*sizeof(float), SEEK_SET);
        fread(PNA_node_conv_weight_float[2], sizeof(float), 12288, f);

        fseek(f, 42496*sizeof(float), SEEK_SET);
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[2], sizeof(float), 32, f);

        fseek(f, 42528*sizeof(float), SEEK_SET);
        fread(PNA_node_conv_weight_float[3], sizeof(float), 12288, f);

        fseek(f, 54816*sizeof(float), SEEK_SET);
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[3], sizeof(float), 32, f);

        fseek(f, 54848*sizeof(float), SEEK_SET);
        fread(bn_weight_PNA_graph_DGN_MLP_1_weight_float, sizeof(float), 512, f);

        fseek(f, 55360*sizeof(float), SEEK_SET);
        fread(bn_bias_PNA_graph_DGN_MLP_1_bias_float, sizeof(float), 16, f);

        fseek(f, 55376*sizeof(float), SEEK_SET);
        fread(bn_mean_PNA_graph_DGN_MLP_2_weight_float_PNA, sizeof(float), 128, f);

        fseek(f, 55504*sizeof(float), SEEK_SET);
        fread(bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float, sizeof(float), 8, f);

        fseek(f, 55512*sizeof(float), SEEK_SET);
        fread(graph_pred_PNA_graph_DGN_MLP_3_weight_float, sizeof(float), 8, f);

        fseek(f, 55520*sizeof(float), SEEK_SET);
        fread(graph_pred_PNA_graph_DGN_MLP_3_bias_float, sizeof(float), 1, f);

        fclose(f);

        //converting into fixed type
        int idx = 0;
        for(int i = 0; i < ND_FEATURE; i++)
        {
            int nd_f = nd_feature_table[i];
            for(int j = 0; j < nd_f; j++)
            {
                for(int dim = 0; dim < EMB_DIM; dim++)
                {
                    node_embedding_weight_float[idx + j][dim] = (WT_TYPE)node_embedding_weight_raw_PNA[(idx + j) * EMB_DIM + dim];
                }
            }
            idx += nd_f;
        }

        for(int ntf = 0; ntf < ND_FEATURE_TOTAL; ntf++)
        {
            for(int dim = 0; dim < EMB_DIM; dim++)
            {
                node_embedding_h_atom_embedding_list_weight_fixed[0][ntf][dim] = (WT_TYPE)node_embedding_weight_float[ntf][dim];
            }
        }

        for(int l = 0; l < DGN_PNA_NUM_LAYERS; l++)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                for(int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                {
                    for(int scaler = 0; scaler < NUM_SCALERS; scaler++)
                    {
                        for(int aggr = 0; aggr < NUM_AGGRS; aggr++)
                        {
                            PNA_node_conv_weight_fixed[l][dim_out][scaler][aggr][dim_in] = (WT_TYPE)PNA_node_conv_weight_float[l][dim_out][scaler][aggr][dim_in];
                        }
                    }
                }
            }
        }

        for(int l = 0; l < DGN_PNA_NUM_LAYERS; l++)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[l][dim_out] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[l][dim_out];
            }
        }

        for(int dim_out = 0; dim_out < DGN_MLP_PNA_GRAPH_MLP_1_OUT; dim_out ++)
        {
            bn_bias_PNA_graph_DGN_MLP_1_bias_fixed[0][dim_out] = (WT_TYPE)bn_bias_PNA_graph_DGN_MLP_1_bias_float[0][dim_out];
            for(int dim_in = 0; dim_in < EMB_DIM; dim_in++)
            {
                bn_weight_PNA_graph_DGN_MLP_1_weight_fixed[dim_out][dim_in] = (WT_TYPE)bn_weight_PNA_graph_DGN_MLP_1_weight_float[dim_out][dim_in];
            }
        }

        for(int dim_out = 0; dim_out < DGN_MLP_PNA_GRAPH_MLP_2_OUT; dim_out++)
        {
            bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed[0][dim_out] = (WT_TYPE)bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[0][dim_out];
            for(int dim_in = 0; dim_in < DGN_MLP_PNA_GRAPH_MLP_1_OUT; dim_in++)
            {
                bn_mean_PNA_graph_DGN_MLP_2_weight_fixed[dim_out][dim_in] = (WT_TYPE)bn_mean_PNA_graph_DGN_MLP_2_weight_float_PNA[dim_out][dim_in];
            }
        }

        for(int t = 0; t < NUM_TASK; t++)
        {
            graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[t] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_bias_float[t];
            for(int dim_in = 0; dim_in < DGN_MLP_PNA_GRAPH_MLP_2_OUT; dim_in++)
            {
                graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[t][dim_in] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_weight_float[t][dim_in];
            }
        }
        GIN_node_mlp_eps_PNA_avg_deg_fixed[0] = 6.885701656341553;
    }
    else if (GNN_instruction == 3)
    {
        f = fopen("DGN_weights_biases/dgn_ep1_noBN_dim32.weights.all.bin", "rb");
        fseek(f, 0*sizeof(float), SEEK_SET);
        float *embedding_h_atom_embedding_list_0_weight_float = new float[3808];
        fread(embedding_h_atom_embedding_list_0_weight_float, sizeof(float), 3808, f);
        for (int i = 0; i < 3808; i++) node_embedding_h_atom_embedding_list_weight_fixed[0][i / 32][i % 32] = WT_TYPE(embedding_h_atom_embedding_list_0_weight_float[i]);
        delete[] embedding_h_atom_embedding_list_0_weight_float;

        fseek(f, 3808*sizeof(float), SEEK_SET);
        float *embedding_h_atom_embedding_list_1_weight_float = new float[160];
        fread(embedding_h_atom_embedding_list_1_weight_float, sizeof(float), 160, f);
        for (int i = 0; i < 160; i++) node_embedding_h_atom_embedding_list_weight_fixed[1][i / 32][i % 32] = WT_TYPE(embedding_h_atom_embedding_list_1_weight_float[i]);
        delete[] embedding_h_atom_embedding_list_1_weight_float;

        fseek(f, 3968*sizeof(float), SEEK_SET);
        float *embedding_h_atom_embedding_list_2_weight_float = new float[384];
        fread(embedding_h_atom_embedding_list_2_weight_float, sizeof(float), 384, f);
        for (int i = 0; i < 384; i++) node_embedding_h_atom_embedding_list_weight_fixed[2][i / 32][i % 32] = WT_TYPE(embedding_h_atom_embedding_list_2_weight_float[i]);
        delete[] embedding_h_atom_embedding_list_2_weight_float;

        fseek(f, 4352*sizeof(float), SEEK_SET);
        float *embedding_h_atom_embedding_list_3_weight_float = new float[384];
        fread(embedding_h_atom_embedding_list_3_weight_float, sizeof(float), 384, f);
        for (int i = 0; i < 384; i++) node_embedding_h_atom_embedding_list_weight_fixed[3][i / 32][i % 32] = WT_TYPE(embedding_h_atom_embedding_list_3_weight_float[i]);
        delete[] embedding_h_atom_embedding_list_3_weight_float;

        fseek(f, 4736*sizeof(float), SEEK_SET);
        float *embedding_h_atom_embedding_list_4_weight_float = new float[3200];
        fread(embedding_h_atom_embedding_list_4_weight_float, sizeof(float), 3200, f);
        for (int i = 0; i < 3200; i++) node_embedding_h_atom_embedding_list_weight_fixed[4][i / 32][i % 32] = WT_TYPE(embedding_h_atom_embedding_list_4_weight_float[i]);
        delete[] embedding_h_atom_embedding_list_4_weight_float;

        fseek(f, 7936*sizeof(float), SEEK_SET);
        float *embedding_h_atom_embedding_list_5_weight_float = new float[192];
        fread(embedding_h_atom_embedding_list_5_weight_float, sizeof(float), 192, f);
        for (int i = 0; i < 192; i++) node_embedding_h_atom_embedding_list_weight_fixed[5][i / 32][i % 32] = WT_TYPE(embedding_h_atom_embedding_list_5_weight_float[i]);
        delete[] embedding_h_atom_embedding_list_5_weight_float;

        fseek(f, 8128*sizeof(float), SEEK_SET);
        float *embedding_h_atom_embedding_list_6_weight_float = new float[192];
        fread(embedding_h_atom_embedding_list_6_weight_float, sizeof(float), 192, f);
        for (int i = 0; i < 192; i++) node_embedding_h_atom_embedding_list_weight_fixed[6][i / 32][i % 32] = WT_TYPE(embedding_h_atom_embedding_list_6_weight_float[i]);
        delete[] embedding_h_atom_embedding_list_6_weight_float;

        fseek(f, 8320*sizeof(float), SEEK_SET);
        float *embedding_h_atom_embedding_list_7_weight_float = new float[64];
        fread(embedding_h_atom_embedding_list_7_weight_float, sizeof(float), 64, f);
        for (int i = 0; i < 64; i++) node_embedding_h_atom_embedding_list_weight_fixed[7][i / 32][i % 32] = WT_TYPE(embedding_h_atom_embedding_list_7_weight_float[i]);
        delete[] embedding_h_atom_embedding_list_7_weight_float;

        fseek(f, 8384*sizeof(float), SEEK_SET);
        float *embedding_h_atom_embedding_list_8_weight_float = new float[64];
        fread(embedding_h_atom_embedding_list_8_weight_float, sizeof(float), 64, f);
        for (int i = 0; i < 64; i++) node_embedding_h_atom_embedding_list_weight_fixed[8][i / 32][i % 32] = WT_TYPE(embedding_h_atom_embedding_list_8_weight_float[i]);
        delete[] embedding_h_atom_embedding_list_8_weight_float;

        fseek(f, 8448*sizeof(float), SEEK_SET);
        float *layers_0_posttrans_fully_connected_0_linear_weight_float = new float[2048];
        fread(layers_0_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 2048, f);
        for (int i = 0; i < 2048; i++) layers_posttrans_fully_connected_0_linear_weight_fixed[0][i / 64][i % 64] = WT_TYPE(layers_0_posttrans_fully_connected_0_linear_weight_float[i]);
        delete[] layers_0_posttrans_fully_connected_0_linear_weight_float;

        fseek(f, 10496*sizeof(float), SEEK_SET);
        float *layers_0_posttrans_fully_connected_0_linear_bias_float = new float[32];
        fread(layers_0_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 32, f);
        for (int i = 0; i < 32; i++) GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[0][i] = WT_TYPE(layers_0_posttrans_fully_connected_0_linear_bias_float[i]);
        delete[] layers_0_posttrans_fully_connected_0_linear_bias_float;


        fseek(f, 10528*sizeof(float), SEEK_SET);
        float *layers_1_posttrans_fully_connected_0_linear_weight_float = new float[2048];
        fread(layers_1_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 2048, f);
        for (int i = 0; i < 2048; i++) layers_posttrans_fully_connected_0_linear_weight_fixed[1][i / 64][i % 64] = WT_TYPE(layers_1_posttrans_fully_connected_0_linear_weight_float[i]);
        delete[] layers_1_posttrans_fully_connected_0_linear_weight_float;

        fseek(f, 12576*sizeof(float), SEEK_SET);
        float *layers_1_posttrans_fully_connected_0_linear_bias_float = new float[32];
        fread(layers_1_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 32, f);
        for (int i = 0; i < 32; i++) GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[1][i] = WT_TYPE(layers_1_posttrans_fully_connected_0_linear_bias_float[i]);
        delete[] layers_1_posttrans_fully_connected_0_linear_bias_float;

        fseek(f, 12608*sizeof(float), SEEK_SET);
        float *layers_2_posttrans_fully_connected_0_linear_weight_float = new float[2048];
        fread(layers_2_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 2048, f);
        for (int i = 0; i < 2048; i++) layers_posttrans_fully_connected_0_linear_weight_fixed[2][i / 64][i % 64] = WT_TYPE(layers_2_posttrans_fully_connected_0_linear_weight_float[i]);
        delete[] layers_2_posttrans_fully_connected_0_linear_weight_float;

        fseek(f, 14656*sizeof(float), SEEK_SET);
        float *layers_2_posttrans_fully_connected_0_linear_bias_float = new float[32];
        fread(layers_2_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 32, f);
        for (int i = 0; i < 32; i++) GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[2][i] = WT_TYPE(layers_2_posttrans_fully_connected_0_linear_bias_float[i]);
        delete[] layers_2_posttrans_fully_connected_0_linear_bias_float;

        fseek(f, 14688*sizeof(float), SEEK_SET);
        float *layers_3_posttrans_fully_connected_0_linear_weight_float = new float[2048];
        fread(layers_3_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 2048, f);
        for (int i = 0; i < 2048; i++) layers_posttrans_fully_connected_0_linear_weight_fixed[3][i / 64][i % 64] = WT_TYPE(layers_3_posttrans_fully_connected_0_linear_weight_float[i]);
        delete[] layers_3_posttrans_fully_connected_0_linear_weight_float;

        fseek(f, 16736*sizeof(float), SEEK_SET);
        float *layers_3_posttrans_fully_connected_0_linear_bias_float = new float[32];
        fread(layers_3_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 32, f);
        for (int i = 0; i < 32; i++) GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[3][i] = WT_TYPE(layers_3_posttrans_fully_connected_0_linear_bias_float[i]);
        delete[] layers_3_posttrans_fully_connected_0_linear_bias_float;

        fseek(f, 16768*sizeof(float), SEEK_SET);
        float *MLP_layer_FC_layers_0_weight_float = new float[512];
        fread(MLP_layer_FC_layers_0_weight_float, sizeof(float), 512, f);
        for (int i = 0; i < 512; i++) bn_weight_PNA_graph_DGN_MLP_1_weight_fixed[i / 32][i % 32] = WT_TYPE(MLP_layer_FC_layers_0_weight_float[i]);
        delete[] MLP_layer_FC_layers_0_weight_float;

        fseek(f, 17280*sizeof(float), SEEK_SET);
        float *MLP_layer_FC_layers_0_bias_float = new float[16];
        fread(MLP_layer_FC_layers_0_bias_float, sizeof(float), 16, f);
        for (int i = 0; i < 16; i++) bn_bias_PNA_graph_DGN_MLP_1_bias_fixed[0][i] = WT_TYPE(MLP_layer_FC_layers_0_bias_float[i]);
        delete[] MLP_layer_FC_layers_0_bias_float;

        fseek(f, 17296*sizeof(float), SEEK_SET);
        float *MLP_layer_FC_layers_1_weight_float = new float[128];
        fread(MLP_layer_FC_layers_1_weight_float, sizeof(float), 128, f);
        for (int i = 0; i < 128; i++) bn_mean_PNA_graph_DGN_MLP_2_weight_fixed[i / 16][i % 16] = WT_TYPE(MLP_layer_FC_layers_1_weight_float[i]);
        delete[] MLP_layer_FC_layers_1_weight_float;

        fseek(f, 17424*sizeof(float), SEEK_SET);
        float *MLP_layer_FC_layers_1_bias_float = new float[8];
        fread(MLP_layer_FC_layers_1_bias_float, sizeof(float), 8, f);
        for (int i = 0; i < 8; i++) bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed[0][i] = WT_TYPE(MLP_layer_FC_layers_1_bias_float[i]);
        delete[] MLP_layer_FC_layers_1_bias_float;

        fseek(f, 17432*sizeof(float), SEEK_SET);
        float *MLP_layer_FC_layers_2_weight_float = new float[8];
        fread(MLP_layer_FC_layers_2_weight_float, sizeof(float), 8, f);
        for (int i = 0; i < 8; i++) graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[i / 8][i % 8] = WT_TYPE(MLP_layer_FC_layers_2_weight_float[i]);
        delete[] MLP_layer_FC_layers_2_weight_float;

        fseek(f, 17440*sizeof(float), SEEK_SET);
        float *MLP_layer_FC_layers_2_bias_float = new float[1];
        fread(MLP_layer_FC_layers_2_bias_float, sizeof(float), 1, f);
        for (int i = 0; i < 1; i++) graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[i] = WT_TYPE(MLP_layer_FC_layers_2_bias_float[i]);
        delete[] MLP_layer_FC_layers_2_bias_float;

        fclose(f);

    }
}

void fetch_one_graph(
    int g,
    char* graph_name,
    node_feature_t* node_feature,
    edge_t* edge_list,
    edge_attr_t* edge_attr,
    int num_of_nodes,
    int num_of_edges
)
{
    //printf("(%d/%d) Loading graph %s ...\n", g, NUM_GRAPHS, graph_name);
    FILE* f;

    char f_node_feature[128];
    char f_edge_list[128];
    char f_edge_attr[128];

    sprintf(f_node_feature, "%s_node_feature.bin", graph_name);
    sprintf(f_edge_list, "%s_edge_list.bin", graph_name);
    sprintf(f_edge_attr, "%s_edge_attr.bin", graph_name);

    f = fopen(f_node_feature, "rb");
    if (!f)
    {
        fprintf(stderr, "failed to open %s\n", f_node_feature);
        exit(1);
    }
    fread(node_feature, sizeof(node_feature_t), num_of_nodes, f);
    fclose(f);

    f = fopen(f_edge_list, "rb");
    if (!f)
    {
        fprintf(stderr, "failed to open %s\n", f_edge_list);
        exit(1);
    }
    fread(edge_list, sizeof(edge_t), num_of_edges, f);
    fclose(f);

    f = fopen(f_edge_attr, "rb");
    if (!f)
    {
        fprintf(stderr, "failed to open %s\n", f_edge_attr);
        exit(1);
    }
    fread(edge_attr, sizeof(edge_attr_t), num_of_edges, f);
    fclose(f);
}
