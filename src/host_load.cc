#include <stdlib.h>
#include <stdio.h>
#include "host.h"

int nd_feature_table[ND_FEATURE] = {119, 5, 12, 12, 10, 6, 6, 2, 2};
int ed_feature_table[EDGE_ATTR] = {5, 6, 2};
//Edge and node embedding parameters
float node_embedding_h_atom_embedding_list_weight_float[ND_FEATURE][ND_FEATURE_TOTAL][EMB_DIM];
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

float embedding_h_atom_embedding_list_0_weight_float[119 * EMB_DIM];
float embedding_h_atom_embedding_list_1_weight_float[4 * EMB_DIM];
float embedding_h_atom_embedding_list_2_weight_float[12 * EMB_DIM];
float embedding_h_atom_embedding_list_3_weight_float[12 * EMB_DIM];
float embedding_h_atom_embedding_list_4_weight_float[10 * EMB_DIM];
float embedding_h_atom_embedding_list_5_weight_float[6 * EMB_DIM];
float embedding_h_atom_embedding_list_6_weight_float[6 * EMB_DIM];
float embedding_h_atom_embedding_list_7_weight_float[2 * EMB_DIM];
float embedding_h_atom_embedding_list_8_weight_float[2 * EMB_DIM];
float layers_0_posttrans_fully_connected_0_linear_weight_float[2 * EMB_DIM * EMB_DIM];
float layers_1_posttrans_fully_connected_0_linear_weight_float[2 * EMB_DIM * EMB_DIM];
float layers_2_posttrans_fully_connected_0_linear_weight_float[2 * EMB_DIM * EMB_DIM];
float layers_3_posttrans_fully_connected_0_linear_weight_float[2 * EMB_DIM * EMB_DIM];
float layers_0_posttrans_fully_connected_0_linear_bias_float[EMB_DIM];
float layers_1_posttrans_fully_connected_0_linear_bias_float[EMB_DIM];
float layers_2_posttrans_fully_connected_0_linear_bias_float[EMB_DIM];
float layers_3_posttrans_fully_connected_0_linear_bias_float[EMB_DIM];

void load_weights(int GNN_instruction)
{
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

        for(int ndf = 0; ndf < 1; ndf++)
        {
            for(int nft = 0; nft < ND_FEATURE_TOTAL; nft++)
            {
                for(int dim = 0; dim < EMB_DIM; dim++)
                {
                    node_embedding_h_atom_embedding_list_weight_fixed[(ndf * ND_FEATURE_TOTAL * EMB_DIM) + (nft * EMB_DIM) + dim] = (WT_TYPE)node_embedding_h_atom_embedding_list_weight_float[ndf][nft][dim];
                }
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int ed_f = 0; ed_f < ED_FEATURE_PER_LAYER; ed_f++)
            {
                for(int dim = 0; dim < EMB_DIM; dim++)
                {
                    edge_embedding_weight_fixed[l * ED_FEATURE_PER_LAYER * EMB_DIM + ed_f * EMB_DIM + dim] = (WT_TYPE)GCN_edge_embedding_weight_float[l][ed_f][dim];
                }
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[l * DGN_LIN_GIN_MLP_1_OUT + dim_out] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[l][dim_out];
                for(int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                {
                    GCN_convs_GIN_node_mlp_1_weight_fixed[l * DGN_LIN_GIN_MLP_1_OUT * EMB_DIM + dim_out * EMB_DIM + dim_in] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_weight_float[l][dim_out][dim_in];
                }
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                bn_weight_PNA_graph_DGN_MLP_1_weight_fixed[l * EMB_DIM + dim_out] = (WT_TYPE)bn_weight_PNA_graph_DGN_MLP_1_weight_float[l][dim_out];
                bn_bias_PNA_graph_DGN_MLP_1_bias_fixed[l * EMB_DIM + dim_out] = (WT_TYPE)bn_bias_PNA_graph_DGN_MLP_1_bias_float[l][dim_out];
                bn_mean_PNA_graph_DGN_MLP_2_weight_fixed[l * EMB_DIM + dim_out] = (WT_TYPE)bn_mean_PNA_graph_DGN_MLP_2_weight_float[l][dim_out];
                bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed[l * EMB_DIM + dim_out] = (WT_TYPE)bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[l][dim_out];
                GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[l * EMB_DIM + dim_out] = (WT_TYPE)GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[l][dim_out];
            }
        }

        for(int t = 0; t < NUM_TASK; t++)
        {
            graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[t] = graph_pred_PNA_graph_DGN_MLP_3_bias_float[t];
            for(int dim = 0; dim < EMB_DIM; dim++)
            {
                graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[t * EMB_DIM + dim] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_weight_float[t][dim];
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

        for(int ndf = 0; ndf < 1; ndf++)
        {
            for(int nft = 0; nft < ND_FEATURE_TOTAL; nft++)
            {
                for(int dim = 0; dim < EMB_DIM; dim++)
                {
                    node_embedding_h_atom_embedding_list_weight_fixed[ndf * ND_FEATURE_TOTAL * EMB_DIM + nft * EMB_DIM + dim] = (WT_TYPE)node_embedding_weight_float[nft][dim];
                }
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int ed_f = 0; ed_f < ED_FEATURE_PER_LAYER; ed_f++)
            {
                for(int dim = 0; dim < EMB_DIM; dim++)
                {
                    edge_embedding_weight_fixed[l * ED_FEATURE_PER_LAYER * EMB_DIM + ed_f * EMB_DIM + dim] = (WT_TYPE)GIN_edge_embedding_weight_float[l][ed_f][dim];
                }
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            GIN_node_mlp_eps_PNA_avg_deg_fixed[l] = (WT_TYPE)GIN_node_mlp_eps_PNA_avg_deg_float[l];
            for(int dim_out = 0; dim_out < DGN_LIN_GIN_MLP_1_OUT; dim_out++)
            {
                GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[l * DGN_LIN_GIN_MLP_1_OUT + dim_out] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[l][dim_out];
                for(int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                {
                    GCN_convs_GIN_node_mlp_1_weight_fixed[l * DGN_LIN_GIN_MLP_1_OUT * EMB_DIM + dim_out * EMB_DIM + dim_in] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_weight_float[l][dim_out][dim_in];
                }
            }
        }

        for(int l = 0; l < NUM_LAYERS; l++)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[l * EMB_DIM + dim_out] = (WT_TYPE)GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[l][dim_out];
                for(int dim_in = 0; dim_in < DGN_LIN_GIN_MLP_1_OUT; dim_in++)
                {
                    GIN_node_mlp_2_weight_fixed[l * EMB_DIM * DGN_LIN_GIN_MLP_1_OUT + dim_out * DGN_LIN_GIN_MLP_1_OUT + dim_in] = (WT_TYPE)GIN_node_mlp_2_weight_float[l][dim_out][dim_in];
                }
            }
        }

        for(int t = 0; t < NUM_TASK; t++)
        {
            graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[t] = graph_pred_PNA_graph_DGN_MLP_3_bias_float[t];
            for(int dim = 0; dim < EMB_DIM; dim++)
            {
                graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[t * EMB_DIM + dim] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_weight_float[t][dim];
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

        for(int ndf = 0; ndf < 1; ndf++)
        {
            for(int ntf = 0; ntf < ND_FEATURE_TOTAL; ntf++)
            {
                for(int dim = 0; dim < EMB_DIM; dim++)
                {
                    node_embedding_h_atom_embedding_list_weight_fixed[ndf * ND_FEATURE_TOTAL * EMB_DIM + ntf * EMB_DIM + dim] = (WT_TYPE)node_embedding_weight_float[ntf][dim];
                }
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
                            PNA_node_conv_weight_fixed[(l * EMB_DIM * NUM_SCALERS * NUM_AGGRS * EMB_DIM) + (dim_out * NUM_SCALERS * NUM_AGGRS * EMB_DIM) + (scaler * NUM_AGGRS * EMB_DIM) + (aggr * EMB_DIM) + dim_in] = (WT_TYPE)PNA_node_conv_weight_float[l][dim_out][scaler][aggr][dim_in];
                        }
                    }
                }
            }
        }

        for(int l = 0; l < DGN_PNA_NUM_LAYERS; l++)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[l * DGN_LIN_GIN_MLP_1_OUT + dim_out] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[l][dim_out];
            }
        }

        for(int dim_out = 0; dim_out < DGN_MLP_PNA_GRAPH_MLP_1_OUT; dim_out ++)
        {
            bn_bias_PNA_graph_DGN_MLP_1_bias_fixed[dim_out] = (WT_TYPE)bn_bias_PNA_graph_DGN_MLP_1_bias_float[0][dim_out];
            for(int dim_in = 0; dim_in < EMB_DIM; dim_in++)
            {
                bn_weight_PNA_graph_DGN_MLP_1_weight_fixed[dim_out * EMB_DIM + dim_in] = (WT_TYPE)bn_weight_PNA_graph_DGN_MLP_1_weight_float[dim_out][dim_in];
            }
        }

        for(int dim_out = 0; dim_out < DGN_MLP_PNA_GRAPH_MLP_2_OUT; dim_out++)
        {
            bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed[dim_out] = (WT_TYPE)bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[0][dim_out];
            for(int dim_in = 0; dim_in < DGN_MLP_PNA_GRAPH_MLP_1_OUT; dim_in++)
            {
                bn_mean_PNA_graph_DGN_MLP_2_weight_fixed[dim_out * EMB_DIM + dim_in] = (WT_TYPE)bn_mean_PNA_graph_DGN_MLP_2_weight_float_PNA[dim_out][dim_in];
            }
        }

        for(int t = 0; t < NUM_TASK; t++)
        {
            graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[t] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_bias_float[t];
            for(int dim_in = 0; dim_in < DGN_MLP_PNA_GRAPH_MLP_2_OUT; dim_in++)
            {
                graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[t * EMB_DIM + dim_in] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_weight_float[t][dim_in];
            }
        }
        GIN_node_mlp_eps_PNA_avg_deg_fixed[0] = 6.885701656341553;
    }
    else if (GNN_instruction == 3)
    {
        f = fopen("/usr/scratch/pguruprasanna3/FlowGNN/DGN/dgn_ep1_noBN_dim100.weights.all.bin", "rb");
	    fseek(f, 0*sizeof(float), SEEK_SET);	fseek(f, 0*sizeof(float), SEEK_SET);
	    fread(embedding_h_atom_embedding_list_0_weight_float, sizeof(float), 11900, f);

	    fseek(f, 11900*sizeof(float), SEEK_SET);
	    fread(embedding_h_atom_embedding_list_1_weight_float, sizeof(float), 400, f);

	    fseek(f, 12300*sizeof(float), SEEK_SET);
	    fread(embedding_h_atom_embedding_list_2_weight_float, sizeof(float), 1200, f);

	    fseek(f, 13500*sizeof(float), SEEK_SET);
	    fread(embedding_h_atom_embedding_list_3_weight_float, sizeof(float), 1200, f);

	    fseek(f, 14700*sizeof(float), SEEK_SET);
	    fread(embedding_h_atom_embedding_list_4_weight_float, sizeof(float), 1000, f);

	    fseek(f, 15700*sizeof(float), SEEK_SET);
	    fread(embedding_h_atom_embedding_list_5_weight_float, sizeof(float), 600, f);

	    fseek(f, 16300*sizeof(float), SEEK_SET);
	    fread(embedding_h_atom_embedding_list_6_weight_float, sizeof(float), 600, f);

	    fseek(f, 16900*sizeof(float), SEEK_SET);
	    fread(embedding_h_atom_embedding_list_7_weight_float, sizeof(float), 200, f);

	    fseek(f, 17100*sizeof(float), SEEK_SET);
	    fread(embedding_h_atom_embedding_list_8_weight_float, sizeof(float), 200, f);

	    fseek(f, 17300*sizeof(float), SEEK_SET);
	    fread(layers_0_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 20000, f);

	    fseek(f, 37300*sizeof(float), SEEK_SET);
	    fread(layers_0_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 100, f);

	
	    fseek(f, 37400*sizeof(float), SEEK_SET);
	    fread(layers_1_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 20000, f);

	    fseek(f, 57400*sizeof(float), SEEK_SET);
	    fread(layers_1_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 100, f);
	
	    fseek(f, 57500*sizeof(float), SEEK_SET);
	    fread(layers_2_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 20000, f);

	    fseek(f, 77500*sizeof(float), SEEK_SET);
	    fread(layers_2_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 100, f);
	
	    fseek(f, 77600*sizeof(float), SEEK_SET);
	    fread(layers_3_posttrans_fully_connected_0_linear_weight_float, sizeof(float), 20000, f);

	    fseek(f, 97600*sizeof(float), SEEK_SET);
	    fread(layers_3_posttrans_fully_connected_0_linear_bias_float, sizeof(float), 100, f);

	    fseek(f, 97700*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_DGN_MLP_1_weight_float, sizeof(float), 5000, f);

	    fseek(f, 102700*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_DGN_MLP_1_bias_float, sizeof(float), 50, f);
	
	    fseek(f, 102750*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_DGN_MLP_2_weight_float_PNA, sizeof(float), 1250, f);

	    fseek(f, 104000*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float, sizeof(float), 25, f);

	    fseek(f, 104025*sizeof(float), SEEK_SET);
	    fread(graph_pred_PNA_graph_DGN_MLP_3_weight_float, sizeof(float), 25, f);

	    fseek(f, 104050*sizeof(float), SEEK_SET);
	    fread(graph_pred_PNA_graph_DGN_MLP_3_bias_float, sizeof(float), 1, f);

	    fclose(f);

        for(int i = 0; i < ND_FEATURE; i++)
	    {
		    for(int j = 0; j < nd_feature_table[i]; j++)
		    {
		    	for(int dim = 0; dim < EMB_DIM; dim++)
		    	{
		    		if(i == 0)
		    		{	
		    			node_embedding_h_atom_embedding_list_weight_float[i][j][dim] = embedding_h_atom_embedding_list_0_weight_float[j * EMB_DIM + dim];
		    		}
		    		else if(i == 1)
		    		{
		    			node_embedding_h_atom_embedding_list_weight_float[i][j][dim] = embedding_h_atom_embedding_list_1_weight_float[j * EMB_DIM + dim];
		    		}
		    		else if(i == 2)
		    		{
		    			node_embedding_h_atom_embedding_list_weight_float[i][j][dim] = embedding_h_atom_embedding_list_2_weight_float[j * EMB_DIM + dim];
		    		}
		    		else if(i == 3)
		    		{
		    			node_embedding_h_atom_embedding_list_weight_float[i][j][dim] = embedding_h_atom_embedding_list_3_weight_float[j * EMB_DIM + dim];
		    		}
		    		else if(i == 4)
		    		{
		    			node_embedding_h_atom_embedding_list_weight_float[i][j][dim] = embedding_h_atom_embedding_list_4_weight_float[j * EMB_DIM + dim];
		    		}
		    		else if(i == 5)
		    		{
		    			node_embedding_h_atom_embedding_list_weight_float[i][j][dim] = embedding_h_atom_embedding_list_5_weight_float[j * EMB_DIM + dim];
		    		}
		    		else if(i == 6)
		    		{	
		    			node_embedding_h_atom_embedding_list_weight_float[i][j][dim] = embedding_h_atom_embedding_list_6_weight_float[j * EMB_DIM + dim];
		    		}
		    		else if(i == 7)
		    		{
		    			node_embedding_h_atom_embedding_list_weight_float[i][j][dim] = embedding_h_atom_embedding_list_7_weight_float[j * EMB_DIM + dim];
		    		}
		    		else if(i == 8)
		    		{
		    			node_embedding_h_atom_embedding_list_weight_float[i][j][dim] = embedding_h_atom_embedding_list_8_weight_float[j * EMB_DIM + dim];
		    		}
		    	}
		    }
	    }

        for(int i = 0; i < ND_FEATURE; i++)
	    {
		    for(int j = 0; j < nd_feature_table[i]; j++)
		    {
		    	for(int dim = 0; dim < EMB_DIM; dim++)
		    	{
		    		node_embedding_h_atom_embedding_list_weight_fixed[i * nd_feature_table[i] * EMB_DIM + j * EMB_DIM + dim] = (WT_TYPE)node_embedding_h_atom_embedding_list_weight_float[i][j][dim];

		    	}
		    }
	    }

        for(int i = 0; i < NUM_LAYERS; i++)
	    {
		    for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
		    {
		    	for(int dim_in = 0; dim_in < 2 * EMB_DIM; dim_in++)
		    	{
		    		if(i == 0)
		    		{
		    			layers_posttrans_fully_connected_0_linear_weight_float_in[i][dim_out][dim_in] = layers_0_posttrans_fully_connected_0_linear_weight_float[dim_out * 2 * EMB_DIM + dim_in];
		    		}
		    		else if(i == 1)
		    		{
		    			layers_posttrans_fully_connected_0_linear_weight_float_in[i][dim_out][dim_in] = layers_1_posttrans_fully_connected_0_linear_weight_float[dim_out * 2 * EMB_DIM + dim_in];
		    		}
		    		else if(i == 2)
		    		{
		    			layers_posttrans_fully_connected_0_linear_weight_float_in[i][dim_out][dim_in] = layers_2_posttrans_fully_connected_0_linear_weight_float[dim_out * 2 * EMB_DIM + dim_in];
		    		}
		    		else if(i == 3)
		    		{
		    			layers_posttrans_fully_connected_0_linear_weight_float_in[i][dim_out][dim_in] = layers_3_posttrans_fully_connected_0_linear_weight_float[dim_out * 2 * EMB_DIM + dim_in];
		    		}
		    	}
		    }
	    }

        for(int i = 0; i < NUM_LAYERS; i++)
	    {
	    	for(int dim_out = 0; dim_out < EMB_DIM; dim_out++)
	    	{
	    		for(int dim_in = 0; dim_in < 2 * EMB_DIM; dim_in++)
	    		{
	    			layers_posttrans_fully_connected_0_linear_weight_fixed[i * EMB_DIM * 2 * EMB_DIM + dim_out * 2 * EMB_DIM + dim_in] = (WT_TYPE)layers_posttrans_fully_connected_0_linear_weight_float_in[i][dim_out][dim_in];
    
	    		}
	    	}
	    }

        for(int i = 0; i < NUM_LAYERS; i++)
	    {
	    	for(int dim = 0; dim < EMB_DIM; dim++)
	    	{
	    		if(i == 0)
	    		{
	    			GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[i][dim] = layers_0_posttrans_fully_connected_0_linear_bias_float[dim];
	    		}
	    		else if(i == 1)
	    		{
	    			GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[i][dim] = layers_1_posttrans_fully_connected_0_linear_bias_float[dim];
	    		}
	    		else if(i == 2)
	    		{
	    			GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[i][dim] = layers_2_posttrans_fully_connected_0_linear_bias_float[dim];
	    		}
	    		else if(i == 3)
	    		{
	    			GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[i][dim] = layers_3_posttrans_fully_connected_0_linear_bias_float[dim];
	    		}
	    	}
	    }

        for(int i = 0; i < NUM_LAYERS; i++)
	    {
	    	for(int dim = 0; dim < EMB_DIM; dim++)
	    	{
	    		GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_fixed[i * EMB_DIM + dim] = (WT_TYPE)GCN_convs_root_emb_weight_GIN_node_mlp_2_LPFC_0_linear_bias_float[i][dim];
	    	}
	    }

        for(int i = 0; i < EMB_DIM / 2; i++)
	    {
	    	bn_bias_PNA_graph_DGN_MLP_1_bias_fixed[i] = (WT_TYPE)bn_bias_PNA_graph_DGN_MLP_1_bias_float[0][i];
	    	for(int dim = 0; dim < EMB_DIM; dim++)
	    	{
	    		bn_weight_PNA_graph_DGN_MLP_1_weight_fixed[i * EMB_DIM + dim] = (WT_TYPE)bn_weight_PNA_graph_DGN_MLP_1_weight_float[i][dim];
	    	}
	    }

        for(int i = 0; i < EMB_DIM / 4; i++)
	    {
	    	bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_fixed[i] = (WT_TYPE)bn_sqrt_var_PNA_graph_DGN_MLP_2_bias_float[0][i];
	    	for(int dim = 0; dim < EMB_DIM / 2; dim++)
	    	{
	    		bn_mean_PNA_graph_DGN_MLP_2_weight_fixed[i * EMB_DIM / 2 + dim] = (WT_TYPE)bn_mean_PNA_graph_DGN_MLP_2_weight_float_PNA[i][dim];
	    	}
	    }

        for(int i = 0; i < 1; i++)
	    {
	    	graph_pred_PNA_graph_DGN_MLP_3_bias_fixed[i] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_bias_float[i];
	    	for(int dim = 0; dim < EMB_DIM / 4; dim++)
	    	{
	    		graph_pred_PNA_graph_DGN_MLP_3_weight_fixed[i * EMB_DIM / 4 + dim] = (WT_TYPE)graph_pred_PNA_graph_DGN_MLP_3_weight_float[i][dim];
	    	}
	    }
    }
}

void fetch_one_graph(
    int g,
    char* graph_name,
    aligned_vector<node_feature_t>& node_feature, 
    aligned_vector<edge_t>& edge_list,
    aligned_vector<edge_attr_t>& edge_attr,
    aligned_vector<node_eigen_t>& node_eigen,
    int num_of_nodes, 
    int num_of_edges
)
{
    printf("(%d/%d) Loading graph %s ...\r", g, NUM_GRAPHS, graph_name);
    fflush(stdout);

    FILE* f;

    char f_node_feature[128];
    char f_edge_list[128];
    char f_edge_attr[128];
    char f_node_eigen[128];

    sprintf(f_node_feature, "%s_node_feature.bin", graph_name);
    sprintf(f_edge_list, "%s_edge_list.bin", graph_name);
    sprintf(f_edge_attr, "%s_edge_attr.bin", graph_name);
    sprintf(f_node_eigen, "eig/g%d.txt", g);

    f = fopen(f_node_feature, "rb");
    size_t node_feature_start = node_feature.size();
    node_feature.resize(node_feature_start + num_of_nodes);
    node_feature_t* node_feature_ptr = &node_feature.data()[node_feature_start];
    fread(node_feature_ptr, sizeof(node_feature_t), num_of_nodes, f);
    fclose(f);

    f = fopen(f_edge_list, "rb");
    size_t edge_list_start = edge_list.size();
    edge_list.resize(edge_list_start + num_of_edges);
    edge_t* edge_list_ptr = &edge_list.data()[edge_list_start];
    fread(edge_list_ptr, sizeof(edge_t), num_of_edges, f);
    fclose(f);

    f = fopen(f_edge_attr, "rb");
    size_t edge_attr_start = edge_attr.size();
    edge_attr.resize(edge_attr_start + num_of_edges);
    edge_attr_t* edge_attr_ptr = &edge_attr.data()[edge_attr_start];
    fread(edge_attr_ptr, sizeof(edge_attr_t), num_of_edges, f);
    fclose(f);

    f = fopen(f_node_eigen, "r");
    size_t node_eigen_start = node_eigen.size();
    node_eigen.resize(node_eigen_start + num_of_nodes);
    node_eigen_t* node_eigen_ptr = &node_eigen.data()[node_eigen_start];
    float node_eigen_float[4];
    fscanf(f, "tensor([[%e, %e,%e,%e],\n", &node_eigen_float[0], &node_eigen_float[1], &node_eigen_float[2], &node_eigen_float[3]);
    for (int i = 0; i < 4; i++) node_eigen_ptr[0][i] = WT_TYPE(node_eigen_float[i]);
    for (int nd = 1; nd < num_of_nodes - 1; nd++)
    {
        fscanf(f, "[%e, %e,%e,%e],\n", &node_eigen_float[0], &node_eigen_float[1], &node_eigen_float[2], &node_eigen_float[3]);
        for (int i = 0; i < 4; i++) node_eigen_ptr[nd][i] = WT_TYPE(node_eigen_float[i]);
    }
    fscanf(f, "[%e, %e,%e,%e]])", &node_eigen_float[0], &node_eigen_float[1], &node_eigen_float[2], &node_eigen_float[3]);
    for (int i = 0; i < 4; i++) node_eigen_ptr[num_of_nodes - 1][i] = WT_TYPE(node_eigen_float[i]);
    fclose(f);
}