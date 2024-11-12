#include <stdlib.h>
#include <stdio.h>
#include "testbench.h"

int nd_feature_table[ND_FEATURE] = {119, 5, 12, 12, 10, 6, 6, 2, 2};
int ed_feature_table[EDGE_ATTR] = {5, 6, 2};

float node_embedding_weight_float[ND_FEATURE_TOTAL][EMB_DIM];

float GCN_edge_embedding_weight_float[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
float GIN_edge_embedding_weight_float[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];

float node_embedding_weight_raw[ND_FEATURE_TOTAL * EMB_DIM];
float node_embedding_weight_raw_PNA[ND_FEATURE_TOTAL * PNA_EMB_DIM];
float edge_embedding_weight_raw[NUM_LAYERS][ED_FEATURE_PER_LAYER * EMB_DIM];

float GCN_convs_GIN_node_mlp_1_weight_float[NUM_LAYERS][GIN_MLP_1_OUT][EMB_DIM];
float GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[NUM_LAYERS][GIN_MLP_1_OUT];
float GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_float[NUM_LAYERS][EMB_DIM];
float PNA_node_conv_weights_float[PNA_NUM_LAYERS][PNA_EMB_DIM][NUM_SCALERS][NUM_AGGRS][PNA_EMB_DIM];

float GIN_node_mlp_eps_PNA_avg_deg_float[NUM_LAYERS];
float GIN_node_mlp_2_weights_float[NUM_LAYERS][EMB_DIM][GIN_MLP_1_OUT];

float bn_sqrt_var_PNA_graph_mlp_2_bias_float[NUM_LAYERS][EMB_DIM];
float bn_weight_PNA_graph_mlp_1_weights_float[PNA_GRAPH_MLP_1_OUT][EMB_DIM];
float bn_bias_PNA_graph_mlp_1_bias_float[NUM_LAYERS][EMB_DIM];
float bn_mean_PNA_graph_mlp_2_weights_float[PNA_GRAPH_MLP_2_OUT][EMB_DIM];

float graph_pred_PNA_graph_mlp_3_weights_float[NUM_TASK][EMB_DIM];
float graph_pred_PNA_graph_mlp_3_bias_float[NUM_TASK];

float bn_weight_PNA_graph_mlp_1_weights_float_PNA[PNA_GRAPH_MLP_1_OUT][PNA_EMB_DIM];
float bn_mean_PNA_graph_mlp_2_weights_float_PNA[PNA_GRAPH_MLP_2_OUT][PNA_GRAPH_MLP_1_OUT];

WT_TYPE GCN_convs_GIN_node_mlp_1_weight_fixed[NUM_LAYERS][GIN_MLP_1_OUT][EMB_DIM];
WT_TYPE GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[NUM_LAYERS][GIN_MLP_1_OUT];
WT_TYPE GIN_node_mlp_2_weights_fixed[NUM_LAYERS][EMB_DIM][GIN_MLP_1_OUT];
WT_TYPE GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_fixed[NUM_LAYERS][EMB_DIM];
WT_TYPE PNA_node_conv_weights_fixed[PNA_NUM_LAYERS][PNA_EMB_DIM][NUM_SCALERS][NUM_AGGRS][PNA_EMB_DIM];
WT_TYPE bn_weight_PNA_graph_mlp_1_weights_fixed[PNA_GRAPH_MLP_1_OUT][EMB_DIM];
WT_TYPE bn_bias_PNA_graph_mlp_1_bias_fixed[NUM_LAYERS][EMB_DIM];
WT_TYPE bn_mean_PNA_graph_mlp_2_weights_fixed[PNA_GRAPH_MLP_2_OUT][EMB_DIM];
WT_TYPE bn_sqrt_var_PNA_graph_mlp_2_bias_fixed[NUM_LAYERS][EMB_DIM];
WT_TYPE node_embedding_weight_fixed[ND_FEATURE_TOTAL][EMB_DIM];
WT_TYPE edge_embedding_weight_fixed[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
WT_TYPE graph_pred_PNA_graph_mlp_3_weights_fixed[NUM_TASK][EMB_DIM];
WT_TYPE graph_pred_PNA_graph_mlp_3_bias_fixed[NUM_TASK];
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
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_float[0], sizeof(float), 32, f);

	    fseek(f, 6656*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[0], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);

	    fseek(f, 7072*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_weight_float[1], sizeof(float), 1024, f);

	    fseek(f, 8096*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[1], sizeof(float), 32, f);

	    fseek(f, 8128*sizeof(float), SEEK_SET);
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_float[1], sizeof(float), 32, f);

	    fseek(f, 8160*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[1], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);

	    fseek(f, 8576*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_weight_float[2], sizeof(float), 1024, f);

	    fseek(f, 9600*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[2], sizeof(float), 32, f);

	    fseek(f, 9632*sizeof(float), SEEK_SET);
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_float[2], sizeof(float), 32, f);

	    fseek(f, 9664*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[2], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);

	    fseek(f, 10080*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_weight_float[3], sizeof(float), 1024, f);

	    fseek(f, 11104*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[3], sizeof(float), 32, f);

	    fseek(f, 11136*sizeof(float), SEEK_SET);
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_float[3], sizeof(float), 32, f);

	    fseek(f, 11168*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[3], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);
	
	    fseek(f, 11584*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_weight_float[4], sizeof(float), 1024, f);

	    fseek(f, 12608*sizeof(float), SEEK_SET);
	    fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[4], sizeof(float), 32, f);

	    fseek(f, 12640*sizeof(float), SEEK_SET);
	    fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_float[4], sizeof(float), 32, f);

	    fseek(f, 12672*sizeof(float), SEEK_SET);
	    fread(edge_embedding_weight_raw[4], sizeof(float), ED_FEATURE_PER_LAYER * EMB_DIM, f);
	

	    fseek(f, 13088*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_mlp_1_weights_float[0], sizeof(float), 32, f);

	    fseek(f, 13120*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_mlp_1_bias_float[0], sizeof(float), 32, f);

	    fseek(f, 13152*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_mlp_2_weights_float[0], sizeof(float), 32, f);

	    fseek(f, 13184*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_mlp_2_bias_float[0], sizeof(float), 32, f);


	    fseek(f, 13217*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_mlp_1_weights_float[1], sizeof(float), 32, f);

	    fseek(f, 13249*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_mlp_1_bias_float[1], sizeof(float), 32, f);

	    fseek(f, 13281*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_mlp_2_weights_float[1], sizeof(float), 32, f);

	    fseek(f, 13313*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_mlp_2_bias_float[1], sizeof(float), 32, f);

		
	    fseek(f, 13346*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_mlp_1_weights_float[2], sizeof(float), 32, f);

	    fseek(f, 13378*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_mlp_1_bias_float[2], sizeof(float), 32, f);

	    fseek(f, 13410*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_mlp_2_weights_float[2], sizeof(float), 32, f);

	    fseek(f, 13442*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_mlp_2_bias_float[2], sizeof(float), 32, f);


	    fseek(f, 13475*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_mlp_1_weights_float[3], sizeof(float), 32, f);

	    fseek(f, 13507*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_mlp_1_bias_float[3], sizeof(float), 32, f);

	    fseek(f, 13539*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_mlp_2_weights_float[3], sizeof(float), 32, f);

	    fseek(f, 13571*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_mlp_2_bias_float[3], sizeof(float), 32, f);
	

		fseek(f, 13604*sizeof(float), SEEK_SET);
	    fread(bn_weight_PNA_graph_mlp_1_weights_float[4], sizeof(float), 32, f);

	    fseek(f, 13636*sizeof(float), SEEK_SET);
	    fread(bn_bias_PNA_graph_mlp_1_bias_float[4], sizeof(float), 32, f);

	    fseek(f, 13668*sizeof(float), SEEK_SET);
	    fread(bn_mean_PNA_graph_mlp_2_weights_float[4], sizeof(float), 32, f);

	    fseek(f, 13700*sizeof(float), SEEK_SET);
	    fread(bn_sqrt_var_PNA_graph_mlp_2_bias_float[4], sizeof(float), 32, f);
	

	    fseek(f, 13733*sizeof(float), SEEK_SET);
	    fread(graph_pred_PNA_graph_mlp_3_weights_float, sizeof(float), 32, f);

	    fseek(f, 13765*sizeof(float), SEEK_SET);
	    fread(graph_pred_PNA_graph_mlp_3_bias_float, sizeof(float), 1, f);

	    fclose(f);

        int idx = 0;
	    for(int i = 0; i < ND_FEATURE; i++) {
		    int nd_f = nd_feature_table[i];
		    for(int j = 0; j < nd_f; j++) {
			    for(int dim = 0; dim < EMB_DIM; dim++) {
				node_embedding_weight_float[idx + j][dim] = node_embedding_weight_raw[(idx + j) * EMB_DIM + dim];
			    }
		    }
		    idx += nd_f;
	    }

		
	for(int l = 0; l < NUM_LAYERS; l++) {
		idx = 0;
        int	idx2 = 0;
        for(int i = 0; i < EDGE_ATTR; i++) {
            int ed_f = ed_feature_table[i];
            for(int j = 0; j < ed_f; j++) {
                for(int dim = 0; dim < EMB_DIM; dim++) {
                    GCN_edge_embedding_weight_float[l][idx + j][dim] = edge_embedding_weight_raw[l][(idx2 + j) * EMB_DIM + dim];
                }
            }
            idx += ed_f;
            idx2 += ed_f;
        }
    }
    }
    else if(GNN_instruction == 1)
    {
        f = fopen("GIN_weights_biases/gin_ep1_mlp_1_weights_dim32.bin", "r");
        fread(GCN_convs_GIN_node_mlp_1_weight_float, sizeof(float), NUM_LAYERS * GIN_MLP_1_OUT * EMB_DIM, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_mlp_1_bias_dim32.bin", "r");
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float, sizeof(float), NUM_LAYERS * GIN_MLP_1_OUT, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_mlp_2_weights_dim32.bin", "r");
        fread(GIN_node_mlp_2_weights_float, sizeof(float), NUM_LAYERS * EMB_DIM * GIN_MLP_1_OUT, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_mlp_2_bias_dim32.bin", "r");
        fread(GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_float, sizeof(float), NUM_LAYERS * EMB_DIM, f);
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
        fread(graph_pred_PNA_graph_mlp_3_weights_float, sizeof(float), NUM_TASK * EMB_DIM, f);
        fclose(f);

        f = fopen("GIN_weights_biases/gin_ep1_pred_bias_dim32.bin", "r");
        fread(graph_pred_PNA_graph_mlp_3_bias_float, sizeof(float), NUM_TASK, f);
        fclose(f);
    }
    else if(GNN_instruction == 2)
    {
        f = fopen("PNA_weights_biases/pna_ep1_noBN_dim32.weights.all.bin", "rb");

        fseek(f, 0*sizeof(float), SEEK_SET);
        fread(node_embedding_weight_raw_PNA, sizeof(float), ND_FEATURE_TOTAL * PNA_EMB_DIM, f);

        fseek(f, 5568*sizeof(float), SEEK_SET);
        fread(PNA_node_conv_weights_float[0], sizeof(float), 12288, f);

        fseek(f, 17856*sizeof(float), SEEK_SET);
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[0], sizeof(float), 32 , f);

        fseek(f, 17888*sizeof(float), SEEK_SET);
        fread(PNA_node_conv_weights_float[1], sizeof(float), 12288 , f);

        fseek(f, 30176*sizeof(float), SEEK_SET);
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[1], sizeof(float), 32, f);

        fseek(f, 30208*sizeof(float), SEEK_SET);
        fread(PNA_node_conv_weights_float[2], sizeof(float), 12288, f);

        fseek(f, 42496*sizeof(float), SEEK_SET);
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[2], sizeof(float), 32, f);

        fseek(f, 42528*sizeof(float), SEEK_SET);
        fread(PNA_node_conv_weights_float[3], sizeof(float), 12288, f);

        fseek(f, 54816*sizeof(float), SEEK_SET);
        fread(GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[3], sizeof(float), 32, f);

        fseek(f, 54848*sizeof(float), SEEK_SET);
        fread(bn_weight_PNA_graph_mlp_1_weights_float_PNA, sizeof(float), 512, f);

        fseek(f, 55360*sizeof(float), SEEK_SET);
        fread(bn_bias_PNA_graph_mlp_1_bias_float, sizeof(float), 16, f);

        fseek(f, 55376*sizeof(float), SEEK_SET);
        fread(bn_mean_PNA_graph_mlp_2_weights_float_PNA, sizeof(float), 128, f);

        fseek(f, 55504*sizeof(float), SEEK_SET);
        fread(bn_sqrt_var_PNA_graph_mlp_2_bias_float, sizeof(float), 8, f);

        fseek(f, 55512*sizeof(float), SEEK_SET);
        fread(graph_pred_PNA_graph_mlp_3_weights_float, sizeof(float), 8, f);

        fseek(f, 55520*sizeof(float), SEEK_SET);
        fread(graph_pred_PNA_graph_mlp_3_bias_float, sizeof(float), 1, f);

        fclose(f);

        //converting into fixed type
        int idx = 0;
        for(int i = 0; i < ND_FEATURE; i++)
        {
            int nd_f = nd_feature_table[i];
            for(int j = 0; j < nd_f; j++)
            {
                for(int dim = 0; dim < PNA_EMB_DIM; dim++)
                {
                    node_embedding_weight_float[idx + j][dim] = (WT_TYPE)node_embedding_weight_raw_PNA[(idx + j) * PNA_EMB_DIM + dim];
                }
            }
            idx += nd_f;
        }
    }

    int max_dim_out;
    int max_dim_in;
    int max_NUM_LAYERS;
    int max_EMB_DIM;

    switch(GNN_instruction)
    {
        case 0 :    max_dim_out = EMB_DIM;
                    max_dim_in = EMB_DIM;
                    max_NUM_LAYERS = NUM_LAYERS;
                    max_EMB_DIM = EMB_DIM;
                    break;
        
        case 1 :    max_dim_out = GIN_MLP_1_OUT;
                    max_dim_in = EMB_DIM;
                    max_NUM_LAYERS = NUM_LAYERS;
                    max_EMB_DIM = EMB_DIM;
                    break;
        
        case 2 :    max_dim_out = PNA_EMB_DIM;
                    max_dim_in = PNA_EMB_DIM;
                    max_NUM_LAYERS = PNA_NUM_LAYERS;
                    max_EMB_DIM = PNA_EMB_DIM;
                    break;
        
        default :   break;
    }

	for(int l = 0; l < max_NUM_LAYERS; l++) {
        if(GNN_instruction == 1)
        {
            GIN_node_mlp_eps_PNA_avg_deg_fixed[l] = (WT_TYPE)GIN_node_mlp_eps_PNA_avg_deg_float[l];
        }
        for(int dim_out = 0; dim_out < max_dim_out; dim_out++) {
            GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_fixed[l][dim_out] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_PNA_node_conv_bias_float[l][dim_out];
            for(int dim_in = 0; dim_in < max_dim_in; dim_in++) {
                if(GNN_instruction == 0 || GNN_instruction == 1)
                {
                    GCN_convs_GIN_node_mlp_1_weight_fixed[l][dim_out][dim_in] = (WT_TYPE)GCN_convs_GIN_node_mlp_1_weight_float[l][dim_out][dim_in];
                }
                else if(GNN_instruction == 2)
                {
                    for(int scaler = 0; scaler < NUM_SCALERS; scaler++)
                    {
                        for(int aggr = 0; aggr < NUM_AGGRS; aggr++)
                        {
                            PNA_node_conv_weights_fixed[l][dim_out][scaler][aggr][dim_in] = (WT_TYPE)PNA_node_conv_weights_float[l][dim_out][scaler][aggr][dim_in];
                        }
                    }
                }
            }
        }
        if(GNN_instruction == 0 || GNN_instruction == 1)
        {
            for(int dim_out = 0; dim_out < EMB_DIM; dim_out++) {
                GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_fixed[l][dim_out] = (WT_TYPE)GCN_convs_root_emb_weight_GIN_node_mlp_2_bias_float[l][dim_out];
                if(GNN_instruction == 1)
                {
                    for(int dim_in = 0; dim_in < GIN_MLP_1_OUT; dim_in++) {
                    GIN_node_mlp_2_weights_fixed[l][dim_out][dim_in] = (WT_TYPE)GIN_node_mlp_2_weights_float[l][dim_out][dim_in];
                    }
                }
            }
        }
    }
    
    //to load the node embedding weights
    
    for (int i = 0; i < ND_FEATURE_TOTAL; i++) {
        for(int j = 0; j < max_EMB_DIM; j++) {
            node_embedding_weight_fixed[i][j] = (WT_TYPE)node_embedding_weight_float[i][j];
        }
    }
    

    if(GNN_instruction == 0 || GNN_instruction == 2)
    {   
        //to load the edge embedding weights for GCN
        if(GNN_instruction == 0)
        {
            for (int i = 0; i < NUM_LAYERS; i++) {
			    for(int j = 0; j < ED_FEATURE_PER_LAYER; j++){
		    	    for(int k = 0; k < EMB_DIM; k++) {
			        edge_embedding_weight_fixed[i][j][k] = (WT_TYPE)GCN_edge_embedding_weight_float[i][j][k];
		            }
                }
	        }
        }

        //to load the batch normalization parameters and graph mlp parameters for PNA
        for(int i = 0; i < PNA_GRAPH_MLP_1_OUT; i++) {
            if(GNN_instruction == 2)
            {
                bn_bias_PNA_graph_mlp_1_bias_fixed[0][i] = (WT_TYPE)bn_bias_PNA_graph_mlp_1_bias_float[0][i];
                if(i < PNA_GRAPH_MLP_2_OUT)
                {
                    bn_sqrt_var_PNA_graph_mlp_2_bias_fixed[0][i] = (WT_TYPE)bn_sqrt_var_PNA_graph_mlp_2_bias_float[0][i];
                }
            }
		    for(int j = 0; j < max_EMB_DIM; j++) {
                if(GNN_instruction == 0 && i < NUM_LAYERS)
			    {
                    bn_weight_PNA_graph_mlp_1_weights_fixed[i][j] = (WT_TYPE)bn_weight_PNA_graph_mlp_1_weights_float[i][j];
			        bn_bias_PNA_graph_mlp_1_bias_fixed[i][j] = (WT_TYPE)bn_bias_PNA_graph_mlp_1_bias_float[i][j];
			        bn_mean_PNA_graph_mlp_2_weights_fixed[i][j] = (WT_TYPE)bn_mean_PNA_graph_mlp_2_weights_float[i][j];
			        bn_sqrt_var_PNA_graph_mlp_2_bias_fixed[i][j] = (WT_TYPE)bn_sqrt_var_PNA_graph_mlp_2_bias_float[i][j];
		        }
                else if(GNN_instruction == 2)
                {
                    bn_weight_PNA_graph_mlp_1_weights_fixed[i][j] = (WT_TYPE)bn_weight_PNA_graph_mlp_1_weights_float_PNA[i][j];
                    if(i < PNA_GRAPH_MLP_2_OUT && j < PNA_GRAPH_MLP_1_OUT)
                    {
                        bn_mean_PNA_graph_mlp_2_weights_fixed[i][j] = (WT_TYPE)bn_mean_PNA_graph_mlp_2_weights_float_PNA[i][j];
                    }
                }
            }
	    }
    }
    else if(GNN_instruction == 1)
    {
        //to load the edge embedding weights
        for(int l = 0; l < NUM_LAYERS; l++) {
            for(int i = 0; i < ED_FEATURE_PER_LAYER; i++) {
                for(int dim = 0; dim < EMB_DIM; dim++) {
                    edge_embedding_weight_fixed[l][i][dim] = (WT_TYPE)GIN_edge_embedding_weight_float[l][i][dim];
                }
            }
        }
    }

    //to load the graph prediction weights and biases
    int max_dim = (instruction == 2) ? PNA_GRAPH_MLP_2_OUT : EMB_DIM;
    for(int t = 0; t < NUM_TASK; t++) {
        graph_pred_PNA_graph_mlp_3_bias_fixed[t] = (WT_TYPE)graph_pred_PNA_graph_mlp_3_bias_float[t];
        for(int dim_in = 0; dim_in < max_dim; dim_in++ ) {
            graph_pred_PNA_graph_mlp_3_weights_fixed[t][dim_in] = (WT_TYPE)graph_pred_PNA_graph_mlp_3_weights_float[t][dim_in];
        }
    }

    if(GNN_instruction == 2)
    {
        GIN_node_mlp_eps_PNA_avg_deg_fixed[0] = 6.885701656341553;
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
