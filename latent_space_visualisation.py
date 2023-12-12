import os,sys
import json
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt

from EVE import VAE_model
from utils import data_utils

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Latent space')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--one_hot_location', type=str, help='File location where the one-hot-encoded 3D array for the MSA is stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name is the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--output_latent_space_location', type=str, help='Output location of latent space variables')
    parser.add_argument('--num_samples_latent_space', type=int, help='Num of samples for latent space')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size when computing sampling latent space')
    args = parser.parse_args()

    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file['protein_name'][args.protein_index]
    msa_location = args.MSA_data_folder + os.sep + mapping_file['msa_location'][args.protein_index]
    print("Protein name: "+str(protein_name))
    print("MSA file: "+str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: "+str(theta))

    data = data_utils.MSA_processing(
            MSA_location=msa_location,
            theta=theta,
            use_weights=True,
            one_hot_location=args.one_hot_location + os.sep + protein_name + '_binary.npy',
            weights_location=args.MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    )

    model_name = protein_name + "_" + args.model_name_suffix
    print("Model name: "+str(model_name))

    model_params = json.load(open(args.model_parameters_location))

    model = VAE_model.VAE_model(
                    model_name=model_name,
                    data=data,
                    encoder_parameters=model_params["encoder_parameters"],
                    decoder_parameters=model_params["decoder_parameters"],
                    random_seed=42
    )
    model = model.to(model.device)

    try:
        checkpoint_name = str(args.VAE_checkpoint_location) + os.sep + model_name + "_final"
        print("Checkpoint name: '{}'".format(checkpoint_name))
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Initialized VAE with checkpoint '{}' ".format(checkpoint_name))
    except:
        print("Unable to locate VAE model checkpoint")
        sys.exit(0)

    print("Encoding latent data")
    latent_variables, mu_array, log_var_array = model.latent_space(msa_data=data, 
                                        num_samples=args.num_samples_latent_space,
                                        batch_size=args.batch_size)
    print("Latent data recorded")
    
    # Create a DataFrame with columns for z, mu, and log_var
    df = pd.DataFrame({
        'latent_variable_dim_1': latent_variables[:, 0],  
        'latent_variable_dim_2': latent_variables[:, 1],  
        'mu_dim_1': mu_array[:, 0], 
        'mu_dim_2': mu_array[:, 1], 
        'log_var_dim_1': log_var_array[:, 0],  
        'log_var_dim_2': log_var_array[:, 1],  
        # Add more columns as needed for the dimensions of z, mu, and log_var
    })

    latent_space_output_filename = args.output_latent_space_location+os.sep+protein_name+'_latent_space.csv'
    try:
        keep_header = os.stat(latent_space_output_filename).st_size == 0
    except:
        keep_header=True 
    df.to_csv(path_or_buf=latent_space_output_filename, index=False, mode='a', header=keep_header)

    # Create a scatter plot
    plt.figure(0)
    plt.scatter(mu_array[:,0], mu_array[:,1], alpha=0.5)
    plt.xlim((-6,6))
    plt.ylim((-6,6))
    plt.xlabel("$Z_1$")
    plt.ylabel("$Z_2$")
    plt.tight_layout()
    plt.grid(True)


    # Save or show the plot
    plt.savefig('./results/mu_scatter_plot.png')  
