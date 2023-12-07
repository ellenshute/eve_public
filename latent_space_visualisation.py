import os,sys
import json
import argparse
import pandas as pd
import torch

from EVE import VAE_model
from utils import data_utils

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Evol indices')
    parser.add_argument('--MSA_data_folder', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--MSA_list', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--protein_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--one_hot_location', type=str, help='File location where the one-hot-encoded 3D array for the MSA is stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_name_suffix', default='Jan1', type=str, help='model checkpoint name is the protein name followed by this suffix')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--computation_mode', type=str, help='Computes evol indices for all single AA mutations or for a passed in list of mutations (singles or multiples) [all_singles,input_mutations_list]')
    parser.add_argument('--output_evol_indices_location', type=str, help='Output location of computed evol indices')
    parser.add_argument('--output_evol_indices_filename_suffix', default='', type=str, help='(Optional) Suffix to be added to output filename')
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
            one_hot_location=args.one_hot_location + os.sep + protein_name + '_binary_.npy',
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
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Initialized VAE with checkpoint '{}' ".format(checkpoint_name))
    except:
        print("Unable to locate VAE model checkpoint")
        sys.exit(0)


    mu, sigma, p, evol_indices = model.latent_space(msa_data=data, 
                                        num_samples=args.num_samples_latent_space,
                                        batch_size=args.batch_size)
    
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


    evol_indices_output_filename = args.output_evol_indices_location+os.sep+protein_name+'_'+str(args.num_samples_compute_evol_indices)+'_samples'+args.output_evol_indices_filename_suffix+'.csv'
    try:
        keep_header = os.stat(evol_indices_output_filename).st_size == 0
    except:
        keep_header=True 
    df.to_csv(path_or_buf=evol_indices_output_filename, index=False, mode='a', header=keep_header)

#To be integrated into VAE_model.py

def latent_space(self, msa_data, num_samples, batch_size=256):

    one_hot_sequences = msa_data.one_hot_location

    one_hot_sequences_tensor = torch.tensor(one_hot_sequences)
    dataloader = torch.utils.data.DataLoader(one_hot_sequences_tensor, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    latent_variables = []  # List to store latent variables (z)
    mu_list = []  # List to store means (mu)
    log_var_list = []  # List to store log variances (log_var)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch.type(self.dtype).to(self.device)
            batch_latent_samples = []  
            batch_mu = [] 
            batch_log_var = []  

            for _ in range(num_samples):
                mu, log_var = self.encoder(x)  
                z = self.sample_latent(mu, log_var)  
                batch_latent_samples.append(z.cpu().numpy()) 
                batch_mu.append(mu.cpu().numpy())  
                batch_log_var.append(log_var.cpu().numpy())  

            latent_variables.extend(batch_latent_samples)  
            mu_list.extend(batch_mu) 
            log_var_list.extend(batch_log_var)  

    latent_variables = np.array(latent_variables)  
    mu_array = np.array(mu_list)  
    log_var_array = np.array(log_var_list)  

    return latent_variables, mu_array, log_var_array

    
