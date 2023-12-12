export MSA_data_folder='./data/MSA'
export MSA_list='./data/mappings/cbbm_mapping_2D.csv'
export MSA_weights_location='./data/weights'
export one_hot_location='./data/MSA'
export VAE_checkpoint_location='./results/VAE_parameters'
export model_name_suffix='CbbM(I)_Ute_MSA_2D'
export model_parameters_location='./EVE/default_model_params.json'
export training_logs_location='./logs/'
export protein_index=0

export output_latent_space_location='./results/evol_indices'
export num_samples_latent_space=20000
export batch_size=2048

python latent_space_visualisation.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --one_hot_location ${one_hot_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --output_latent_space_location ${output_latent_space_location} \
    --num_samples_latent_space ${num_samples_latent_space} \
    --batch_size ${batch_size}