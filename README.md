The repository contains all the necessary to deploy state-of-art perturbation methods and synthetic ones, in particular the Noise transfer based on Fast Fourier Transform and Vertical-Horizontal-stretching perturbation. For noise transfer, starting from a dataset , we want to extract the kind of noise, and transfer it to another dataset, that we want to degrade. 
The folder src has two python scripts, 'fourier_noise_transfer.py' contains only the perturbation functions, 'fourier_perturbation_script.py' contains all the flow to transform an entire dataset.

Regarding the second, it is set for our specific enviroment and you can launch it considering the following parameters: 

..python fourier_perturbation_script.py
--gcs_source_folder, that is the gcs folder containing input images
--query, query to get the images to perturbate with BigQuery
--gcs_destination_folder, that is the gcs folder containing noised output
--output_counter, an int defining the number of desired outputs
--treshold, a float defining the treshold for noise filtering (this has to be set depending on the input)
--alpha, that float that define the weight of the noise with respect to the good signal

The Framework is optimized for Google Cloud Platform enviroment, but it can still be transfered on another Python enviroment.
