
import numpy as np

def her_config(env_id,
               study,
               data_path,
               noise='ou_0.3',
               trial_id=999,
               seed=int(np.random.random()*1e6),
               nb_epochs=50,
               buffer_location=None,
               gep_memory=None,
               num_cpu=19
               ):
    args_dict = dict(env_name=env_id,
                     logdir=data_path,
                     n_epochs=nb_epochs,
                     num_cpu=num_cpu,
                     seed=seed,
                     policy_save_interval=1,
                     replay_strategy='future',
                     clip_return=1,
                     trial_id=trial_id,
                     noise_type=noise,
                     gep_memory=gep_memory,
                     buffer_location=buffer_location,
                     study=study
                     )

    return args_dict

