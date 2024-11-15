import os
import torch

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge


from huggingface_hub import HfApi

def upload_model_to_hf(folder_path : str, repo_id: str):
    api = HfApi()
    # Create the repository if it doesn't exist
    repo = api.create_repo(repo_id, repo_type="model")
    
    # Upload the folder to the specified repository
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
    )

class MergeConfig:
    def __init__(self, method: str = "linear"):
        self.method = method
        self.policy_model_path = None  
        self.target_model_path = None  

        # Initialize relevant parameters based on the method
        if method == 'linear':
            self.policy_model_weight = 0.5
            self.target_model_weight = 0.5
            self.dtype = 'float16'
        elif method == 'ties':
            self.policy_model_weight = 1.0
            self.policy_model_density = [1.0, 0.7, 0.1]
            self.target_model_weight = 1.0
            self.target_model_density = [1.0]
            self.normalize = 1.0
            self.dtype = 'float16'
        elif method == 'dare_ties':
            self.policy_model_weight = 1.0
            self.policy_model_density = [1.0, 0.7, 0.1]
            self.target_model_weight = 1.0
            self.target_model_density = [1.0]
            self.normalize = 1.0
            self.dtype = 'float16'
        elif method == 'slerp':
            self.t_values = 0.5
            self.dtype = 'float16'
        else:
            raise ValueError(f"Unsupported merge method: {method}")
    
    def create_merge_config_linear(self) -> MergeConfiguration:
        """
        Creates a merge configuration for a linear merge of two models with specified weights.
        
        Args:
            policy_model_path (str): Path to the policy model.
            target_model_path (str): Path to the target model.
            policy_model_weight (float, optional): Weight for the policy model. Defaults to 0.5.
            target_model_weight (float, optional): Weight for the target model. Defaults to 0.5.
            dtype (str, optional): Data type for the merge. Defaults to 'float16'.
        
        Returns:
            MergeConfiguration: A MergeConfiguration object with the provided settings.
        """
        # Create the merge configuration dictionary
        merge_config_dict = {
            'dtype': self.dtype,
            'merge_method': 'linear',
            'models': [
                {'model': self.policy_model_path, 'parameters': {'weight': self.policy_model_weight}},
                {'model': self.target_model_path, 'parameters': {'weight': self.target_model_weight}}
            ]
        }
        
        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)
        
        return merge_config

    def create_merge_config_ties(self) -> MergeConfiguration:

        """
        Creates a merge configuration for a TIES merge of two models, with specified weights and densities.
        
        Args:
            policy_model_path (str): Path to the policy model (the one being merged with target).
            target_model_path (str): Path to the target base model.
            policy_model_weight (float, optional): Weight for the policy model. Defaults to 1.0.
            policy_model_density (list, optional): Density values for TIES merge of policy model. Defaults to [1.0, 0.7, 0.1].
            target_model_weight (float, optional): Weight for the target model. Defaults to 1.0.
            target_model_density (list, optional): Density values for TIES merge of target model. Defaults to [1.0].
            normalize (float, optional): Normalization parameter. Defaults to 1.0.
            dtype (str, optional): Data type for the merge. Defaults to 'float16'.
        
        Returns:
            MergeConfiguration: A MergeConfiguration object with the provided settings.
        """
        # Create the TIES merge configuration dictionary
        merge_config_dict = {
            'merge_method': 'ties',
            'slices': None,  # Optional slices if needed
            'models': [
                {
                    'model': {
                        'model': {
                            'path': self.target_model_path,
                            'revision': None
                        },
                        'lora': None,
                        'override_architecture': None
                    },
                    'parameters': {
                        'density': self.target_model_density,
                        'weight': self.target_model_weight
                    }
                },
                {
                    'model': {
                        'model': {
                            'path': self.policy_model_path,
                            'revision': None
                        },
                        'lora': None,
                        'override_architecture': None
                    },
                    'parameters': {
                        'density': self.policy_model_density,
                        'weight': self.policy_model_weight
                    }
                }
            ],
            'parameters': {
                'normalize': self.normalize
            },
            'base_model': {
                'model': {
                    'path': self.policy_model_path,
                    'revision': None
                },
                'lora': None,
                'override_architecture': None
            },
            'dtype': self.dtype,
            'tokenizer_source': None,
            'tokenizer': None,
            'chat_template': None,
            'out_dtype': None
        }
        
        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)
        
        return merge_config

    def create_merge_config_dare_ties(self) -> MergeConfiguration:
        
        """
        Creates a merge configuration for a DARE TIES merge of two models, with specified weights and densities.
        
        Args:
            policy_model_path (str): Path to the policy model (the one being merged with target).
            target_model_path (str): Path to the target base model.
            policy_model_weight (float, optional): Weight for the policy model. Defaults to 1.0.
            policy_model_density (list, optional): Density values for DARE TIES merge of policy model. Defaults to [1.0, 0.7, 0.1].
            target_model_weight (float, optional): Weight for the target model. Defaults to 1.0.
            target_model_density (list, optional): Density values for DARE TIES merge of target model. Defaults to [1.0].
            normalize (float, optional): Normalization parameter. Defaults to 1.0.
            dtype (str, optional): Data type for the merge. Defaults to 'float16'.
        
        Returns:
            MergeConfiguration: A MergeConfiguration object with the provided settings.
        """
        # Create the DARE TIES merge configuration dictionary
        merge_config_dict = {
            'merge_method': 'dare_ties',
            'slices': None,  # Optional slices if needed
            'models': [
                {
                    'model': {
                        'model': {
                            'path': self.target_model_path,
                            'revision': None
                        },
                        'lora': None,
                        'override_architecture': None
                    },
                    'parameters': {
                        'density': self.target_model_density,
                        'weight': self.target_model_weight
                    }
                },
                {
                    'model': {
                        'model': {
                            'path': self.policy_model_path,
                            'revision': None
                        },
                        'lora': None,
                        'override_architecture': None
                    },
                    'parameters': {
                        'density': self.policy_model_density,
                        'weight': self.policy_model_weight
                    }
                }
            ],
            'parameters': {
                'normalize': self.normalize
            },
            'base_model': {
                'model': {
                    'path': self.policy_model_path,
                    'revision': None
                },
                'lora': None,
                'override_architecture': None
            },
            'dtype': self.dtype,
            'tokenizer_source': None,
            'tokenizer': None,
            'chat_template': None,
            'out_dtype': None
        }
        
        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)
        
        return merge_config

    def create_merge_config_slerp(self) -> MergeConfiguration:
        """
        Creates a merge configuration for a SLERP merge of a model with a base model.
        
        Args:
            model_path (str): Path to the model to be merged.
            base_model_path (str): Path to the base model.
            t_values (list, optional): List of ConditionalParameter values for SLERP. Defaults to None.
            dtype (str, optional): Data type for the merge. Defaults to 'float16'.
        
        Returns:
            MergeConfiguration: A MergeConfiguration object with the provided settings.
        """
        
        # Create the SLERP merge configuration dictionary
        merge_config_dict = {
            'merge_method': 'slerp',
            'slices': None,  # Optional slices if needed
            'models': [
                {
                    'model': {
                        'model': {
                            'path': self.target_model_path,
                            'revision': None
                        },
                        'lora': None,
                        'override_architecture': None
                    },
                    'parameters': None  # No specific parameters for SLERP model
                }
            ],
            'parameters': {
                't': self.t_values  # Set the t values for SLERP
            },
            'base_model': {
                'model': {
                    'path': self.policy_model_path,
                    'revision': None
                },
                'lora': None,
                'override_architecture': None
            },
            'dtype': self.dtype,
            'tokenizer_source': None,
            'tokenizer': None,
            'chat_template': None,
            'out_dtype': None
        }
        
        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)
        
        return merge_config

    
    def create(self) -> MergeConfiguration:
        if self.method == 'linear':
            return self.create_merge_config_linear()
        elif self.method == 'ties':
            return self.create_merge_config_ties()
        elif self.method == 'dare_ties':
            return self.create_merge_config_dare_ties()
        elif self.method == 'slerp':
            return self.create_merge_config_slerp()

def merge_models(config : MergeConfig, out_path : str):
    """
    Merge two models using mergekit 

    Args:
        config (MergeConfig): The merge configuration.
        out_path (str): The output path for the merged model.
    """
    run_merge(
        config,
        out_path=out_path,
        options=MergeOptions(
            cuda=torch.cuda.is_available(),
            copy_tokenizer=True,
            lazy_unpickle=False,
            low_cpu_memory=False,
        ),
    )
