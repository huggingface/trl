import tempfile
import torch

class BaseModelTester:
    all_model_names = None
    trl_model_class = None

    def test_from_save(self):
        """
        Test if the model can be saved and loaded from a directory and get the same weights
        """
        for model_name in self.all_model_names:
            model = self.trl_model_class.from_pretrained(model_name)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                model_from_save = self.trl_model_class.from_pretrained(tmp_dir)
            
            # Check if the weights are the same 
            for key in model_from_save.state_dict():
                self.assertTrue(torch.allclose(model_from_save.state_dict()[key], model.state_dict()[key]))

