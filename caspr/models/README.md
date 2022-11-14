The model architecture should follow the following guidelines to support explainability

Basic changes made:
1.  Every model class should have the flags - explain, interpretable_emb_non_seq and interpretable_emb_seq

2. The nn.Embedding layers and the dropout after that need to be modularised
out of the model and the Seq_Cat_Embedding and Non_Seq_Cat_Embedding classes present in the Embedding_Layers.py file should be used for them

3. The input to every forward function should be a single concatenated vector

4. The activate_explainer_mode and deactivate_explainer_mode functions should be a part of every model class (also every model wrapper class)


"""
Some notes regarding the explainer:
1. When we join multiple models to form a new model - 
    use the activate_explainer_mode functions to call the 
    respective functions for all consituent sub_model classes

2. Right now the architecture supports only model wrappers which join the model in a vertical fashion (the case for all our models for now)

3. The explainer modes are activated by the DLExplainer module and 
    also deactivated by it

4. The indices to embedding conversion happens in the DLExplainer module


"""

Please refer the mlp_autoencoder.py file to have a look