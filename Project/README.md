# File index:

This repository contains the following files:

- `Data Processing.ipynb`: Process the data to be used as a base for the BPR models
- `BPR.ipynb`: Create the 2 BPR models required to be used as a base for the CB2CF model
- `CB2CFMultiModalEncoder.py`: The CB2CF embedding model
- `CB2CFMultiModalEncoderDataset.py`: Dataset model for CB2CFMultiModalEncoder
- `create_CB2CFMultiModalEncoder_data_set.ipynb`: Notebook for the creation of a dataset for CB2CFMultiModalEncoder model.
- `evaluate_cb2cf_multi_modal_encoder.ipynb`: An evaluation notebook for the trained CB2CFMultiModalEncoder model.
- `train_CB2CFMultiModalEncoder.ipynb`: Notebook for training CB2CFMultiModalEncoder model
- `utils.py`: Utils for NextItemPredTransformer dataset creation.
- `Tokenizer.py`: Tokenizer for NextItemPredTransformer.
- `NextItemPredTransformer.py`: The predictive model for the next user ranking.
- `NextItemPredDataset.py`: Dataset definition for NextItemPredTransformer.
- `train_and_evaluate_next_item_pred_transformer.ipynb`: Notebook for training and evaluation of NextItemPredTransformer.

# Order in which to run notebooks:

Follow these steps to run the notebooks in order:

1. Run the `Data Processing.ipynb` notebook from a working directory containing a folder named "data" that holds the files from https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset.
2. Run the `BPR.ipynb` notebook from the same directory to produce the 2 BPR models.
3. Run `create_CB2CFMultiModalEncoder_data_set.ipynb` notebook.
4. Run `train_CB2CFMultiModalEncoder.ipynb` notebook.
5. Run `evaluate_cb2cf_multi_modal_encoder.ipynb` notebook.
6. Run `train_and_evaluate_next_item_pred_transformer.ipynb` notebook.
