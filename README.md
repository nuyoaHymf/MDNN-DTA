# MDNN-DTA
a multimodal deep neural network for drug-target affinity prediction
# Datasets
```
Davis: dataset/davis/original
```
```
KIBA: dataset/kiba/original
```
# Requirements
Python 3.8.18 <br> PyTorch 1.13.1<br> Biopython<br> Numpy<br> Pandas<br> Pillow<br> Rdkit-pypi<br> Scikit-learn<br> Scipy<br> Fair-esm<br>
# Training
* We need to extract the protein feature from the ESM first
```
python run_esm.py \
--model-location "esm1v_t33_650M_UR90S_1" \
--fasta-file dataset/davis/original/davisFASTA.fasta \
--output-dir dataset/protein_feature/
```
* Training
```
python train.py --config default_config.json
```
