
<img src="images/logo.png" width=6%> ![](https://badgen.net/badge/lsr/instructions/red?icon=github) ![](https://badgen.net/badge/python/3.9.12/green?icon=python)
# LSR: A unified framework for efficient and effective learned sparse retrieval

The framework provides a simple yet effective toolkit for defining, training, and evaluating learned sparse retrieval methods. The framework is composed of standalone modules, allowing for easy mixing and matching of different modules or integration with your own implementation. This provides flexibility to experiment and customize the retrieval model to meet your specific needs.

The structure of this repository is as following: 

```.
├── analysis #scripts that support experimental analysis
├── configs  #configuration of different components
│   ├── dataset 
│   ├── experiment #define exp details: dataset, loss, model, hp 
│   ├── loss 
│   ├── model
│   └── wandb
├── datasets    #implementations of dataset loading & collator
├── evaluate_bier   #code to evaluate on beir benchmark
├── losses  #implementations of different losses + regularizer
├── models  #implementations of different models
├── preprocess  #script to preprocess data (e.g., expand psg)
├── tests   #unit tests for some components
├── tokenizer   #a wrapper of HF's tokenizers
├── trainer     #trainer for training 
└── utils   #common utilities used in different places
```
* The list of all configurations used in the paper could be found [here](#list-of-configurations-used-in-the-paper)

* The instruction for running experiments could be found [here](#training-and-inference-instructions)

## Training and inference instructions 

### 1. Create conda environment and install dependencies: 

Create `conda` environemt:
```
conda create --name lsr python=3.9.12
conda activate lsr
```
Install dependencies with `pip`
```
pip install -r requirements.txt
```

### 2. Downwload/generate datasets
 We have included all pre-defined dataset configurations under `lsr/configs/dataset`. Before starting training, ensure that you have the `ir_datasets` and (huggingface) `datasets` libraries installed, as the framework will automatically download and store the necessary data to the correct directories.

For datasets from `ir_datasets`, the downloaded files are saved by default at `~/.ir_datasets/`. You can modify this path by changing the `IR_DATASETS_HOME` environment variable.

Similarly, for datasets from the HuggingFace's `datasets`, the downloaded files are stored at `~/.cache/huggingface/datasets` by default. To specify a different cache directory, set the `HF_DATASETS_CACHE` environment variable. 

To train a customed model on your own dataset, please use the sample configurations under `lsr/config/dataset` as templates. Overall, you need three important files (see `lsr/dataset_utils` for the file format): 
- document collection: maps `document_id` to `document_text` 
- queries: maps `query_id` to `query_text`
- train triplets or scored pairs:
    - train triplets, used for contrastive learning, contains a list of <`query_id`, `positive_document_id`, `negative_document_id`> triplets.
    - scored_pairs, used for distillation training, contain pairs of <`query`, `document_id`> with a relevance score.  



<!-- #### 2.1 Hard negatives and  CE's scores for distillation
The full dataset consisting of hard negatives and CE's scores could be downloaded from [here](https://download.europe.naverlabs.com/splade/sigir22/data.tar.gz).

To use this dataset with our code, you need to put this dataset to the right directories specified in the coressponding data configuration file at `lsr/configs/dataset/msmarco_distil_nils.yaml`

#### 2.2 BM25 negatives 

The pretokenized BM25 negatives (+queries + positives) could downloaded from [here](http://boston.lti.cs.cmu.edu/luyug/coil/msmarco-psg/).

Similar to 2.1, you need to put this data to the directory specified in `lsr/configs/dataset/coil_pretokenized.yaml`

#### 2.3 Pre-expanding the passages 
To expand the passges with an external model (docT5query or TILDE), you can use resources like [here](https://huggingface.co/doc2query/msmarco-t5-base-v1) or [here](https://github.com/ielab/TILDE/blob/main/create_psg_train_with_tilde.py). We prepare  scripts for expanding passages with TILDE in: `lsr/preprocess`

#### 2.4 Term-recall datasets: 
For training DeepCT model, the term-recall dataset derived from MSMARCO relevant query-passage pairs could be downloaded [here](http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/data/myalltrain.relevant.docterm_recall) -->

### 3. Train a model 

To train a LSR model, you can just simply run the following command:

```bash
python -m lsr.train +experiment=sparta_msmarco_distil \
training_arguments.fp16=True 
```

In this command, `sparta_msmarco_distil` refers to the experiment configuration file located at `lsr/configs/experiment/sparta_msmarco_distil.yaml`. If you wish to use a different experiment, simply change this value to the name of the desired configuration file under `lsr/configs/experiment`.

Please note that we use `wandb` (by default) to monitor the training process, including loss, regularization, query length, and document length. If you wish to disable this feature, you can do so by adding `training_arguments.report_to='none'` to the above command. Alternatively, you can follow the instructions [here](https://docs.wandb.ai/ref/cli/wandb-login) to set up wandb.


### 4. Run inference on MSMARCO dataset 

When the training finished, you can use our inference scripts to generate new queries and documents as following: 

#### 4.1 Generate queries
```
input_path=data/msmarco/dev_queries/raw.tsv
output_file_name=raw.tsv
batch_size=256
type='query'
python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
+experiment=sparta_msmarco_distil 
```
#### 4.2 Generate documents 
```
input_path=data/msmarco/full_collection/split/part01
output_file_name=part01
batch_size=256
type='doc'
python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-400  \
+experiment=sparta_msmarco_distil \ 
```
Note: 
- The `top_k` argument is the number of terms you want to keep; negative `top_k` means no pruning (all positive terms are kept).   
- `scale_factor` is used for weight quantization; float weights are multiplied by this `scale_factor` and rounded to the nearest integer. 
- The inference in document collection will take a long time. Therefore, it is better to split the collection into multiple partitions and run inference using multiple GPUs. 
- All the generated queries and documents are stored in the`output/{exp_name}/inference/` directory by default, where the `exp_name` parameter is defined in the experiment configuration file. You can change it as you like. 

### 5. Index generated documents 
#### 5.1 Download and install our modified Anserini indexing software:
We made simple changes in the indexing procedure in Anserini to improve the indexing speed (by `10x`). 
In the old method, Anserini first creates fake documents from JSON weight files (e.g., `{"hello": 3}`) by repeating the term (e.g., `"helo hello hello"`) and then indexes these documents as regular documents. The process of creating these fake documents can cause a substantial delay in indexing LSR where the number of terms and weights are usually large. To get rid of this issue, we leverage the [FeatureField](https://lucene.apache.org/core/9_3_0/core/org/apache/lucene/document/FeatureField.html) in Lucene to inject the (term, weight) pairs directly to the index. The change is simple but quite effective, especially when you have to index multiple times (as in the paper).   
You can download the modified Anserini version [here](https://anonymous.4open.science/r/anserini-lsr-AD27), then follow the instructions in the [README](https://anonymous.4open.science/r/anserini-lsr-AD27/README.md) for installation. If the tests fail, you can skip it by adding `-Dmaven.test.skip=true`.

When the installation is done, you can continue with the next steps. 
#### 5.2 Index with Anserini
```
./anserini-lsr/target/appassembler/bin/IndexCollection \
-collection JsonTermWeightCollection \
-input outputs/sparta_distil_sentence_transformers/inference/doc/  \
-index outputs/sparta_distil_sentence_transformers/index \
-generator TermWeightDocumentGenerator \
-threads 60 -impact -pretokenized
```
Note that you have to change `sparta_distil_sentence_transformers` to the output defined in your experiment configuation flie (here: `lsr/configs/experiment/sparta_msmarco_distil.yaml`)
### 6. Search on the Inverted Index
```
./anserini-lsr/target/appassembler/bin/SearchCollection \
-index outputs/sparta_distil_sentence_transformers/index/  \
-topics outputs/sparta_distil_sentence_transformers/inference/query/raw.tsv \
-topicreader TsvString \
-output outputs/sparta_distil_sentence_transformers/run.trec \
-impact -pretokenized -hits 1000 -parallelism 60
```
Here, you may need to change the output directory as in 5.2. 
### 7. Evaluate the run file
```
ir_measures qrels.msmarco-passage.dev-subset.txt outputs/sparta_distil_sentence_transformers/run.trec MRR@10 R@1000 NDCG@10
```
`qrels.msmarco-passage.dev-subset.txt` is the qrels file for MSMARCO-dev in TREC format. You can find it on the MSMARCO or TREC DL(19,20) website. Note that for TREC DL (19,20), you have to change `R@1000` to `"R(rel=2)@1000"` (with the quote). 

## List of configurations used in the paper 
* **RQ1: Are the results from recent LSR papers reproducible?**

Results in Table 3 are the outputs of following experiments: 

|  Method  | Configuration  |
| :-------- | :--------------|
| DeepCT | `lsr/configs/experiment/deepct_msmarco_term_level.yaml` |
| uniCOIL| `lsr/configs/experiment/unicoil_msmarco_multiple_negative.yaml` |
| uniCOIL<sub>dT5q</sub>| `lsr/configs/experiment/unicoil_doct5query_msmarco_multiple_negative.yaml` | 
| uniCOIL<sub>tilde</sub>| `lsr/configs/experiment/unicoil_tilde_msmarco_multiple_negative.yaml` | 
| EPIC | `lsr/configs/experiment/epic_original.yaml`| 
| DeepImpact | `lsr/configs/experiment/deep_impact_original.yaml` | 
| TILDE<sub>v2</sub>| `lsr/configs/experiment/tildev2_msmarco_multiple_negative.yaml` |
| Sparta | `lsr/configs/experiment/sparta_original.yaml` |
| Splade<sub>max</sub>| `lsr/configs/experiment/splade_msmarco_multiple_negative.yaml` |
| distilSplade<sub>max</sub>|`lsr/configs/experiment/splade_msmarco_distil_flops_0.1_0.08.yaml`|


* **RQ2: How do LSR methods perform with recent advanced training
techniques?**

Results in Table 4 are the outputs of following experiments: 

|  Method  | Configuration  |
| :-------- | :-------------- |
| uniCOIL| `lsr/configs/experiment/unicoil_msmarco_distil.yaml` |
| uniCOIL<sub>dT5q</sub>| `lsr/configs/experiment/unicoil_doct5query_msmarco_distil.yaml`| 
| uniCOIL<sub>tilde</sub>| `lsr/configs/experiment/unicoil_tilde_msmarco_distil.yaml` | 
| EPIC | `lsr/configs/experiment/epic_msmarco_distil.yaml` | 
| DeepImpact | `lsr/configs/experiment/deep_impact_msmarco_distil.yaml` | 
| TILDE<sub>v2</sub>| `lsr/configs/experiment/tildev2_msmarco_distil.yaml` |
| Sparta | `lsr/configs/experiment/sparta_msmarco_distil.yaml` |
| distilSplade<sub>max</sub>|`lsr/configs/experiment/splade_msmarco_distil.yaml` |
| distilSplade<sub>sep</sub>| `lsr/configs/experiment/splade_asm_msmarco_distil_0.1_0.08.yaml`|

* **RQ3: How does the choice of encoder architecture and regularization
affect results?**

Results in Table 5 are the outputs of following experiments: 

|  Effect  |  Row | Configuration  |
| :-------- | :---- | :-------------- |
| Doc weighting | 1a | Before: `lsr/configs/experiment/splade_asm_dbin_msmarco_distil.yaml` <br> After: `lsr/configs/experiment/splade_asm_dmlp_msmarco_distil.yaml`  |
|  | 1b | Before: `lsr/configs/experiment/unicoil_dbin_tilde_msmarco_distil.yaml` <br> After: `lsr/configs/experiment/unicoil_tilde_msmarco_distil.yaml` |
| Query weighting | 2a | Before: `lsr/configs/experiment/tildev2_msmarco_distil.yaml` <br> After: `lsr/configs/experiment/unicoil_tilde_msmarco_distil.yaml`|
|  | 2b | Before: `lsr/configs/experiment/epic_noq_msmarco_distil.yaml` <br> After: `lsr/configs/experiment/epic_msmarco_distil.yaml`|
| Doc expansion | 3a | Before: `lsr/configs/experiment/splade_asm_dmlp_msmarco_distil.yaml` <br> After: `lsr/configs/experiment/unicoil_tilde_msmarco_distil.yaml`|
|  | 3b | Before: `lsr/configs/experiment/unicoil_msmarco_distil.yaml` <br> After: `lsr/configs/experiment/splade_asm_msmarco_distil_0.1_0.08.yaml` |
|  | 3c | Before: `lsr/configs/experiment/unicoil_msmarco_distil.yaml` <br> After: `lsr/configs/experiment/splade_asm_qmlp_msmarco_distil_0.0_0.08.yaml`|
| Query expansion | 4a | Before: `lsr/configs/experiment/splade_asm_qmlp_msmarco_distil_0.0_0.08.yaml` <br> After: `lsr/configs/experiment/splade_asm_msmarco_distil_0.1_0.08.yaml`|
|  | 4b | Before: ``lsr/configs/experiment/unicoil_tilde_msmarco_distil.yaml`` <br> After: `lsr/configs/experiment/splade_asm_dmlp_msmarco_distil.yaml`|
| Regularization | 5a | Before: `lsr/configs/experiment/splade_asm_qmlp_msmarco_distil_0.0_0.08.yaml` <br> After: `lsr/configs/experiment/splade_asm_qmlp_msmarco_distil_0.0_0.00.yaml`|