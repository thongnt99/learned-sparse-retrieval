## You need to set at least the path to anserini to make this work. The rest is optional and commented, but we need that on our tests.

#export TRANSFORMERS_CACHE=IF NEEDED 
#export HF_DATASETS_CACHE=IF NEEDED
export ANSERINI_PATH=#PATH TO ANSERINI
export WANDB_MODE=offline
#export IR_DATASETS_HOME=IF NEEDED
export HYDRA_FULL_ERROR=1

experiment=$1

dataset=$2

if [[ "$dataset" == "msmarco" ]]
then
    EXTRA="/dev"
elif [[ "$dataset" == "fiqa" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "dbpedia-entity" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "fever" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "hotpotqa" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "nfcorpus" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "quora" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "scifact" ]]
then
    EXTRA="/test"
elif [[ "$dataset" == "webis-touche2020" ]]
then
    EXTRA="/v2"
else
    EXTRA=""
fi


mkdir outputs/$experiment/inference/
mkdir outputs/$experiment/inference/doc/
mkdir outputs/$experiment/inference/doc/$dataset/
mkdir outputs/$experiment/inference/runs/

input_path=beir/${dataset}${EXTRA}
output_file_name=$dataset.tsv
batch_size=256
type='query'

python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.input_format=ir_datasets \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
+experiment=$experiment 

input_path=beir/${dataset}${EXTRA}
output_file_name=$dataset/test.jsonl
batch_size=256
type='doc'
python -m lsr.inference \
inference_arguments.input_path=$input_path \
inference_arguments.input_format=ir_datasets \
inference_arguments.output_file=$output_file_name \
inference_arguments.type=$type \
inference_arguments.batch_size=$batch_size \
inference_arguments.scale_factor=100 \
inference_arguments.top_k=-400  \
+experiment=$experiment

rm -r outputs/$experiment/index/$dataset/

$ANSERINI_PATH/target/appassembler/bin/IndexCollection \
  -collection JsonVectorCollection \
  -input outputs/$experiment/inference/doc/$dataset \
  -index outputs/$experiment/index/$dataset/ \
  -generator DefaultLuceneDocumentGenerator \
  -threads 1 -impact -pretokenized \

$ANSERINI_PATH/target/appassembler/bin/SearchCollection \
-index outputs/$experiment/index/$dataset  \
-topics outputs/$experiment/inference/query/$dataset.tsv \
-topicreader TsvString \
-output outputs/$experiment/inference/runs/$dataset.trec \
-impact -pretokenized -hits 1000 -parallelism 60

python clean.py outputs/$experiment/inference/runs/$dataset.trec

mkdir outputs/$experiment/inference/results/

ir_measures beir/${dataset}${EXTRA} outputs/$experiment/inference/runs/$dataset.trec.fixed MRR@1 MRR@10 NDCG@10 R@100 > outputs/$experiment/inference/results/$dataset

cat outputs/$experiment/inference/results/$dataset
