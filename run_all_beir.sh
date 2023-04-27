#Don't forget to change the config file on the beir.sh
for experiment in splade_asm_msmarco_distil_flops_0.1_0.08
do
    for dataset in arguana fiqa nfcorpus quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq
    do
        bash beir.sh $experiment $dataset
    done
done