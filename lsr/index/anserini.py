from glob import glob
from tqdm import tqdm
import json
import tempfile
import ir_measures
from pathlib import Path
import subprocess
from collections import defaultdict

class AnseriniIndex:
    def __init__(self, anserini_path, quantization_factor=100):
        self.anserini_path = anserini_path
        self.quantization_factor = quantization_factor
        self.anserini_tmp_dir = Path(tempfile.mkdtemp())
        self.anserini_doc_dir = self.anserini_tmp_dir/"docs"
        self.anserini_query_path = self.anserini_tmp_dir/"queries.tsv"
        self.anserini_index_dir = self.anserini_tmp_dir/"index"
        self.anserini_run_path = self.anserini_tmp_dir/"run.trec"
        
    def index(self, doc_dir):
        # Read document representationa and quantize term weights
        print(f"Storing quantized documents to: {doc_dir}")
        for idx, doc_file in tqdm(enumerate(glob(str(doc_dir)+"/*")), desc="Reading doc weights and quantize"):
            with open(doc_file) as fin, open(self.anserini_doc_dir/f"part_{idx}.jsonl", "w") as fout:
                for line in fin:
                    raw_doc = json.loads(line)
                    quantized_vector = {
                        term: int(weight*self.quantization_factor) for term, weight in raw_doc["vecotr"].items()}
                    quantized_doc = {
                        "id": raw_doc["id"], "vector": quantized_vector}
                    fout.write(json.dumps(quantized_doc)+"\n")
        ANSERINI_INDEX_COMMAND = f"""{self.anserini_path}/target/appassembler/bin/IndexCollection
                -collection JsonSparseVectorCollection 
                -input {self.anserini_doc_dir}  
                -index {self.anserini_index_dir}  
                -generator SparseVectorDocumentGenerator 
                -threads 18 
                -impact 
                -pretokenized
                """
        process = subprocess.run(ANSERINI_INDEX_COMMAND.split(), check=True)

    def retrieve(self, query_path):
        # read and quantize
        with open(query_path, "r") as fin, open(self.anserini_query_path, "w") as fout:
            for line in tqdm(fin, desc="Reading queries and quantize"):
                query = json.loads(line)
                toks = []
                for term in query["vector"]:
                    reps = (query["vector"][term] * self.quantization_factor)
                    toks.extend([term] * reps)
                toks = " ".join(toks)
                fout.write(f"{query["id"]}\t{toks}\n")
        ANSERINI_RETRIEVE_COMMAND = f"""{self.anserini_path}/target/appassembler/bin/SearchCollection
          -index {self.anserini_index_dir}  
          -topics {self.anserini_query_path} 
          -topicreader TsvString 
          -output {self.anserini_run_path}  
          -impact 
          -pretokenized 
          -hits 1000 
          -parallelism 18"""
        process = subprocess.run(ANSERINI_RETRIEVE_COMMAND.split(), check=True)
        run = defaultdict(dict)
        trec_run = ir_measures.read_trec_run(self.anserini_run_path)
        for row in trec_run:
            run[row.query_id][row.doc_id] = row.score 
        return run 