from glob import glob
from tqdm import tqdm
import json
import tempfile
import ir_measures
from pathlib import Path
import subprocess
from collections import defaultdict
import shutil
import os 
class AnseriniIndex:
    def __init__(self, anserini_path, quantization_factor=100, num_processes=18):
        self.anserini_path = anserini_path
        self.quantization_factor = quantization_factor
        anserini_rep_dir = os.getenv("ANSERINI_OUTPUT_PATH")
        if anserini_rep_dir is None:
            self.anserini_tmp_dir = Path(tempfile.mkdtemp(prefix="anserini_lsr"))
        else:
            self.anserini_tmp_dir = anserini_rep_dir
        self.anserini_doc_dir = self.anserini_tmp_dir/"docs"
        self.anserini_query_path = self.anserini_tmp_dir/"queries.tsv"
        self.anserini_index_dir = self.anserini_tmp_dir/"index"
        self.num_processes = num_processes

    def index(self, raw_doc_dir):
        # Read document representationa and quantize term weights
        print(f"Storing quantized documents to: {raw_doc_dir}")
        if self.anserini_doc_dir.is_dir():
            shutil.rmtree(self.anserini_doc_dir)
        self.anserini_doc_dir.mkdir()
        for idx, doc_file in tqdm(enumerate(glob(str(raw_doc_dir)+"/*")), desc="Reading raw doc weights and quantize"):
            with open(doc_file) as fin, open(self.anserini_doc_dir/f"part_{idx}.jsonl", "w") as fout:
                for line in fin:
                    raw_doc = json.loads(line)
                    quantized_vector = {
                        term: int(weight*self.quantization_factor) for term, weight in raw_doc["vector"].items()}
                    quantized_vector = {w: v for w, v in quantized_vector.items() if v > 0}
                    quantized_doc = {
                        "id": raw_doc["id"], "vector": quantized_vector}
                    fout.write(json.dumps(quantized_doc)+"\n")
        ANSERINI_INDEX_COMMAND = f"""{self.anserini_path}/target/appassembler/bin/IndexCollection
                -collection JsonSparseVectorCollection 
                -input {self.anserini_doc_dir}  
                -index {self.anserini_index_dir}  
                -generator SparseVectorDocumentGenerator 
                -threads {self.num_processes} 
                -impact 
                -pretokenized
                """
        process = subprocess.run(ANSERINI_INDEX_COMMAND.split(), check=True)

    def retrieve(self, raw_query_path, run_path):
        # read and quantize
        with open(raw_query_path, "r") as fin, open(self.anserini_query_path, "w") as fout:
            for line in tqdm(fin, desc="Reading raw query weights and quantize"):
                query = json.loads(line)
                toks = []
                for term in query["vector"]:
                    reps = int(query["vector"][term] * self.quantization_factor)
                    toks.extend([term] * reps)
                toks = " ".join(toks)
                fout.write(f"{query["id"]}\t{toks}\n")
        ANSERINI_RETRIEVE_COMMAND = f"""{self.anserini_path}/target/appassembler/bin/SearchCollection
          -index {self.anserini_index_dir}  
          -topics {self.anserini_query_path} 
          -topicreader TsvString 
          -output {run_path}  
          -impact 
          -pretokenized 
          -hits 1000 
          -parallelism {self.num_processes}"""
        process = subprocess.run(ANSERINI_RETRIEVE_COMMAND.split(), check=True)
        trec_run = ir_measures.read_trec_run(run_path)
        return trec_run 