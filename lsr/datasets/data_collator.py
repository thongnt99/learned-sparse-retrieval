from cProfile import label
import torch


class SepDataCollator:
    "Tokenize and batch of (query, pos, neg, pos_score, neg_score)"

    def __init__(self, tokenizer, q_max_length, d_max_length):
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch):
        batch_queries = []
        pos_docs = []
        neg_docs = []
        batch_scores = []
        for (query, doc_group, *args) in batch:
            batch_queries.append(query)
            pos_docs.append(doc_group[0])
            neg_docs.append(doc_group[1])
            if len(args) == 0:
                continue
            batch_scores.append(args[0])
        tokenized_queries = self.tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokenized_pos_docs = self.tokenizer(
            pos_docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokenized_neg_docs = self.tokenizer(
            neg_docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        return {
            "queries": dict(tokenized_queries),
            "pos_docs": dict(tokenized_pos_docs),
            "neg_docs": dict(tokenized_neg_docs),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }


class DataCollator:
    "Tokenize and batch of (query, pos, neg, pos_score, neg_score)"

    def __init__(self, tokenizer, q_max_length, d_max_length):
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch):
        batch_queries = []
        batch_query_ids = []
        batch_docs = []
        batch_docs_ids = []
        batch_scores = []
        for example in batch:
            if "query_id" in example:
                batch_query_ids.append(example["query_id"])
            if "query_text" in example:
                batch_queries.append(example["query_text"])
            if "doc_id" in example:
                batch_docs_ids.append(example["doc_id"])
            if "doc_text" in example:
                batch_docs.append(example["doc_text"])
            if "score" in example:
                batch_scores.append(example["score"])
        if len(batch_queries) > 0:
            tokenized_queries = self.tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=self.q_max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
        else:
            tokenized_queries = {}
        if len(batch_docs) > 0:
            if isinstance(batch_docs[0], list):
                doc_groups = list(zip(*batch_docs))
                tokenized_docs = [self.tokenizer(
                    doc_group,
                    padding=True,
                    truncation=True,
                    max_length=self.d_max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                ) for doc_group in doc_groups]
            else:
                tokenized_docs = self.tokenizer(
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.d_max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )
        else:
            tokenized_docs = {}
        return {
            "queries": tokenized_queries,
            "doc_groups": tokenized_docs,
            "query_ids": batch_query_ids,
            "doc_ids": batch_docs_ids,
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }


class TermRecallCollator:
    "Pack queries, docs, term_recall to a single batch"

    def __init__(self, tokenizer, q_max_length, d_max_length) -> None:
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch):
        queries = [exp["query"] for exp in batch]
        docs = [exp["doc"]["title"] for exp in batch]
        queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        docs = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        # term recall
        labels = []
        for exp in batch:
            term_recall = exp["term_recall"]
            pieces = []
            weights = []
            for term in term_recall:
                tks = self.tokenizer.tokenizer.tokenize(term)
                pieces.extend(tks)
                weights.extend([term_recall[term]] + [0] * (len(tks) - 1))
            # convert into vectors
            piece_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(pieces)
            weight_vector = torch.zeros(
                self.tokenizer.get_vocab_size(), dtype=torch.float
            )
            assert len(piece_ids) == len(weights)
            for idx, weight in zip(piece_ids, weights):
                weight_vector[idx] += float(weight)
            labels.append(weight_vector)
        labels = torch.stack(labels, dim=0)
        assert docs["input_ids"].size(0) == labels.size(0)
        return {"queries": dict(queries), "docs_batch": dict(docs), "labels": labels}


class PretokenizedGroupCollator:
    """Collate pre-tokinized training samples with multiple negatives"""

    def __init__(self, tokenizer, q_max_length, d_max_length):
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch):
        queries = [
            self.tokenizer.encode_plus(
                item[0],
                truncation="only_first",
                return_attention_mask=False,
                max_length=self.q_max_length,
                return_special_tokens_mask=True,
            )
            for item in batch
        ]
        doc_groups = [
            self.tokenizer.encode_plus(
                item,
                truncation="only_first",
                return_attention_mask=False,
                max_length=self.d_max_length,
                return_special_tokens_mask=True,
            )
            for item_group in batch
            for item in item_group[1]
        ]
        queries = self.tokenizer.pad(
            queries,
            padding="max_length",
            max_length=self.q_max_length,
            return_tensors="pt",
        )
        doc_groups = self.tokenizer.pad(
            doc_groups,
            padding="max_length",
            max_length=self.d_max_length,
            return_tensors="pt",
        )
        return {"queries": dict(queries), "docs_batch": dict(doc_groups)}
