"""Implementation ò sparse representation"""
import torch
from torch.nn.utils.rnn import pad_sequence


class SparseRep:
    """Sparse representation which could be easily converted to dense or sparse format"""

    DENSE_FORMAT = "dense"
    SPARSE_FORMAT = "sparse"

    def __init__(self, indices=None, values=None, size=None, dense=None) -> None:
        assert dense is not None or (
            indices is not None and values is not None and size is not None
        )
        if dense is not None:
            self.dense = dense
            self.format = SparseRep.DENSE_FORMAT
            self.device = dense.device
        else:
            self.indices = indices
            self.values = values
            self.device = self.values.device
            if size.dim() == 3:
                self.size = torch.tensor(
                    (size[:, 0].sum(), size[0][1]), device=self.device
                )
            else:
                self.size = size
            self.format = SparseRep.SPARSE_FORMAT

    def to_dict(self):
        "return a dictionary of data"
        if self.format == SparseRep.SPARSE_FORMAT:
            return {
                "indices": self.indices,
                "values": self.values,
                "size": self.size.unsqueeze(0),
            }
        else:
            return {"dense": self.dense}

    def batch_size(self):
        """Return number of examples in batch"""
        if self.format == SparseRep.DENSE_FORMAT:
            return self.dense.size(0)
        else:
            return self.indices.size(0)

    def len(self):
        """Return mean length of sparse vectors"""
        if self.format == SparseRep.DENSE_FORMAT:
            return (self.dense > 0).sum(dim=1).float().mean()
        else:
            return (self.values > 0).sum(dim=1).float().mean()

    def repeat_interleave(self, n, dim=0):
        """Repeat along some dimension"""
        if self.format == SparseRep.DENSE_FORMAT:
            new_dense = self.dense.repeat_interleave(n, dim=dim)
            return SparseRep(dense=new_dense)
        else:
            new_indices = self.indices.repeat_interleave(n, dim=dim)
            new_values = self.values.repeat_interleave(n, dim=dim)
            return SparseRep(indices=new_indices, values=new_values, size=self.size)

    def to_dense(self, reduce="amax"):
        """convert sparse reps to dense reps"""
        if self.format == SparseRep.DENSE_FORMAT:
            return self.dense
        elif self.format == SparseRep.SPARSE_FORMAT:
            dense = torch.zeros(
                self.size.tolist(), device=self.indices.device, dtype=self.values.dtype
            ).scatter_reduce_(1, self.indices, self.values, reduce=reduce)
            return dense
        else:
            raise Exception(f"sparse format {self.format} is not available")

    def to_sparse(self):
        """return sparse reps in tuple of (indices,values, size)"""
        if self.format == SparseRep.DENSE_FORMAT:
            indices = []
            values = []
            for row in self.dense:
                indices.append(row.nonzero(as_tuple=True)[0])
                values.append(row[indices[-1]])
            indices = pad_sequence(indices, batch_first=True, padding_value=0)
            values = pad_sequence(values, batch_first=True, padding_value=0)
            size = self.dense.size()
            return (indices, values, size)
        elif self.format == SparseRep.SPARSE_FORMAT:
            return (self.indices, self.values, self.size())
        else:
            raise Exception(f"sparse format {self.format} is not available")

    def element_wise_dot(self, second):
        """return dot(first[i], second[i]) for all i in range(batch_size)"""
        if self.format == SparseRep.DENSE_FORMAT:
            if second.format == SparseRep.DENSE_FORMAT:
                return (self.dense * second.dense).sum(dim=1)
            else:
                raise Exception(
                    "Dot product between {self.format} and {second.format} is not supported"
                )
        elif self.format == SparseRep.SPARSE_FORMAT:
            if second.format == SparseRep.DENSE_FORMAT:
                selected_vals = second.dense.gather(1, self.indices)
                return (self.values * selected_vals).sum(dim=1)
            else:
                # some redundency, but more efficient
                exact_match = self.indices.unsqueeze(-1) == second.indices.unsqueeze(
                    -2
                )  # BATCH_SIZE x FIRST_LENGTH x SECOND_LENGTH
                score_mat = self.values.unsqueeze(-1) * second.values.unsqueeze(-2)
                score_mat = score_mat * exact_match
                return score_mat.max(dim=-1).values.sum(dim=-1)
        else:
            raise Exception(f"sparse format {self.format} is not available")

    def cross_dot(self, second):
        """return dot(first[i], second[j]) for all i,j"""
        if self.format == SparseRep.DENSE_FORMAT:
            if second.format == SparseRep.DENSE_FORMAT:
                return torch.mm(self.dense, second.dense.transpose(0, 1))
            else:
                raise Exception(
                    "Dot product between {self.format} and {second.format} is not supported"
                )
        elif self.format == SparseRep.SPARSE_FORMAT:
            if second.format == SparseRep.DENSE_FORMAT:
                # some redundency, but more efficient
                ids = self.indices.view(1, -1).expand(
                    second.dense.size(0), -1
                )  # B_SIZE x A_SIZE x SEQ_LENGTH
                selected_vals = second.dense.gather(1, ids)
                selected_vals = selected_vals.view(
                    second.dense.size(0), self.indices.size(0), self.indices.size(1)
                ).permute(
                    1, 2, 0
                )  # A_SIZE x B_SIZE X SEQ_LENGTH
                score_mat = self.values.unsqueeze(-2).bmm(selected_vals)
                return score_mat.squeeze(1)
            else:
                exact_match = (
                    self.indices.unsqueeze(1).unsqueeze(-1)
                    == second.indices.unsqueeze(0).unsqueeze(-2)
                ).float()
                score_mat = self.values.unsqueeze(1).unsqueeze(
                    -1
                ) * second.values.unsqueeze(0).unsqueeze(-2)
                score_mat = score_mat * exact_match
                return score_mat.max(dim=-1).values.sum(dim=-1)
        else:
            raise Exception(f"sparse format {self.format} is not available")
