import logging

import torch


class DoubleSVD:
    def __init__(self, proj_rank, rank, decay):
        self.proj_rank = proj_rank
        self.rank = rank
        self.decay_factor = decay
        self.sketch_matrix = None
        self.projector_matrix = None
        self.sketch_project_matrix = None
        self.sketch_sigma = None

    def increment_update(self, matrix):
        u_mat, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)

        self.projector_matrix = u_mat[:, : self.proj_rank]
        head = singular_values[: self.rank + 1]
        shrunk = torch.sqrt(torch.maximum(head**2 - head[self.rank] ** 2, torch.tensor(0.0, device=matrix.device)))
        update = u_mat[:, : self.rank] @ torch.diag(shrunk[: self.rank])

        if self.sketch_matrix is None:
            self.sketch_matrix = update
            self.sketch_sigma = shrunk[: self.rank]
            self.sketch_project_matrix = u_mat[:, : self.rank]
            return

        merged = torch.cat([self.decay_factor * self.sketch_matrix, update], dim=1)
        merged_u, merged_s, _ = torch.linalg.svd(merged, full_matrices=False)
        merged_head = merged_s[: self.rank + 1]
        merged_shrunk = torch.sqrt(
            torch.maximum(merged_head**2 - merged_head[self.rank] ** 2, torch.tensor(0.0, device=matrix.device))
        )
        self.sketch_matrix = merged_u[:, : self.rank] @ torch.diag(merged_shrunk[: self.rank])
        self.sketch_sigma = merged_shrunk[: self.rank]
        self.sketch_project_matrix = merged_u[:, : self.rank]

    def get_projector_matrix(self):
        return self.projector_matrix

    def get_sketch_project_matrix(self):
        return self.sketch_project_matrix

    def get_sketch_sigma(self):
        return self.sketch_sigma


class CoSOProjector:
    def __init__(self, proj_rank, rank, db_decay=0.5, update_proj_gap=1, cur_task=0):
        self.proj_rank = proj_rank
        self.rank = rank
        self.db_decay = db_decay
        self.update_proj_gap = update_proj_gap
        self.cur_task = cur_task
        self.dbsvd = None
        self.fd_project_matrix = None
        self.full_former_task_proj = None

    def project(self, full_rank_grad, step_idx):
        if self.cur_task > 0:
            full_rank_grad = full_rank_grad - self.full_former_task_proj @ (
                self.full_former_task_proj.T @ full_rank_grad
            )

        if self.fd_project_matrix is None or step_idx % self.update_proj_gap == 0:
            if self.dbsvd is None:
                self.dbsvd = DoubleSVD(self.proj_rank, self.rank, self.db_decay)
            self.dbsvd.increment_update(full_rank_grad)
            self.fd_project_matrix = self.dbsvd.get_projector_matrix()

        return self.fd_project_matrix.T @ full_rank_grad

    def project_back(self, low_rank_grad):
        return self.fd_project_matrix @ low_rank_grad

    def update_projector(self, cur_task):
        self.cur_task = cur_task
        self.dbsvd = None
        self.fd_project_matrix = None

    def update_historical_space(self, threshold):
        sketch_sigma = self.dbsvd.get_sketch_sigma()
        sval_total = (sketch_sigma**2).sum()
        sval_ratio = (sketch_sigma**2) / sval_total
        rank_num = torch.sum(torch.cumsum(sval_ratio, dim=0) < threshold)

        if self.full_former_task_proj is None:
            self.full_former_task_proj = self.dbsvd.get_sketch_project_matrix()[:, :rank_num]
        else:
            self.full_former_task_proj = torch.cat(
                [self.full_former_task_proj, self.dbsvd.get_sketch_project_matrix()[:, :rank_num]],
                dim=1,
            )
        logging.info("Current historical projector shape: %s", tuple(self.full_former_task_proj.shape))
