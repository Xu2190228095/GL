import numpy as np
import time
import sys
import cvxpy as cp
from BaseGraphicalLasso import BaseGraphicalLasso
from DataHandler import DataHandler


class SingleCLIME(BaseGraphicalLasso):

    # Child class of BaseGraphicalLasso class.
    # Computes a single CLIME problem
    # for the whole data set

    def __init__(self, *args, **kwargs):
        super(SingleCLIME, self).__init__(blocks=1, beta=0, processes=1,
                                       *args, **kwargs)
        self.iteration = "n/a"

    def get_rho(self):
        return self.obs + 1

    def generate_real_thetas(self, line, splitter):
        dh = DataHandler()
        infos = line.split(splitter)
        for network_info in infos:
            filename = network_info.split(":")[0].strip("#").strip()
            dh.read_network(filename, inversion=False)
        self.real_thetas = dh.inverse_sigmas
        dh = None

    def run_algorithm(self, max_iter=10000):
        start_time = time.time()
        # 获取样本协方差矩阵和维度
        sigma = self.emp_cov_mat[0]
        p = sigma.shape[0]
        lambdas = np.full(p, self.lambd)
        print(f"lambda: {lambdas}")

        # 初始化精度矩阵的列存储
        theta_columns = np.zeros((p, p))

        # 定义求解器配置字典（按需选择）
        solver_config = {
            "solver": cp.ECOS,  # 默认使用ECOS
            "max_iters": 1000,
            "verbose": False
        }

        # 遍历每一列，逐列求解CLIME问题
        for i in range(p):
            # 构造标准基向量 e_i (第i个元素为1，其余为0)
            e_i = np.zeros(p)
            e_i[i] = 1.0

            # 定义优化变量：beta（精度矩阵的第i列）和辅助变量u
            beta = cp.Variable(p)
            u = cp.Variable(p, nonneg=True)  # u >= 0

            # 约束条件
            constraints = [
                beta <= u,  # u_j >= beta_j
                -beta <= u,  # u_j >= -beta_j (即u_j >= |beta_j|)
                sigma @ beta <= e_i + lambdas[i],  # 上界约束
                sigma @ beta >= e_i - lambdas[i]  # 下界约束
            ]

            # 定义目标函数：最小化u的和（即L1范数松弛）
            problem = cp.Problem(cp.Minimize(cp.sum(u)), constraints)

            # 调用求解器（内点法）
            try:
                problem.solve(**solver_config)
                if problem.status not in ["optimal", "optimal_inaccurate"]:
                    raise RuntimeError(f"Column {i}求解失败，状态码: {problem.status}")
                theta_columns[:, i] = beta.value
                theta_columns[np.abs(theta_columns) < 2e-1] = 0  # 截断阈值
            except Exception as e:
                print(f"第{i}列求解异常: {str(e)}", file=sys.stderr)
                theta_columns[:, i] = np.zeros(p)  # 失败时填充零

        self.run_time = '{0:.3g}'.format(time.time() - start_time)
        # 对称化处理：取对角线元素和对称元素的较小值
        self.thetas[0] = np.minimum(theta_columns, theta_columns.T)
        # self.only_true_false_edges()


    def temporal_deviations(self):
        self.deviations = ["n/a"]
        self.norm_deviations = ["n/a"]
        self.dev_ratio = "n/a"

    def correct_edges(self):
        self.real_edges = 0
        self.real_edgeless = 0
        self.correct_positives = 0
        self.all_positives = 0
        for real_network in self.real_thetas:
            for i in range(self.dimension - 1):
                for j in range(i + 1, self.dimension):
                    if real_network[i, j] != 0:
                        self.real_edges += 1
                        if self.thetas[0][i, j] != 0:
                            self.correct_positives += 1
                            self.all_positives += 1
                    elif real_network[i, j] == 0:
                        self.real_edgeless += 1
                        if self.thetas[0][i, j] != 0:
                            self.all_positives += 1
        self.precision = float(self.correct_positives)/float(
            self.all_positives)
        self.recall = float(self.correct_positives)/float(
            self.real_edges)
        self.f1score = 2*(self.precision*self.recall)/float(
            self.precision + self.recall)

if __name__ == "__main__" and len(sys.argv) == 3:

    # Input parameters from command line:
    #  1. Data file in csv format
    #  2. lambda

    start_time = time.time()
    datahandler = DataHandler()

    """ Parameters for creating solver instance """
    filename = sys.argv[1]
    real_data = True
    if "synthetic_data" in filename:
        real_data = False
    lambd = float(sys.argv[2])

    """ Create solver instance """
    print("\nReading file: %s\n" % filename)
    solver = SingleCLIME(filename=filename,
                      lambd=lambd,
                      datecolumn=real_data)
    print("Total data samples: %s" % solver.datasamples)
    print("Blocks: %s" % solver.blocks)
    print("Observations in a block: %s" % solver.obs)
    print("Rho: %s" % solver.rho)
    print("Lambda: %s" % solver.lambd)
    print("Beta: %s" % solver.beta)
    print("Processes: %s" % solver.processes)

    """ Run algorithm """
    print("\nRunning algorithm...")
    solver.run_algorithm()

    """ Evaluate and print results """
    print("\nNetwork 0:")
    for j in range(solver.dimension):
        print(solver.thetas[0][j, :])
    print("\nTemporal deviations: ")
    solver.temporal_deviations()
    print(solver.deviations)
    print("Normalized Temporal deviations: ")
    print(solver.norm_deviations)
    try:
        print("Temp deviations ratio: {0:.3g}".format(solver.dev_ratio))
    except ValueError:
        print("Temp deviations ratio: n/a")

    """ Evaluate and create result file """
    if not real_data:
        solver.correct_edges()
        print("\nTotal Edges: %s" % solver.real_edges)
        print("Correct Edges: %s" % solver.correct_positives)
        print("Total Zeros: %s" % solver.real_edgeless)
        false_edges = solver.all_positives - solver.correct_positives
        print("False Edges: %s" % false_edges)
        print("F1 Score: %s" % solver.f1score)
        datahandler.write_results(filename, solver)
    else:
        datahandler.write_network_results(filename, solver)

    """ Running times """
    print("\nAlgorithm run time: %s seconds" % (solver.run_time))
    print("Execution time: %s seconds" % (time.time() - start_time))



