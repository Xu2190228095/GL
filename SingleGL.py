from BaseGraphicalLasso import BaseGraphicalLasso
from DataHandler import DataHandler
import penalty_functions as pf
import numpy as np
import time
import os
import glob
import matplotlib.pyplot as plt


def select_lambda_via_f1score(solver_class, filename, real_data, lambdas, EDGE=False, isOneBit=False):
    best_f1score = -np.inf
    best_lambda = None
    best_theta = None
    best_solver = None

    tpr_list = []
    fpr_list = []
    for lambd in lambdas:
        # 初始化并拟合模型
        solver = solver_class(filename=filename, lambd=lambd, datecolumn=real_data, EDGE=EDGE, isOneBit=isOneBit)
        solver.run_algorithm()

        theta = solver.thetas[0]  # 获取估计的精度矩阵

        theta_no_diag = theta.copy()  # 避免修改原矩阵
        np.fill_diagonal(theta_no_diag, 0)  # 将对角线元素置为0
        if not theta_no_diag.any():
            continue
        solver.correct_edges()
        print("\nTotal Edges: %s" % solver.real_edges)
        print("Correct Edges: %s" % solver.correct_positives)
        print("Total Zeros: %s" % solver.real_edgeless)
        false_edges = solver.all_positives - solver.correct_positives
        print("False Edges: %s" % false_edges)
        tpr = solver.correct_positives/solver.real_edges
        fpr = (solver.all_positives - solver.correct_positives)/solver.real_edgeless
        tpr_list.append(tpr)
        fpr_list.append(fpr)

        # 检查精度矩阵是否正定
        try:
            sign, log_det = np.linalg.slogdet(theta)
            if sign < 0:
                print(f"Warning: non-positive definite precision matrix,sign: {sign},log_det: {log_det}")
                continue  # 跳过非正定情况
        except np.linalg.LinAlgError:
            continue  # 奇异矩阵，无法计算行列式

        # 计算f1score
        f1score = solver.f1score

        # 更新最优解
        if f1score > best_f1score:
            best_f1score = f1score
            best_lambda = lambd
            best_theta = theta
            best_solver = solver

    # 转换为数组并排序
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    # 添加起始点(0,0)
    fpr_sorted = np.concatenate([[0.0], fpr_sorted])
    tpr_sorted = np.concatenate([[0.0], tpr_sorted])

    # 将FPR扩展到1.0，保持最后一个TPR值
    fpr_sorted = np.append(fpr_sorted, 1.0)
    tpr_sorted = np.append(tpr_sorted, tpr_sorted[-1])
    # 计算AUC
    auc = np.trapz(tpr_sorted, fpr_sorted)

    return best_lambda, best_theta, best_f1score, auc, best_solver

def draw_picture(result, dimension, title, xlable, ylable, filename):
    # 提取横纵坐标
    result_sorted = sorted(result, key=lambda pair: pair[0])
    x = [pair[0] for pair in result_sorted]
    y = [pair[1] for pair in result_sorted]
    y_EDGE = [pair[2] for pair in result_sorted]
    y_GL = [pair[3] for pair in result_sorted]

    # 创建画布
    plt.figure(figsize=(12, 7))

    # 绘制三条折线（可自定义颜色和样式）
    plt.plot(x, y,
             linestyle='--',
             marker='o',
             color='blue',
             label='Origin Data',
             alpha=0.8,
             linewidth=2)

    plt.plot(x, y_EDGE,
             linestyle='-.',
             marker='s',
             color='green',
             label='EDGE',
             alpha=0.8,
             linewidth=2)

    plt.plot(x, y_GL,
             linestyle='-',
             marker='^',
             color='red',
             label='Graphical Lasso',
             alpha=0.8,
             linewidth=2)

    # 设置坐标轴和标题
    plt.title(title, fontsize=14)
    plt.xlabel(xlable, fontsize=12)
    plt.ylabel(ylable, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 优化显示范围
    plt.xlim(min(x) - 100, max(x) + 100)
    plt.ylim(0, max(y + y_EDGE + y_GL) * 1.1)  # 自动适应纵坐标范围

    # # 添加数据标签（可选）
    # for xi, yi, ye, yg in zip(x, y, y_EDGE, y_GL):
    #     plt.text(xi, yi + 0.01, f"{yi:.2f}", ha='center', fontsize=8, color='blue')
    #     plt.text(xi, ye + 0.01, f"{ye:.2f}", ha='center', fontsize=8, color='green')
    #     plt.text(xi, yg + 0.01, f"{yg:.2f}", ha='center', fontsize=8, color='red')

    # 设置图例位置（避免遮挡数据）
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

class SingleGL(BaseGraphicalLasso):

    # Child class of BaseGraphicalLasso class.
    # computes a single Graphical Lasso problem
    # for the whole data set

    def __init__(self, *args, **kwargs):
        super(SingleGL, self).__init__(blocks=1, beta=0, processes=1,
                                       *args, **kwargs)
        self.nju = float(self.obs)/float(self.rho)
        self.iteration = "n/a"
        self.penalty_function = "n/a"
        self.e = 1e-7

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

    def theta_update(self):
        a = self.z0s[0] - self.u0s[0]
        at = a.transpose()
        m = self.nju*(a + at)/2 - self.emp_cov_mat[0]
        d, q = np.linalg.eig(m)
        qt = q.transpose()
        sqrt_matrix = np.sqrt(d**2 + 4/self.nju*np.ones(self.dimension))
        diagonal = np.diag(d) + np.diag(sqrt_matrix)
        self.thetas[0] = np.real(
            self.nju/2*np.dot(np.dot(q, diagonal), qt))

    def z_update(self):
        self.z0s[0] = pf.soft_threshold_odd(self.thetas[0] + self.u0s[0],
                                            self.lambd, self.rho)

    def u_update(self):
        self.u0s[0] = self.u0s[0] + self.thetas[0] - self.z0s[0]

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


if __name__ == "__main__":

    start_time = time.time()
    datahandler = DataHandler()

    """ Parameters for creating solver instance """
    filepath = "synthetic_data/"
    dimension = 100
    pattern = os.path.join(filepath, f"{dimension}x*")
    file_list = glob.glob(pattern)

    lambdas = np.linspace(0.005, 200, 10)  # 生成候选lambda值（从0.001到10的等间距）
    # lambdas = [1]
    print(f"\n候选lambda值：{lambdas}")

    if not file_list:
        print(f"未找到dimension = '{dimension}' 的文件")
    else:
        # 遍历所有匹配的文件
        f1scores = []
        auc_scores = []
        for filename in file_list:
            print(f"\n正在处理文件：{filename}")
            real_data = True
            if "synthetic_data" in filename:
                real_data = False

            """ Create solver instance """
            best_lambda, best_theta, best_f1score, auc_score, solver = select_lambda_via_f1score(
                solver_class=SingleGL,
                filename=filename,
                real_data=real_data,
                lambdas=lambdas
            )
            print(f"\n最优lambda: {best_lambda}, f1score: {best_f1score}, auc: {auc_score}")

            best_lambda_EDGE, best_theta_EDGE, best_f1score_EDGE, auc_score_EDGE, solver_EDGE = select_lambda_via_f1score(
                solver_class=SingleGL,
                filename=filename,
                real_data=real_data,
                lambdas=lambdas,
                EDGE=True
            )
            print(f"\n最优lambda: {best_lambda_EDGE}, f1score: {best_f1score_EDGE}, auc: {auc_score_EDGE}")

            best_lambda_GL, best_theta_GL, best_f1score_GL,auc_score_GL, solver_GL = select_lambda_via_f1score(
                solver_class=SingleGL,
                filename=filename,
                real_data=real_data,
                lambdas=lambdas,
                isOneBit=True
            )
            print(f"\n最优lambda: {best_lambda_GL}, f1score: {best_f1score_GL}, auc: {auc_score_GL}")

            """ Evaluate and print results """
            # print("\nNetwork 0:")
            # for j in range(solver.dimension):
            #     print(solver.thetas[0][j, :])
            #     print(solver.u0s[0][j, :])
            # print("\nTemporal deviations: ")

            """ Evaluate and create result file """
            if not real_data:
                print(f"F1 Score: {solver.f1score}, {solver_EDGE.f1score}, {solver_GL.f1score}")
                f1scores.append((solver.obs, solver.f1score,solver_EDGE.f1score, solver_GL.f1score))
                auc_scores.append((solver.obs, auc_score,auc_score_EDGE, auc_score_GL))
            else:
                datahandler.write_network_results(filename, solver)

            """ Running times """
            print("\nAlgorithm run time: %s seconds" % (solver.run_time))
            print("Execution time: %s seconds" % (time.time() - start_time))

        draw_picture(f1scores, dimension,f"F1 Score, dimension={dimension}",
                     "sample number", "F1 Score", f"pictures/{dimension}_f1_scores.png")
        draw_picture(auc_scores, dimension, f"AUC Score, dimension={dimension}",
                     "sample number", "AUC Score", f"pictures/{dimension}_auc_scores.png")
