import numpy as np
import datetime
import sys
import random


class DataHandler(object):
    # Object for creating inverse covariance matrices
    # and covariance matrices from network files,
    # creating synthetic data sets,
    # writing solver results

    """ Initialize attributes """

    def __init__(self):
        self.inverse_sigmas = []
        self.sigmas = []
        self.network_files = []

    """ Reads a network in given file and generates
        inverse covariance matrices. Expected format for
        networks in given files is:
        [start node index],[end node index],[edge weight]  """

    def generate_network(self, filename, perturbed_filename, node, edge):
        max_possible = node * (node - 1) // 2
        if edge > max_possible:
            raise ValueError(f"Error: 无法生成 {edge} 条边，最多只能生成 {max_possible} 条边")

        # 生成所有可能的无向边（u < v）
        possible_edges = [(u, v) for u in range(1, node + 1) for v in range(u + 1, node + 1)]

        # 随机选择指定数量的边
        selected_edges = random.sample(possible_edges, edge)

        precision_matrix = np.eye(node)

        # 生成带权重的数据
        for u, v in selected_edges:
            weight = round(random.uniform(0.5, 1), 1)
            precision_matrix[u - 1, v - 1] = weight
            precision_matrix[v - 1, u - 1] = weight

        perturbed_node = random.randint(0, node - 1)
        perturbed_precision_matrix = precision_matrix.copy()
        for i in range(node):
            if i == perturbed_node:
                continue
            perturbed_precision_matrix[i, perturbed_node] = 0
            perturbed_precision_matrix[perturbed_node, i] = 0

        eigvals_min = min(np.linalg.eigvals(precision_matrix).min(), np.linalg.eigvals(perturbed_precision_matrix).min())
        precision_matrix += (0.1 + np.abs(eigvals_min)) * np.eye(node)
        perturbed_precision_matrix += (0.1 + np.abs(eigvals_min)) * np.eye(node)

        # eigvals_max = max(np.linalg.eigvals(precision_matrix).max(), np.linalg.eigvals(perturbed_precision_matrix).max())
        # precision_matrix = precision_matrix / eigvals_max
        # perturbed_precision_matrix = perturbed_precision_matrix / eigvals_max

        graph_data = []
        for i in range(node):
            for j in range(i, node):
                if precision_matrix[i][j] != 0:
                    graph_data.append((i + 1, j + 1, precision_matrix[i][j]))

        graph_data = sorted(graph_data, key=lambda x: (x[0], x[1]))

        perturbed_graph_data = []
        for i in range(node):
            for j in range(i, node):
                if perturbed_precision_matrix[i][j] != 0:
                    perturbed_graph_data.append((i + 1, j + 1, perturbed_precision_matrix[i][j]))

        perturbed_graph_data = sorted(perturbed_graph_data, key=lambda x: (x[0], x[1]))

        with open(filename, "w") as f:
            f.write("#First node,Second node,Weight\n")
            for data in graph_data:
                f.write(f"{data[0]},{data[1]},{data[2]}\n")

        with open(perturbed_filename, "w") as f:
            f.write("#First node,Second node,Weight\n")
            for data in perturbed_graph_data:
                f.write(f"{data[0]},{data[1]},{data[2]}\n")

    def read_network(self, filename, comment="#", splitter=",",
                     inversion=True):
        nodes = []
        self.network_files.append(filename)
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if comment in line:
                    continue
                data = line.split(splitter)
                if data[0] not in nodes:
                    nodes.append(int(data[0]))
                if data[1] not in nodes:
                    nodes.append(int(data[1]))
        self.dimension = max(nodes)
        network = np.zeros((self.dimension, self.dimension))
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if comment in line:
                    continue
                data = line.split(splitter)
                network[int(data[0]) - 1, int(data[1]) - 1] = float(data[2])
                network[int(data[1]) - 1, int(data[0]) - 1] = float(data[2])
        self.inverse_sigmas.append(network)
        if inversion:
            sigma = np.linalg.inv(network)
            print(np.linalg.eigvals(sigma))
            self.sigmas.append(sigma)
            print(sigma)
            print(np.shape(sigma))
            print(network)

    """ Generates a data file (.csv) from networks previously defined in
        self.sigmas (covariance matrix) """

    def generate_real_data(self, counts=[100, 100]):
        if len(counts) is not len(self.sigmas):
            raise Exception(
                "Lengths of networks and data lengths do not match.")
        z = None
        total_count = 0
        for sigma, datacount in zip(self.sigmas, counts):
            x = np.random.multivariate_normal(np.zeros(self.dimension),
                                              sigma, datacount)
            total_count += datacount
            if z is None:
                z = x
            else:
                z = np.vstack((z, x))
        filename = (f"synthetic_data/"
                    f"{self.dimension}x{total_count}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
        header = "# Data generated from networks:\n# "
        for f, datacount in zip(self.network_files, counts):
            header += f"{f}: {datacount}, "
        header = header[:-2]
        header += "\n"
        with open(filename, "w") as new_file:
            new_file.write(header)
            for datarow in z:
                line = ""
                for value in datarow:
                    line += "," + str("{0:.4f}".format(value))
                line = line[1:]
                new_file.write(f"{line}\n")

    """ Converts a network into matrix form """

    def write_network_to_matrix_form(self, datafile):
        new_filename = datafile.split(".")[0] + "_matrix.csv"
        self.read_network(datafile, inversion=False)
        with open(new_filename, "w") as f:
            for sig in self.inverse_sigmas:
                for i in range(np.shape(sig)[0]):
                    for j in range(np.shape(sig)[0]):
                        f.write(str(sig[i, j]) + ",")
                    f.write("\n")
                f.write("\n\n")

    """ Creates a file containing results, with network details,
        from converged algorithm instance """

    def write_network_results(self, datafile, solver, splitter=","):
        run_time = datetime.datetime.now()
        results_name = "network_results/%s_la%sbe%s_%s.csv" % (
            solver.__class__.__name__, int(solver.lambd),
            int(solver.beta), run_time.strftime("%Y%m%d%H%M%S"))
        """ Read features """
        with open(datafile, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    feats = line.strip().split(splitter)[1:]
                    break
        features = {}
        for i, feat in enumerate(feats):
            features[i] = feat
        """ Write Results """
        with open(results_name, "w") as f:
            f.write("# Information\n")
            f.write("Run datetime, %s\n" %
                    run_time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("Data file, %s\n" % datafile)
            f.write("Solver type, %s\n" % solver.__class__.__name__)
            f.write("Penalty function, %s\n" % solver.penalty_function)
            f.write("Data dimension, %s\n" % solver.dimension)
            f.write("Blocks, %s\n" % solver.blocks)
            f.write("Observations in a block, %s\n" % solver.obs)
            f.write("Rho, %s\n" % solver.rho)
            f.write("Beta, %s\n" % solver.beta)
            f.write("Lambda, %s\n" % solver.lambd)
            f.write("Processes used, %s\n" % solver.processes)
            f.write("\n")
            f.write("# Results\n")
            f.write("Algorithm run time, %s seconds\n" % solver.run_time)
            f.write("Iterations to complete, %s\n\n" % solver.iteration)
            try:
                f.write("Temporal deviations ratio (max/mean), {0:.3f}\n"
                        .format(solver.dev_ratio))
            except ValueError:
                f.write("Temporal deviations ratio (max/mean), %s\n"
                        % solver.dev_ratio)
            f.write("Temporal deviations ")
            for dev in solver.deviations:
                try:
                    f.write(",{0:.3f}".format(dev))
                except ValueError:
                    f.write(",%s" % dev)
            f.write("\nNormalized Temporal deviations ")
            for dev in solver.norm_deviations:
                try:
                    f.write(",{0:.3f}".format(dev))
                except ValueError:
                    f.write(",%s" % dev)
            """ Write networks """
            f.write("\n\n#Networks:\n\n")
            for k in range(solver.blocks):
                f.write("Block %s," % k)
                f.write(solver.blockdates[k] + "\n")
                if k > 0:
                    f.write("Dev to prev,")
                    f.write("{0:.3f},".format(solver.deviations[k - 1]))
                if k < solver.blocks - 1:
                    f.write("Dev to next,")
                    f.write("{0:.3f}".format(solver.deviations[k]))
                f.write("\n")
                for feat in feats:
                    f.write("," + feat)
                f.write("\n")
                for i in range(solver.dimension):
                    f.write(features[i])
                    for j in range(solver.dimension):
                        f.write("," + str(solver.thetas[k][i, j]))
                    f.write("\n")
                f.write("\n\n")

    """ Creates a file containing results, without network details,
        from converged solver instance """

    def write_results(self, datafile, solver, splitter=','):
        run_time = datetime.datetime.now()
        results_name = "results/%s_la%sbe%s_%s.csv" % (
            solver.__class__.__name__, int(solver.lambd),
            int(solver.beta), run_time.strftime("%Y%m%d%H%M%S"))
        with open(results_name, "w") as f:
            f.write("# Information\n")
            f.write("Run datetime, %s\n" %
                    run_time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("Data file, %s\n" % datafile)
            f.write("Solver type, %s\n" % solver.__class__.__name__)
            f.write("Penalty function, %s\n" % solver.penalty_function)
            f.write("Data dimension, %s\n" % solver.dimension)
            f.write("Blocks, %s\n" % solver.blocks)
            f.write("Observations in a block, %s\n" % solver.obs)
            f.write("Rho, %s\n" % solver.rho)
            f.write("Beta, %s\n" % solver.beta)
            f.write("Lambda, %s\n" % solver.lambd)
            f.write("Processes used, %s\n" % solver.processes)
            f.write("Total edges, %s\n" % solver.real_edges)
            f.write("Total edgeless, %s\n" % solver.real_edgeless)
            f.write("\n")
            f.write("# Results\n")
            f.write("Algorithm run time, %s seconds\n" % solver.run_time)
            f.write("Iterations to complete, %s\n\n" % solver.iteration)
            f.write("Correct positive edges, %s\n" % solver.correct_positives)
            f.write("All positives, %s\n" % solver.all_positives)
            f.write("F1 Score, {0:.3f}\n"
                    .format(solver.f1score))
            try:
                f.write("Temporal deviations ratio (max/mean), {0:.3f}\n"
                        .format(solver.dev_ratio))
            except ValueError:
                f.write("Temporal deviations ratio (max/mean), %s\n"
                        % solver.dev_ratio)
            f.write("Temporal deviations ")
            for dev in solver.deviations:
                try:
                    f.write(",{0:.3f}".format(dev))
                except ValueError:
                    f.write(",%s" % dev)
            f.write("\nNormalized Temporal deviations ")
            for dev in solver.norm_deviations:
                try:
                    f.write(",{0:.3f}".format(dev))
                except ValueError:
                    f.write(",%s" % dev)
            f.write("\n")


if __name__ == "__main__" and len(sys.argv) % 2 == 1:

    # Input arguments need to be pairwise.
    # First item of the pair is the network file.
    # Second item of the pair is number of datapoints
    # to create from the given network.
    # Arbitrary number of pair can be inputted.

    dh = DataHandler()

    dh.generate_network("networks/network6.csv", "networks/network7.csv", 100, 600)

    # data_counts = []
    # for i in range(1, len(sys.argv), 2):
    #     dh.read_network(sys.argv[i])
    #     data_counts.append(int(sys.argv[i + 1]))
    # if len(data_counts) > 0:
    #     dh.generate_real_data(data_counts)

    dh.read_network("networks/network6.csv")
    for i in range(250, 5001, 250):
        dh.generate_real_data([i])
