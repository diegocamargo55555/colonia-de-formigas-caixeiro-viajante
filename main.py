import numpy as np
from numpy.random import choice as np_choice

class colonia:
    def __init__(self, distancia, n_ants, n_best, n_iterations, decay, alpha=1, beta=1): 
        self.distancia = distancia
        self.feromonio = np.ones(self.distancia.shape) / len(distancia)
        self.all_inds = range(len(distancia))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        menor_caminho = None
        menor_distancia = np.inf

        for i in range(self.n_iterations):
            all_paths = self.gerar_all_paths()
            self.spread_feromonio(all_paths, self.n_best)
            caminho_atual = min(all_paths, key=lambda x: x[1])

            print(f"Iteração {i + 1}: {caminho_atual}")

            if menor_caminho is None or caminho_atual[1] < menor_distancia:
                menor_caminho = caminho_atual[0]
                menor_distancia = caminho_atual[1]

            self.feromonio *= self.decay

        return menor_caminho, menor_distancia

    def spread_feromonio(self, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for i in range(len(path) - 1):
                move = (path[i], path[i + 1])
                self.feromonio[move] += 1.0 / dist
            move = (path[-1], path[0])
            self.feromonio[move] += 1.0 / dist

    def gerar_path_dist(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distancia[path[i], path[i + 1]]
        total_dist += self.distancia[path[-1], path[0]]
        return total_dist

    def gerar_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            start_node = np.random.randint(0, len(self.distancia))
            path = self.gerar_path(start_node)
            dist = self.gerar_path_dist(path)
            all_paths.append((path, dist))
            
        return all_paths

    def gerar_path(self, start):
        path = [start]
        visited = set()
        visited.add(start)
        prev = start

        for _ in range(len(self.distancia) - 1):
            move = self.escolher_movimento(self.feromonio[prev], self.distancia[prev], visited)
            path.append(move)
            prev = move
            visited.add(move)
        return path

    def escolher_movimento(self, feromonio, dist, visited):
        feromonio = np.copy(feromonio)
        feromonio[list(visited)] = 0
        
        row = feromonio ** self.alpha * ((1.0 / dist) ** self.beta)
        row_sum = row.sum()

        if row_sum == 0:
            movimentos = list(set(self.all_inds) - visited)
            if movimentos:
                return np_choice(movimentos)
            else:
                return np_choice(self.all_inds)

        norm_row = row / row_sum
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move

n_best = 3
n_iterations = 20
decay = 0.5
alpha = 1
beta = 5

import numpy as np

distancia = np.array([
    # 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
    [ 0, 20,  0,  0,  0,  0,  0, 29,  0,  0,  0, 29,  0,  0,  0,  0,  0,  0], # 1
    [20,  0, 25,  0,  0,  0,  0, 28,  0,  0,  0, 39,  0,  0,  0,  0,  0,  0], # 2
    [ 0, 25,  0, 25,  0,  0,  0, 30,  0,  0,  0,  0, 54,  0,  0,  0,  0,  0], # 3
    [ 0,  0, 25,  0,  0, 32, 42,  0, 23, 33,  0,  0,  0,  0,  0,  0,  0,  0], # 4
    [ 0,  0,  0,  0,  0, 12, 26,  0,  0, 19, 30,  0,  0,  0,  0,  0,  0,  0], # 5
    [ 0,  0,  0, 32, 12,  0, 17,  0,  0, 35,  0,  0,  0,  0,  0,  0,  0,  0], # 6
    [ 0,  0,  0, 42, 26, 17,  0,  0,  0,  0, 38,  0,  0,  0,  0,  0,  0,  0], # 7
    [29, 28, 30,  0,  0,  0,  0,  0,  0,  0,  0, 37, 22,  0,  0,  0,  0,  0], # 8
    [ 0,  0,  0, 23,  0,  0,  0,  0,  0, 26,  0,  0, 34, 56,  0, 43,  0,  0], # 9
    [ 0,  0,  0, 33, 19, 35,  0,  0, 26,  0, 24,  0,  0, 30, 19,  0,  0,  0], # 10
    [ 0,  0,  0,  0, 30,  0, 38,  0,  0, 24,  0,  0,  0,  0, 26,  0,  0, 36], # 11
    [29, 39,  0,  0,  0,  0,  0, 37,  0,  0,  0,  0, 27,  0,  0, 43,  0,  0], # 12
    [ 0,  0, 54,  0,  0,  0,  0, 22, 34,  0,  0, 27,  0, 24,  0, 19,  0,  0], # 13
    [ 0,  0,  0,  0,  0,  0,  0,  0, 56, 30,  0,  0, 24,  0, 20, 19, 17,  0], # 14
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 19, 26,  0,  0, 20,  0,  0, 18, 21], # 15
    [ 0,  0,  0,  0,  0,  0,  0,  0, 43,  0,  0, 43, 19, 19,  0,  0, 26,  0], # 16
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 17, 18, 26,  0, 15], # 17
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 36,  0,  0,  0, 21,  0, 15,  0]  # 18
])
distancia = np.where(distancia == 0, 999, distancia)
distancia = np.minimum(distancia, distancia.T)


# Testando diferentes quantidades de formigas
for n_ant in [100, 200, 300, 500]:
    print(f"\nExecutando o ACO com {n_ant} formigas:\n")
    ant_colony = colonia(distancia, n_ant, n_best, n_iterations, decay, alpha, beta)
    best_path, best_distance = ant_colony.run()
    
    path_ajustado = [no + 1 for no in best_path]
    print(f"Melhor caminho com {n_ant} formigas: {path_ajustado}")
    print(f"Melhor distância com {n_ant} formigas: {best_distance}")
    
