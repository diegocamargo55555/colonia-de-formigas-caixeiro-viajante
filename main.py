import numpy as np
from numpy.random import choice as np_choice

class colonia:
    def __init__(self, distancia, n_ants, n_best, n_iterations, decay, alpha=1, beta=1): 
        self.distancia = distancia
        self.pheromone = np.ones(self.distancia.shape) / len(distancia)
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
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best)
            caminho_atual = min(all_paths, key=lambda x: x[1])

            print(f"Iteração {i + 1}: {caminho_atual}")

            if menor_caminho is None or caminho_atual[1] < menor_distancia:
                menor_caminho = caminho_atual[0]
                menor_distancia = caminho_atual[1]

            self.pheromone *= self.decay

        return menor_caminho, menor_distancia

    def spread_pheromone(self, all_paths, n_best):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for i in range(len(path) - 1):
                move = (path[i], path[i + 1])
                self.pheromone[move] += 1.0 / dist
            move = (path[-1], path[0])
            self.pheromone[move] += 1.0 / dist

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distancia[path[i], path[i + 1]]
        total_dist += self.distancia[path[-1], path[0]]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            start_node = np.random.randint(0, len(self.distancia))
            path = self.gen_path(start_node)
            dist = self.gen_path_dist(path)
            all_paths.append((path, dist))
            
        return all_paths

    def gen_path(self, start):
        path = [start]
        visited = set()
        visited.add(start)
        prev = start

        for _ in range(len(self.distancia) - 1):
            move = self.pick_move(self.pheromone[prev], self.distancia[prev], visited)
            path.append(move)
            prev = move
            visited.add(move)
        return path

    def pick_move(self, feromonio, dist, visited):
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

distancia = np.array([
    # 1   2   3   4   5   6   7   8   9   10
    [ 0,  8, 13,  0,  0, 14,  0,  8,  0,  0], #1
    [ 8,  0,  9, 12,  0,  0,  0,  0,  0, 11], #2
    [13,  9,  0,  0, 13, 15,  0,  0,  0,  0], #3
    [ 0, 12,  0,  0, 19,  0,  0,  0,  0,  0], #4
    [ 0,  0, 13, 19,  0, 15,  0,  0,  0,  0], # 5
    [14,  0, 15,  0, 15,  0, 22, 18,  0,  0], #6
    [ 0,  0,  0,  0,  0, 22,  0,  0, 21,  0], #7
    [ 8,  0,  0,  0,  0, 18,  0,  0,  0,  8], #8
    [ 0,  0,  0,  0,  0,  0, 21,  0,  0, 12], #9
    [ 0, 11,  0,  0,  0,  0,  0,  8, 12,  0]  #10
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
    
