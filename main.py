import numpy as np
from numpy.random import choice as np_choice

class colonia:
    def __init__(self, distancia, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):  # Corrigido o nome do construtor
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

            self.pheromone *= self.decay  # Decay pheromone after each iteration

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
            path = self.gen_path(0)
            dist = self.gen_path_dist(path)
            all_paths.append((path, dist))
            
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start

        for _ in range(len(self.distancia) - 1):
            move = self.pick_move(self.pheromone[prev], self.distancia[prev], visited)
            path.append(move)
            prev = move
            visited.add(move)

        path.append(start)
        return path

    def pick_move(self, feromonio, dist, visited):
        feromonio = np.copy(feromonio)
        feromonio[list(visited)] = 0
        row = feromonio ** self.alpha * ((1.0 / dist) ** self.beta)
        row_sum = row.sum()

        if row_sum == 0:
            return np.random.choice(self.all_inds)

        norm_row = row / row_sum
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move




""" # Definindo o número de locais
n_locais = 10

# Criando uma matriz de distâncias com valores aleatórios entre 1 e 100 e mudando o tipo para float
distancia = np.random.randint(1, 101, size=(n_locais, n_locais)).astype(float)

# Definindo a diagonal principal como um valor grande (mas não infinito)
np.fill_diagonal(distancia, 1000)  # Definindo um valor suficientemente grande para evitar auto-loops

# Tornando a matriz simétrica (garantindo que todos os locais estejam conectados)
distancia = np.minimum(distancia, distancia.T)

# Parâmetros do algoritmo (constantes)
 """
n_best = 2
n_iterations = 20
decay = 0.5
alpha = 1
beta = 5
# 21 + 10 + 18 + 15 + 
distancia = np.array([
    [ 9999,  8, 13,  9999,  9999, 14,  9999,  8,  9999,  9999],
    [ 8,  9999,  9, 12,  9999,  9999,  9999,  9999,  9999, 11],
    [13,  9,  9999,  9999, 13, 15,  9999,  9999,  9999,  9999],
    [ 9999, 12,  9999,  9999, 19,  9999,  9999,  9999,  9999,  9999],
    [ 9999,  9999, 13, 19,  9999, 15,  9999,  9999,  9999,  9999],
    [14,  9999, 15,  9999, 15,  9999, 22, 18,  9999,  9999],
    [ 9999,  9999,  9999,  9999,  9999, 22,  9999,  9999, 21,  9999],
    [ 8,  9999,  9999,  9999,  9999, 18,  9999,  9999, 19999,  8],
    [ 9999,  9999,  9999,  9999,  9999,  9999, 21, 19999,  9999, 12],
    [ 9999, 11,  9999,  9999,  9999,  9999,  9999,  8, 12,  9999]
])

distancia = np.where(distancia == 0, 1000, distancia)

# Testando diferentes quantidades de formigas

for n_formigas in [100, 200, 300, 500]:
    print(f"\nExecutando o ACO com {n_formigas} formigas:\n")
    ant_colony = colonia(distancia, n_formigas, n_best, n_iterations, decay, alpha, beta)
    best_path, best_distance = ant_colony.run()
    print(f"Melhor caminho com {n_formigas} formigas: {best_path}")
    print(f"Melhor distância com {n_formigas} formigas: {best_distance}")
    
print(distancia)
