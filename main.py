import numpy as np
from numpy.random import choice as np_choice

class colonia:
    def __init__(self, distancias, n_ants, n_melhores, n_iteracoes, evaporacao, alfa=1, beta=1):  
        self.distancias = distancias
        self.feromonio = np.ones(self.distancias.shape) / len(distancias)
        self.all_inds = range(len(distancias))
        self.n_ants = n_ants
        self.n_melhores = n_melhores
        self.n_iteracoes = n_iteracoes
        self.evaporacao = evaporacao
        self.alfa = alfa
        self.beta = beta

    def run(self):
        all_menor_path = None
        all_menor_distancia = np.inf

        for i in range(self.n_iteracoes):
            all_paths = self.gerar_todos_caminhos()
            self.depositar_feromonio(all_paths, self.n_melhores)
            menor_path = min(all_paths, key=lambda x: x[1])

            print(f"Iteração {i + 1}: {menor_path}")

            if all_menor_path is None or menor_path[1] < all_menor_distancia:
                all_menor_path = menor_path[0]
                all_menor_distancia = menor_path[1]

            self.feromonio *= self.evaporacao  

        return all_menor_path, all_menor_distancia

    def depositar_feromonio(self, all_paths, n_melhores):
        caminhos_ordenados = sorted(all_paths, key=lambda x: x[1])
        for path, dist in caminhos_ordenados[:n_melhores]:
            for i in range(len(path) - 1):
                move = (path[i], path[i + 1])
                self.feromonio[move] += 1.0 / dist

            move = (path[-1], path[0])
            self.feromonio[move] += 1.0 / dist

    def get_distancia(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distancias[path[i], path[i + 1]]
        total_dist += self.distancias[path[-1], path[0]]
        return total_dist

    def gerar_todos_caminhos(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.gerar_path(0)
            dist = self.get_distancia(path)
            all_paths.append((path, dist))
        return all_paths

    def gerar_path(self, start):
        path = []
        visitado = set()
        visitado.add(start)
        prev = start

        for _ in range(len(self.distancias) - 1):
            move = self.escolher_movimento(self.feromonio[prev], self.distancias[prev], visitado)
            path.append(move)
            prev = move
            visitado.add(move)

        path.append(start)
        return path

    def escolher_movimento(self, feromonio, dist, visitado):
        feromonio = np.copy(feromonio)
        feromonio[list(visitado)] = 0
        linha = feromonio ** self.alfa * ((1.0 / dist) ** self.beta)
        linha_sum = linha.sum()

        if linha_sum == 0:
            return np.random.choice(self.all_inds)

        normalizada_linha = linha / linha_sum
        move = np_choice(self.all_inds, 1, p=normalizada_linha)[0]
        return move


n_locais = 35
distancias = np.random.randint(1, 101, size=(n_locais, n_locais)).astype(float)
np.fill_diagonal(distancias, 1000)  # Definindo um valor suficientemente grande para evitar auto-loops
distancias = np.minimum(distancias, distancias.T)

n_melhores = 5
n_iteracoes = 200
evaporacao = 0.5
alfa = 1
beta = 5

# Testando diferentes quantidades de formigas
for n_ants in [50, 100, 150, 250]:
    print(f"\nExecutando o ACO com {n_ants} formigas:\n")
    ant_colony = colonia(distancias, n_ants, n_melhores, n_iteracoes, evaporacao, alfa, beta)
    best_path, best_distancia = ant_colony.run()
    print(f"Melhor caminho com {n_ants} formigas: {best_path}")
    print(f"Melhor distância com {n_ants} formigas: {best_distancia}")
