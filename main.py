import numpy as np
from numpy.random import choice as np_choice

class colonia:
    def __init__(self, grafo, n_ants, n_melhores, n_iteracoes, evaporacao, alfa=1, beta=1):  
        self.grafo = grafo
        self.feromonio = np.ones(self.grafo.shape) / len(grafo)
        self.all_inds = range(len(grafo))
        self.n_ants = n_ants
        self.n_melhores = n_melhores
        self.n_iteracoes = n_iteracoes
        self.evaporacao = evaporacao
        self.alfa = alfa
        self.beta = beta

    def run(self):
        all_menor_caminho = None
        all_menor_distancia = np.inf

        for i in range(self.n_iteracoes):
            all_caminhos = self.gerar_todos_caminhos()
            self.depositar_feromonio(all_caminhos, self.n_melhores)
            menor_caminho = min(all_caminhos, key=lambda x: x[1])

            print(f"Iteração {i + 1}: {menor_caminho}")

            if all_menor_caminho is None or menor_caminho[1] < all_menor_distancia:
                all_menor_caminho = menor_caminho[0]
                all_menor_distancia = menor_caminho[1]

            self.feromonio *= self.evaporacao  

        return all_menor_caminho, all_menor_distancia

    def depositar_feromonio(self, all_caminhos, n_melhores):
        caminhos_ordenados = sorted(all_caminhos, key=lambda x: x[1])
        for caminho, dist in caminhos_ordenados[:n_melhores]:
            for i in range(len(caminho) - 1):
                move = (caminho[i], caminho[i + 1])
                self.feromonio[move] += 1.0 / dist

            move = (caminho[-1], caminho[0])
            self.feromonio[move] += 1.0 / dist

    def get_distancia(self, caminho):
        total_dist = 0
        for i in range(len(caminho) - 1):
            total_dist += self.grafo[caminho[i], caminho[i + 1]]
        total_dist += self.grafo[caminho[-1], caminho[0]]
        return total_dist

    def gerar_todos_caminhos(self):
        all_caminhos = []
        for _ in range(self.n_ants):
            caminho = self.gerar_caminho(0)
            dist = self.get_distancia(caminho)
            all_caminhos.append((caminho, dist))
        return all_caminhos

    def gerar_caminho(self, start):
        caminho = []
        visitado = set()
        visitado.add(start)
        prev = start

        for _ in range(len(self.grafo) - 1):
            move = self.escolher_movimento(self.feromonio[prev], self.grafo[prev], visitado)
            caminho.append(move)
            prev = move
            visitado.add(move)

        caminho.append(start)
        return caminho

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


n_locais = 13
grafo = np.random.randint(1, 101, size=(n_locais, n_locais)).astype(float)
np.fill_diagonal(grafo, 1000)  # Definindo um valor suficientemente grande para evitar auto-loops
grafo = np.minimum(grafo, grafo.T)
print(grafo)
n_melhores = 3
n_iteracoes = 50
evaporacao = 0.5
alfa = 1
beta = 10

# Testando diferentes quantidades de formigas
for n_ants in [100, 200, 400, 800]:
    print(f"\nExecutando o ACO com {n_ants} formigas:\n")
    ant_colony = colonia(grafo, n_ants, n_melhores, n_iteracoes, evaporacao, alfa, beta)
    best_caminho, best_distancia = ant_colony.run()
    print(f"Melhor caminho com {n_ants} formigas: {best_caminho}")
    print(f"Melhor distância com {n_ants} formigas: {best_distancia}")
