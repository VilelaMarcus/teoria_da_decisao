# -*- coding: utf-8 -*-
"""
Grupo:

*   Bernardo Araujo Ribeiro
*   Marcus Vinicius Ferreira Vilela
*   Maria Luiza Coelho Rodrigues
"""

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import copy as cp
class Struct:
    pass

"""Dados Clientes"""


file_path = './clientes.csv'
clientes_dataset = pd.read_csv(file_path, header=None, names=['x', 'y', 'bandwidth'])

"""Dados Gerais"""





class ProblemData:
    def __init__(self, clients, max_p_as, pa_capacity, max_distance, lambda_exposure, gamma):
        self.clients = clients
        self.max_p_as = max_p_as
        self.pa_capacity = pa_capacity
        self.max_distance = max_distance
        self.lambda_exposure = lambda_exposure
        self.gamma = gamma
        self.pas = []  # To store access point positions
        self.assignments = []  # To store access point positions

    def initialize_solution(self, use_constructive_heuristic=True, qtd_pas_inicial=20):
        np.random.seed(42)  # For reproducibility
        grid_spacing = 5  # Grid spacing in meters

        if use_constructive_heuristic:

            # Número de PAs (pontos de acesso) desejados
            num_pas = qtd_pas_inicial

            # Defina a área e o intervalo do grid
            area_size = 400
            grid_interval = 5

            # Crie os pontos do grid
            grid_points = [(i, j) for i in range(0, area_size + grid_interval, grid_interval)
                            for j in range(0, area_size + grid_interval, grid_interval)]
            grid_points = np.array(grid_points)

            # Execute o K-means
            kmeans = KMeans(n_clusters=num_pas, random_state=0).fit(clientes_dataset[['x','y']])

            # Obtenha os centros dos clusters (onde os PAs serão colocados)
            centros = kmeans.cluster_centers_

            # Função para encontrar o ponto do grid mais próximo
            def find_nearest_grid_point(point, grid_points):
                distances = np.sqrt((grid_points[:, 0] - point[0])**2 + (grid_points[:, 1] - point[1])**2)
                nearest_index = np.argmin(distances)
                return grid_points[nearest_index]

            # Encontre os pontos do grid mais próximos dos centros dos clusters
            nearest_grid_points = np.array([find_nearest_grid_point(centro, grid_points) for centro in centros])

            for i in range(len(nearest_grid_points)):
                x = nearest_grid_points[i][0]
                y = nearest_grid_points[i][1]
                self.pas.append((x, y))


            while len(self.pas) < self.max_p_as :
                x = 999
                y = 999
                self.pas.append((x,y))
                i +=1;


        else:
            # Randomly distribute access points within the convention center area
            area_width = 400
            grid_points_x = np.arange(0, area_width + 1, grid_spacing)
            grid_points_y = np.arange(0, area_width + 1, grid_spacing)

            for _ in range(self.max_p_as):
                x = np.random.choice(grid_points_x)
                y = np.random.choice(grid_points_y)
                self.pas.append((x, y))

        # Initialize assignments based on client proximity and capacity
        pa_bandwidth_usage = {i: 0 for i in range(self.max_p_as)}

        for index, client in self.clients.iterrows():
            assigned = False
            distances = [np.sqrt((pa[0] - client['x'])**2 + (pa[1] - client['y'])**2) for pa in self.pas]
            possible_pas = sorted(range(len(distances)), key=lambda k: distances[k])

            for pa_index in possible_pas:
                if distances[pa_index] <= self.max_distance and pa_bandwidth_usage[pa_index] + client['bandwidth'] <= self.pa_capacity:
                    self.assignments.append(pa_index)
                    pa_bandwidth_usage[pa_index] += client['bandwidth']
                    assigned = True
                    break

            if not assigned:
                # If no PA can accommodate the client, don't assign it
                self.assignments.append(-1)




    def plot_initial_solution(self):
        x_pos, y_pos = zip(*self.pas)
        plt.figure(figsize=(8, 8))
        plt.scatter(x_pos, y_pos, alpha=0.8, color='red')
        plt.title('Distribuição Espacial dos PAs na Solução Inicial')
        plt.xlabel('Posição X')
        plt.ylabel('Posição Y')
        plt.grid(True, which='major', linestyle='-', linewidth=0.5)
        plt.axis([0, 400, 0, 400])
        plt.xticks(range(0, 401, 25))
        plt.yticks(range(0, 401, 25))
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5)

"""Solução Inicial"""

def sol_inicial(probdata, use_constructive_heuristic=True, qtd_pas_inicial=20):

    solution = ProblemData(
        clients=probdata.clients,
        max_p_as=probdata.max_p_as,
        pa_capacity=probdata.pa_capacity,
        max_distance=probdata.max_distance,
        lambda_exposure=probdata.lambda_exposure,
        gamma=probdata.gamma
    )

    solution.initialize_solution(use_constructive_heuristic, qtd_pas_inicial)
    # solution.plot_initial_solution()

    return solution

"""Modelando Função Objetivo Distância"""

def fobj(solution, probdata, n_pas):
    active_pas = set() # Inicialize active PAs
    number_of_clients = len(probdata.clients)
    allowed_unserved = int(0.02 * number_of_clients)  # 2% of the clientes it's allowed to be unserved
    unserved_clients = 0
    total_distance = 0 # Inicialize distance
    total_penalty = 0  # Inicialize penalty

    for client_index, pa_index in enumerate(solution.assignments):
        if pa_index != -1:
            # # Client info
            client = probdata.clients.iloc[client_index]
            pa = solution.pas[pa_index]

            # Verify distance
            distance = np.sqrt((pa[0] - client['x'])**2 + (pa[1] - client['y'])**2)

            # Add active PA
            active_pas.add(pa_index)

            #Add distance
            total_distance += distance
        else:
            unserved_clients += 1

    n_pas_ativos = len(list(set(item for item in solution.assignments if item != -1)))

    if n_pas_ativos != n_pas:
        total_penalty += ((n_pas-n_pas_ativos)*(n_pas-n_pas_ativos))*10000

    if unserved_clients > allowed_unserved:
        total_penalty += (unserved_clients*10000)

    # Sum distance and penalty
    solution.fitness = total_distance + total_penalty
    
    return solution

"""Neighborhood Change"""

def neighborhoodChange(x, y, k):
    if y.fitness < x.fitness or (y.fitness == x.fitness and np.random.rand() < 0.5):
        x = cp.deepcopy(y)
        k = 1
    else:
        k += 1
    return x, k

"""Vizinhanças"""

def shake(solution, k, probdata):
    np.random.seed()

    new_solution = cp.deepcopy(solution)  # Copy the solution to change it
    grid_spacing = 5
    area_width = 400
    area_height = 400
    # Randomly Realocate  PA's
    if k == 1:
            pa_index = np.random.randint(len(new_solution.pas))
            pa = new_solution.pas[pa_index]
            if pa[0] != 999:
              new_x = np.random.choice(range(max(0,pa[0]-10), min(area_width+1,pa[0]+11), grid_spacing))
              new_y = np.random.choice(range(max(0,pa[1]-10), min(area_height+1,pa[1]+11), grid_spacing))
              new_solution.pas[pa_index] = (new_x, new_y)

    elif k == 2:
            pa_index = np.random.randint(len(new_solution.pas))
            pa = new_solution.pas[pa_index]
            if pa[0] != 999:
              new_x = np.random.choice(range(max(0,pa[0]-50), min(area_width+1,pa[0]+51), grid_spacing))
              new_y = np.random.choice(range(max(0,pa[1]-50), min(area_height+1,pa[1]+51), grid_spacing))
              new_solution.pas[pa_index] = (new_x, new_y)

    elif k == 3:
            if np.random.rand() < 0.5:
              stop_while = False
              i = len(new_solution.pas)
              while stop_while == False and i > 0:
                pa_index = np.random.randint(len(new_solution.pas))
                pa = new_solution.pas[pa_index]
                if pa[0] != 999:
                  new_x = 999
                  new_y = 999
                  new_solution.pas[pa_index] = (new_x, new_y)
                  stop_while = True
                else:
                  stop_while = False
                i -= 1
            else:
              i = len(new_solution.pas)
              stop_while = False
              while stop_while == False and i > 0:
                pa_index = np.random.randint(len(new_solution.pas))
                pa = new_solution.pas[pa_index]
                if pa[0] == 999:
                  new_x = np.random.choice(range(0, area_width + 1, grid_spacing))
                  new_y = np.random.choice(range(0, area_height + 1, grid_spacing))
                  new_solution.pas[pa_index] = (new_x, new_y)
                  stop_while = True
                else:
                  stop_while = False
                i -= 1

    # Initialize assignments based on client proximity and capacity
    new_solution.assignments = []
    pa_bandwidth_usage = {i: 0 for i in range(new_solution.max_p_as)}

    for index, client in new_solution.clients.iterrows():
        assigned = False
        distances = [np.sqrt((pa[0] - client['x'])**2 + (pa[1] - client['y'])**2) for pa in new_solution.pas]
        possible_pas = sorted(range(len(distances)), key=lambda k: distances[k])

        for pa_index in possible_pas:
           if distances[pa_index] <= new_solution.max_distance and pa_bandwidth_usage[pa_index] + client['bandwidth'] <= new_solution.pa_capacity:
              new_solution.assignments.append(pa_index)
              pa_bandwidth_usage[pa_index] += client['bandwidth']
              assigned = True
              break

        if not assigned:
           # If no PA can accommodate the client, don't assign it
           new_solution.assignments.append(-1)

    pa_counts = {}
    for pa_id in new_solution.assignments:
      if pa_id != -1:
            if pa_id in pa_counts:
                pa_counts[pa_id] += 1
            else:
                pa_counts[pa_id] = 1
    # Verificar modems com menos de 3 alocações
    pa_para_desalocar = [modem for modem, count in pa_counts.items() if count < 5]

    # Desalocar clientes dos modems com menos de 3 alocações
    new_solution.assignments = [-1 if modem in pa_para_desalocar else modem for modem in new_solution.assignments]


    return new_solution

"""Best Improvement"""

def bestImprovement(current_solution, kmax, probdata):
    best_solution = cp.deepcopy(current_solution)

    for i in range(1, kmax + 1):
        neighbor_solution = shake(best_solution, i, probdata)
        if neighbor_solution.fitness < best_solution.fitness or (neighbor_solution.fitness == best_solution.fitness and np.random.rand() < 0.5):
            best_solution = neighbor_solution

    return best_solution

def plote_grafico(dados, titulo):
    # Convertendo os valores para float
    distancias = [item[0] for item in dados]
    valores = [float(item[1]) for item in dados]

    # Plotando o gráfico de curva
    plt.figure(figsize=(8, 5))  # Tamanho da figura (opcional)
    plt.plot(distancias, valores, marker='o', linestyle='-', label='Valores')
    plt.xlabel('Distância')
    plt.ylabel('Valores')
    plt.title(titulo)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Exibindo o gráfico
    plt.show()


def removeDominatedSolutions (solutionsList):
    for solution in solutionsList:
        for solution2 in solutionsList:
            if(solution2.f1 >= solution.f1 and solution2.f2 > solution.f2):
                solutionsList.remove(solution2)
    return solutionsList

def nondominatedsolution(solutionsList, currentSolution):
    if(len(solutionsList) == 0):
        return True
    for solution in solutionsList:
        if(currentSolution[0] >= solution.f1 and currentSolution[1] > solution.f2):
            return False
    return True


# def main():
#     times = 0
#     historico_fit_1 = []
#     historico_fit_2 = []
#     historico_fit_3 = []
#     historico_fit_4 = []
#     historico_fit_5 = []
#     while times < 5:
#         pas = 30

#         # max number of solutions
#         max_num_sol_avaliadas = 10

#         # Number of neighborhoods
#         kmax = 3

#         probdata = ProblemData(
#             clients=clientes_dataset,
#             max_p_as=30,
#             pa_capacity=54,
#             max_distance=85,
#             lambda_exposure=1,
#             gamma=1
#         )
#         # Create inicial solution
#         while pas > 10:
            
#             # solutions counter
#             num_sol_avaliadas = 0
            
#             x = sol_inicial(
#                 probdata,
#                 use_constructive_heuristic=True,
#                 qtd_pas_inicial = pas)

#             # Avaliate it
#             x = fobj(x,probdata, pas)
#             num_sol_avaliadas += 1

#             # Data to plot
#             historico = Struct()
#             historico.sol = []
#             historico.fit = []
#             historico.sol.append(x.pas)
#             historico.fit.append(x.fitness)
            
#             # Method
#             pas -= 1
#             while num_sol_avaliadas < max_num_sol_avaliadas:
#                 k = 1
#                 while k <= kmax:
#                     # Generate a solution in the current neighborhood
#                     y = shake(x,k,probdata)
#                     y = fobj(y,probdata, pas)
#                     z = bestImprovement(y,3,probdata)
#                     num_sol_avaliadas += 1
                    
#                     # Atualize it
#                     x,k = neighborhoodChange(x,z,k)

#             n_pas_ativos = len(list(set(item for item in x.assignments if item != -1)))
            
#             if times == 0:
#                 historico_fit_1.append([n_pas_ativos, format(x.fitness, '.1f')])
#             elif times == 1:
#                 historico_fit_2.append([n_pas_ativos, format(x.fitness, '.1f')])
#             elif times == 2:
#                 historico_fit_3.append([n_pas_ativos, format(x.fitness, '.1f')])
#             elif times == 3:
#                 historico_fit_4.append([n_pas_ativos, format(x.fitness, '.1f')])
#             else:
#                 historico_fit_5.append([n_pas_ativos, format(x.fitness, '.1f')])
#         times += 1
    
#     # Lista de cores para as cinco curvas
#     cores = ['blue', 'green', 'red', 'purple', 'orange']

#     # Criando uma figura para o gráfico
#     plt.figure(figsize=(10, 6))

#     # Plotando cada conjunto de dados com uma cor diferente
#     for i, historico_fit in enumerate([historico_fit_1, historico_fit_2, historico_fit_3, historico_fit_4, historico_fit_5]):
#         distancias = [item[0] for item in historico_fit]
#         valores = [item[1] for item in historico_fit]
#         plt.plot(distancias, valores, marker='o', linestyle='-', color=cores[i], label=f'Histórico {i+1}')

#     # Configurações adicionais do gráfico
#     plt.xlabel('Número de PA ativos')
#     plt.ylabel('Fitness')
#     plt.title('Histórico de Fitness por Número de PAs Ativos')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()

#     # Exibindo o gráfico
#     plt.show()




def main():
    # Number of neighborhoods
    kmax = 3

    # max number of solutions
    max_num_sol_avaliadas = 10
    
    probdata = ProblemData(
        clients=clientes_dataset,
        max_p_as=30,
        pa_capacity=54,
        max_distance=85,
        lambda_exposure=1,
        gamma=1
    )
    
    # Gera solução inicial
    initial_solution = sol_inicial(
        probdata,
        use_constructive_heuristic=True,
        qtd_pas_inicial = 30)

    # Contador do número de soluções candidatas avaliadas
    min_episolon = 9#6891
    max_episolon = 25#56510

    # Ciclo iterativo do método
    for color in ["b.", "g.", "r.", "y.", "k."]:
        solucoes = []
        num_iteracoes = 0
        max_iteracoes = 50
        while len(solucoes) < 20 and num_iteracoes < max_iteracoes:
            if(num_iteracoes > max_iteracoes):
                break
            num_iteracoes += 1
            print ("Num Solucoes: {} Num Iteracoes {}".format(len(solucoes), num_iteracoes))

            num_sol_avaliadas = 1
            episolon = random.randint(min_episolon, max_episolon)
            x = fobj(initial_solution,probdata, episolon)
            fit = None
            globalX = None

            while num_sol_avaliadas < max_num_sol_avaliadas:

                k = 1
                while k <= kmax:

                    # Gera uma solução candidata na k-ésima vizinhança de x
                    y = shake(x,k,probdata)
                    y = fobj(y,probdata, episolon)

                    y = bestImprovement(y,3,probdata)
                    num_sol_avaliadas += 1

                    # Atualiza solução corrente e estrutura de vizinhança (se necessário)
                    x,k = neighborhoodChange(x,y,k)
                    if fit == None or fit >= x.fitness:
                        fit = x.fitness
                        globalX = x
                    if num_sol_avaliadas > max_num_sol_avaliadas:
                        break

            isNonDominatedSolution = nondominatedsolution(solucoes, [globalX.f1, globalX.f2])
            if(isNonDominatedSolution == True):
                solucoes.append(globalX)
                solucoes = removeDominatedSolutions(solucoes)

        for i in solucoes:
            print("f1 {} f2 {}".format(i.f1, i.f2))
            plt.plot(i.f1,i.f2,color)
plt.title('Soluções estimadas')
plt.xlabel('f1(x)')
plt.ylabel('f2(x)')
plt.show()


if __name__ == "__main__":
    main()