# epsilon_restrito2
from typing import List, Dict, Tuple
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy as cp
from collections import defaultdict
import random

clients_df = pd.read_csv('clientes.csv', header=None, names=['x', 'y', 'bandwidth'])

# Valores mínimos e máximos esperados para normalização
min_f1 = 5
max_f1 = 25
min_f2 = 5000
max_f2 = 50000

class Solution:
    pass

class Struct:
    pass

'''
Definição de dados iniciais do problema.
'''
def probdef(clients_df: pd.DataFrame = clients_df) -> Struct:
    probdata = Struct()

    probdata.clients = getClientsListFromDF(clients_df)
    probdata.max_pas = 30
    probdata.max_distance = 85
    probdata.pa_max_capacity = 54
    probdata.grid_spacing = 5
    probdata.width_area = 400
    probdata.height_area = 400
    probdata.num_clients = 495

    return probdata

'''
Parâmetros de entrada:
probdata = Os dados do problema
clients_df = Dataframe com as coordenadas e consumo de banda de cada cliente

Retorno: Uma solução candidata (A primeira do problema)
'''
def sol_inicial(probdata) -> Solution:
    start_time = time.time()

    solution = Solution()
    solution.pas = []
    solution.assignments = []
    grid_spacing = probdata.grid_spacing 
    max_radius = probdata.max_distance
    num_clients = probdata.num_clients
    minimum_clients_assigned = 0.7
    minimum_bandwidth_consumption = (0.5 * 54)  
    solution.clients = probdata.clients
    possible_assignments = []

    # Removendo possiveis atribuicoes de PAs a clientes
    removeAssigned(solution.clients)

    # Inicia uma posição aleatória para o PA dentro do grid de 400x400
    pa_pos_x = int(np.random.choice(np.arange(0, 401, grid_spacing)))
    pa_pos_y = int(np.random.choice(np.arange(0, 401, grid_spacing)))

    for pa_index in range(30): # Mudar para receber o valor que vem do probdata
        pa_bandwidth_usage = 0
        num_iterations = 0

        if (len(solution.assignments) >= (minimum_clients_assigned * num_clients)): 
            break

        while(pa_bandwidth_usage <= minimum_bandwidth_consumption):
            pa_bandwidth_usage = 0
            num_iterations += 1
            if (num_iterations >= 50000):
                minimum_bandwidth_consumption -= 0.01

            # Obter as distâncias entre os clientes e o PA em questão
            distances = getDistancesClientsPA(pa_pos_x, pa_pos_y, solution.clients)
            for obj in distances:
                if (pa_bandwidth_usage <= (54 - obj['bandwidth']) 
                    and obj['distance'] <= max_radius):
                    pa_bandwidth_usage += obj['bandwidth']
                    obj['pa_index'] = pa_index
                    possible_assignments.append({'x_pa': pa_pos_x, 
                                                 'y_pa': pa_pos_y, 
                                                 'x_client': obj['x'], 
                                                 'y_client': obj['y'],
                                                 'pa_index': pa_index,
                                                 'client_index': obj['client_index'],
                                                 'distance_client_pa': obj['distance']}) 
                else:
                    break
            
            if (pa_bandwidth_usage <= minimum_bandwidth_consumption):
                possible_assignments = []
                pa_bandwidth_usage = 0
                pa_pos_x, pa_pos_y = UpdatePAPosition(pa_pos_x, pa_pos_y)
            else:
                for possible_assignment in possible_assignments:
                    for client in solution.clients:
                        if client['client_index'] == possible_assignment['client_index']:
                            client['assigned'] = True
                            break

                solution.assignments.extend(possible_assignments)
                solution.pas.append((float(pa_pos_x), float(pa_pos_y)))
                pa_pos_x, pa_pos_y = UpdatePAPosition(pa_pos_x, pa_pos_y)

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print("\nTempo de execução para solução inicial: ", execution_time)

    return solution


'''
Zera a propriedade 'assigned' dos clientes para que possa ser feito um rearranjo.
'''
def removeAssigned(clients: List[dict]) -> List[dict]:
    for client in clients:
        client['assigned'] = False

    return clients

'''
Isolar essa parte do código para ser reusada em outros pontos do problema.
Esse trecho constrói uma solução para o problema a partir da posição dos PAs e dos clientes.
Retorna uma solução para o problema.
'''
def rearrangeClientsAndPAs(solution: Solution, clients_list: List) -> List[dict]:
    solution.assignments = []
    removeAssigned(clients_list)
    
    for index, value in enumerate(solution.pas):
        pa_bandwidth_usage = 0
        distances = getDistancesClientsPA(value[0], value[1], clients_list)
        for obj in distances:
            if (pa_bandwidth_usage <= (54 - obj['bandwidth']) and obj['distance'] <= 85):
                solution.assignments.append({
                    'x_pa': float(value[0]), 
                    'y_pa': float(value[1]), 
                    'x_client': float(obj['x']), 
                    'y_client': float(obj['y']),
                    'pa_index': index,
                    'client_index': obj['client_index'],
                    'distance_client_pa': obj['distance']
                })
                clients_list[obj['client_index']]['assigned'] = True
    
    return solution.assignments

'''
Parâmetros de entrada: Uma solução candidata
Retorno: Porcentagem de clientes não conectados
'''
def getPercentOfConnectedClients(assignments: List, clients_list: List) -> float:
    return (len(assignments)/len(clients_list)) * 100

'''
Parâmetros de entrada: Uma solução candidata
Retorno: Distância total entre clientes e os PAs aos quais estão atribuídos
'''
def getSumDistanceClientsAndPAs(assignments: List) -> int:
    return sum(assignment['distance_client_pa'] for assignment in assignments)

def getPenalityForUnservedClients(assignments: List, clients_list: List) -> int:
    penality = 0

    # Medir a porcentagem de clientes não servidos
    porc_unserved_clients = getPercentOfConnectedClients(assignments, clients_list)
    if (porc_unserved_clients < 98 and porc_unserved_clients >= 96):
        penality += 2
    elif (porc_unserved_clients < 96 and porc_unserved_clients >= 94):
        penality += 3
    elif (porc_unserved_clients < 94 and porc_unserved_clients >= 91):
        penality += 5
    elif (porc_unserved_clients < 91 and porc_unserved_clients >= 88):
        penality += 8
    elif (porc_unserved_clients < 88 and porc_unserved_clients >= 85):
        penality += 13
    elif (porc_unserved_clients < 85 and porc_unserved_clients >= 82):
        penality += 21
    elif (porc_unserved_clients < 79 and porc_unserved_clients >= 75):
        penality += 34
    else:
        penality += 55

    return penality

'''
Função objetivo 1
Parâmetros de entrada: Uma solução candidata e os dados do problema
Retorno: Um número inteiro representando o fitnesse da solução em questão
'''
def fobj1(solution) -> int:
    penality = 0

    # Medir a porcentagem de clientes não servidos
    penality += getPenalityForUnservedClients(solution.assignments, solution.clients)

    # Medir a quantidade de PAs que estão sendo usados (Relativo somente a função objetivo 1)
    qtd_pas = len(solution.pas)

    # Calcula o fitness da solução 
    fitness = qtd_pas + penality

    return fitness

'''
Função objetivo 2
Parâmetros de entrada: Uma solução candidata e os dados do problema
Retorno: Um número inteiro representando o fitnesse da solução em questão
'''
def fobj2(solution) -> float:
    penality = 0

    # Medir a porcentagem de clientes não servidos
    penality += getPenalityForUnservedClients(solution.assignments, solution.clients) * 500

    # Medir a distância total entre clientes e seus respectivos PAs
    total_distance = getSumDistanceClientsAndPAs(solution.assignments) * 0.2

    # Calcula o fitness da solução
    fitness = total_distance + penality 

    return fitness

'''
Parâmetros de entrada: Coordenadas do PA e lista de clientes
Retorno: Vetor de objetos, que possuem informações dos clientes, ordenado pela propriedade 'distance'
'''
def getDistancesClientsPA(pa_x: float, pa_y: float, clients: List):
    distances = []
    unassigned_clients = [client for client in clients if not client['assigned']]

    for client in unassigned_clients:
        distances.append({'distance': getDistance(pa_x, pa_y, float(client['x']), float(client['y'])), 
                        'x': float(client['x']), 'y': float(client['y']),
                        'bandwidth': float(client['bandwidth']),
                        'client_index': client['client_index']})

    distances = sorted(distances, key=lambda k: k['distance'])

    return distances

def getDistance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def UpdatePAPosition(x, y, grid_spacing: int = 5) -> int:
    direction = np.random.randint(0, 8)
    if (direction == 0):
        y += grid_spacing
    elif (direction == 1):
        x += grid_spacing
    elif (direction == 2):
        y -= grid_spacing
    elif (direction == 3):
        x -= grid_spacing
    elif (direction == 4):
        x -= grid_spacing
        y += grid_spacing
    elif (direction == 5):
        x += grid_spacing
        y += grid_spacing
    elif (direction == 6):
        x += grid_spacing
        y -= grid_spacing
    elif (direction == 7):
        x -= grid_spacing
        y -= grid_spacing

    # Limitar as coordenadas entre 0 e 400
    x = max(0, min(x, 400))
    y = max(0, min(y, 400))

    return float(x), float(y)

def getClientsListFromDF(clients_df: pd.DataFrame) -> List:
    clients = []

    for index, client in clients_df.iterrows():
        clients.append({'x': float(client['x']), 
                        'y': float(client['y']),
                        'bandwidth': float(client['bandwidth']),
                        'client_index': index,
                        'assigned': False})
    
    return clients

'''
Função neighborhood change
Compara o fitnesse de duas soluções e faz a troca baseado nisso.
Retorna a nova ou a antiga solução e também a nova ou antiga estrutura de vizinhança
'''
def neighborhoodChange(x, y, k):
    if y.fitness < x.fitness:
        x = cp.deepcopy(y)
        k = 1
    else:
        k += 1
    return x, k

'''
Retorna a posição de um cliente aleatório que não está servido por um PA.
'''
def getRandomUnservedClient(assignments: List[Dict], clients: List[Dict]) -> Tuple[float, float]:
    served_clients = {client['client_index'] for client in assignments}

    if (len(served_clients) == 495):
        return 0
    
    unserved_clients = [(float(client['x']), float(client['y'])) for client in clients if client['client_index'] not in served_clients]
    
    client_index = np.random.randint(0, len(unserved_clients))

    return unserved_clients[client_index]

'''
Retorna a posição do PA, na lista de PAs, que está sendo menos utilizado.
'''
def getLessUsedPA(assignments: List[Dict], clients: List[Dict]) -> int:
    pas = [{'pa_index': obj['pa_index'], 'bd_consumption': clients[obj['client_index']]['bandwidth']} for obj in assignments]
    
    # Usando defaultdict para somar bd_consumption por pa_index
    bd_consumption_by_pa = defaultdict(float)

    for item in pas:
        bd_consumption_by_pa[item['pa_index']] += item['bd_consumption']

    bd_consumption_by_pa = dict(bd_consumption_by_pa)
    sorted_bd_consumption = sorted(bd_consumption_by_pa.items(), key=lambda x: x[1], reverse=False)

    return sorted_bd_consumption[0][0]

'''
Retorna uma coordenada no grid de posibilidades do PA, 5x5, que seja mais próxima relativamente 
a coordenada do cliente informada.
'''
def getPAPositionFromClientPosition(x_client: float, y_client: float) -> Tuple[int, int]:
    return (int(round(x_client/5)*5), int(round(y_client/5)*5))

'''
Função shake
Parâmetros de entrada: Uma solução candidata, uma estrutura de vizinhança e os dados do problema.
Retorno: Uma nova solução candidata
'''
def shake(solution, k, probdata):
    np.random.seed() 
    new_solution = cp.deepcopy(solution) 

    '''
    Estruturas de vizinhança:
    1 - Ativar um PA próximo a um cliente que não está servido por um PA.
    2 - Desativar o PA que está sendo menos usado.
    3 - Mover um PA aleatório em uma direção aleatória em um raio de 5 metros.
    4 - Mover um PA aleatório para um ponto aleatório do mapa.
    '''
    if k == 1 and len(solution.pas) < 30:
        '''
        1- Encontrar um cliente que não está servido por um PA.
        2- Calcular as coordenadas onde o PA será inserido.
        3- Inserir um PA na coordenada encontrada no passo 2.
        4- Rearranjar a atribuição de clientes a PAs.
        '''
        unserved_client = getRandomUnservedClient(solution.assignments, probdata.clients)
        if (unserved_client == 0):
            return new_solution
        
        pa_position = getPAPositionFromClientPosition(unserved_client[0], unserved_client[1])
        new_solution.pas.append(pa_position)
        new_solution.assignments = rearrangeClientsAndPAs(new_solution, probdata.clients)

    elif k == 2 and len(solution.pas) > 1:
        '''
        1- Encontrar o PA que está sendo menos utilizado.
        2- Retirar o PA da lista de PAs.
        3- Rearranjar a solução.
        '''
        pa_index = getLessUsedPA(solution.assignments, probdata.clients)
        new_solution.pas.pop(pa_index)
        new_solution.assignments = rearrangeClientsAndPAs(new_solution, probdata.clients)

    elif k == 3:
        '''
        1- Selecionar um número aleatório de 0 até o tamanho da lista de PAs.
        2- Atualizar a posição do PA selecionado.
        3- Rearranjar a solução.
        '''
        pa_index = np.random.randint(0, len(new_solution.pas))
        pa_x, pa_y = new_solution.pas[pa_index] 

        new_pa_x, new_pa_y = UpdatePAPosition(pa_x, pa_y)

        # Convertendo a tupla para uma lista, fazendo a atribuição e convertendo de volta para uma tupla
        new_pa_position = list(new_solution.pas[pa_index])
        new_pa_position[0] = new_pa_x
        new_pa_position[1] = new_pa_y
        new_solution.pas[pa_index] = tuple(new_pa_position)

        new_solution.assignments = rearrangeClientsAndPAs(new_solution, probdata.clients)

    elif k == 4:
        multiples_of_5 = np.arange(0, 401, 5)
        pa_index = np.random.randint(0, len(new_solution.pas))
        new_pa_x = np.random.choice(multiples_of_5)
        new_pa_y = np.random.choice(multiples_of_5)

        # Convertendo a tupla para uma lista, fazendo a atribuição e convertendo de volta para uma tupla
        new_pa_position = list(new_solution.pas[pa_index])
        new_pa_position[0] = new_pa_x
        new_pa_position[1] = new_pa_y
        new_solution.pas[pa_index] = tuple(new_pa_position)

        new_solution.assignments = rearrangeClientsAndPAs(new_solution, probdata.clients)

    return new_solution

'''
Heurística de busca local.
'''
def bestImprovement(current_solution, kmax, probdata):
    best_solution = cp.deepcopy(current_solution)

    for i in range(1, kmax + 1):
        neighbor_solution = shake(best_solution, i, probdata) 
        if neighbor_solution.fitness < best_solution.fitness:
            best_solution = neighbor_solution
    
    return best_solution

'''
Implementa a meta-heurística BVNS com a soma ponderada para otimização multiobjetivo.
'''
def bvns(fobj, x, probdata, approachinfo, maxeval=1000):
    # Contador do número de soluções candidatas avaliadas
    num_sol_avaliadas = 0

    # Máximo número de soluções candidatas avaliadas
    max_num_sol_avaliadas = maxeval

    # Número de estruturas de vizinhanças definidas
    kmax = 4

    # Avalia solução inicial
    x = fobj(x, approachinfo)
    num_sol_avaliadas += 1

    # Ciclo iterativo do método
    while num_sol_avaliadas < max_num_sol_avaliadas:
        k = 1
        while k <= kmax:        
            y = shake(x, k, probdata)
            y = fobj(y, approachinfo)
            z = bestImprovement(y, 4, probdata) # heurística de busca local
            num_sol_avaliadas += 1
            
            x, k = neighborhoodChange(x, z, k)
    
    return x

'''
Implementa a função objetivo do problema
'''
def obj_functions(x, e):
    f1_fitness = fobj1(x) # Já obtem o valor de fitness para f1 devidamente penalizado
    f2_fitness = fobj2(x) # Já obtem o valor de fitness para f2 devidamente penalizado
    
    # Calcula os valores normalizados de fitness
    f1_fitness_normalizado = normalize(f1_fitness, min_f1, max_f1)
    f2_fitness_normalizado = normalize(f2_fitness, min_f2, max_f2)

    x.f1_fitness = f1_fitness_normalizado
    x.f2_fitness = f2_fitness_normalizado

    x.total_distance = getSumDistanceClientsAndPAs(x.assignments)
    
    penalidade = 0
    if (x.f1_fitness >= e):
        penalidade += (x.f1_fitness - e)**2 * 100

    # Calcula o fitness da otimização multiobjetivo (com os valores normalizados)
    x.fitness = f1_fitness_normalizado + penalidade

    return x

'''
Função para normalizar os valores de fitness
'''
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

'''
Implementa a função soma ponderada
'''
def pw_function(x, approachinfo):
    x = obj_functions(x, approachinfo.episolon)

    return x

def is_dominated(sol_a, sol_b):
    """Retorna True se sol_a for dominada por sol_b, caso contrário False."""
    return all(x >= y for x, y in zip(sol_a, sol_b)) and any(x > y for x, y in zip(sol_a, sol_b))

def find_non_dominated_solutions(solutions):
    """Retorna uma lista de soluções não dominadas."""
    non_dominated = []
    for i, sol_a in enumerate(solutions):
        dominated = False
        for j, sol_b in enumerate(solutions):
            if i != j and is_dominated(sol_a, sol_b):
                dominated = True
                break
        if not dominated:
            non_dominated.append(sol_a)
    return non_dominated


def removeDominatedSolutions (solutionsList):
    for solution in solutionsList:
        for solution2 in solutionsList:
            if(solution2.f1 >= solution.f1 and solution2.f2 > solution.f2):
                solutionsList.remove(solution2)
    return solutionsList

def plot_solutions(all_solutions, non_dominated_solutions, title = "Soluções e Soluções Não Dominadas"):
    # Separando os valores de fitness de f1 e f2 para todas as soluções
    all_f1 = [sol[0] for sol in all_solutions]
    all_f2 = [sol[1] for sol in all_solutions]
    
    # Separando os valores de fitness de f1 e f2 para as soluções não dominadas
    nd_f1 = [sol[0] for sol in non_dominated_solutions]
    nd_f2 = [sol[1] for sol in non_dominated_solutions]
    
    # Criando o gráfico de dispersão
    plt.figure(figsize=(10, 6))
    
    # Plotando todas as soluções
    plt.scatter(all_f1, all_f2, c='blue', label='Todas as soluções', alpha=0.5)
    
    # Destacando as soluções não dominadas
    plt.scatter(nd_f1, nd_f2, c='red', label='Soluções não dominadas', edgecolors='black', s=100)
    
    # Adicionando título e rótulos aos eixos
    plt.title(title)
    plt.xlabel('Fitness f1')
    plt.ylabel('Fitness f2')
    
    # Adicionando legenda
    plt.legend()
    
    # Mostrando o gráfico
    plt.show()
# Faz a leitura dos dados da instância do problema 
probdata = probdef()

# Armazena dados para plot
archive = Struct()
archive.sol = []
archive.fitpen = []

# Armazena dados da estratégia de otimização mono-objetivo
approachinfo = Struct()
min_episolon = 8
max_episolon = 25

N = 10

for i in range(0, 20):
    num_sol_avaliadas = 1
    episolon = random.randint(min_episolon, max_episolon)               

    # Gera solução inicial
    x = sol_inicial(probdata)

    while num_sol_avaliadas < 50:
        num_sol_avaliadas += 1
        approachinfo.episolon = episolon
        x = bvns(pw_function, x, probdata, approachinfo, maxeval=500)
        archive.fitpen.append((x.f1_fitness, x.f2_fitness))
            
        
non_dominated_solutions = find_non_dominated_solutions(archive.fitpen)
plot_solutions(archive.fitpen, non_dominated_solutions, title = "Epsilon restrito")
print("Soluções não dominadas:", non_dominated_solutions)
print("Todas as soluções: ", archive.fitpen)