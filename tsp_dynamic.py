import random
import matplotlib.pyplot as plt
import networkx as nx
import math as mt
from datetime import datetime


def distance(x1, y1, x2, y2):
    return mt.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


INF = 2 ** 31 - 1

# random.seed(1)

n = 11

v1 = []
points = {}
for i in range(n):
    points[i] = (random.randint(1, 1000), random.randint(1, 1000))

input_matrix = []
for i, vi in points.items():
    m1 = []
    for j, vj in points.items():
        if i == j:
            m1.append(INF)
        else:
            m1.append(int(distance(vi[0], vi[1], vj[0], vj[1])))
            v1.append([i, j, int(distance(vi[0], vi[1], vj[0], vj[1]))])
    input_matrix.append(m1.copy())

plt.figure(figsize=(8, 8))

graph = nx.Graph()
graph.add_nodes_from(points)

for i in v1:
    graph.add_edge(i[0], i[1], weight=i[2])

def tsp(input_matrix):
    n = len(input_matrix)   
    s = (1 << (n - 1)) - 1
    path = [0] * s
    local_sum = [0] * s

    for i in range(s):
        path[i] = [0] * (n - 1)
        local_sum[i] = [-1] * (n - 1)
    m = [n - 1, input_matrix.copy(), path, local_sum]

    sum_path = INF
    for i in range(m[0]):
        index = 1 << i
        if s & index != 0:
            sum_temp = tsp_next(m, s ^ index, i) + m[1][i + 1][0]
            if sum_temp < sum_path:
                sum_path = sum_temp
                m[2][0][0] = i + 1
    m[3][0][0] = sum_path

    res = []
    init_point = int(path[0][0])
    res.append(init_point)
    s = ((1 << m[0]) - 1) ^ (1 << init_point - 1)
    for i in range(1, m[0]):
        init_point = int(path[s][init_point - 1])
        res.append(init_point)
        s = s ^ (1 << init_point - 1)
    res.append(0)
    return [sum_path, res]

def tsp_next(m, s, init_point):
    if m[3][s][init_point] != -1:
        return m[3][s][init_point]
    if s == 0:
        return m[1][0][init_point + 1]
    sum_path = INF
    for i in range(m[0]):
        index = 1 << i
        if s & index != 0:
            sum_temp = tsp_next(m, s ^ index, i) + m[1][i + 1][init_point + 1]
            if sum_temp < sum_path:
                sum_path = sum_temp
                m[2][s][init_point] = i + 1
    m[3][s][init_point] = sum_path
    return sum_path

start_time = datetime.now()
res = tsp(input_matrix)
print(datetime.now() - start_time)
print(res)

d = []
s = res[1]
for i, v in enumerate(s):
    d.append([int(s[i - 1]), int(s[i])])

plt.figure(figsize=(9, 6), edgecolor='black', linewidth=1)
plt.axis("equal")
plt.title(f'Size: {n}; Length: {sum}', fontsize=10)
nx.draw(graph, points, width=1, edge_color="#C0C0C0", with_labels=True)
nx.draw(graph, points, width=2, edge_color="red", edgelist=d, style="dashed")
plt.show()
