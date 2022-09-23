import random
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tabulate
from io import BytesIO
import os

# random.seed(1)  # for same results

BINS=32         # number of columns on charts 
DEG=6           #from 2 to 6, for N
N=10**DEG 

if not os.path.exists('./files'):
        os.mkdir('./files')


def R() -> float:
    R = 0
    while R == 0:
        R = random.random()
    return R

# Равномерное распределение
def even_dist(a: int, b: int, n: int) -> list:
    dist = [0] * n
    for i in range(n):
        dist[i] = a + R()*(b-a)
    return dist

# Экспотенциальное распределение
def exp_dist(l: int, n: int) -> list:
    dist = [0] * n
    for i in range(n):
        dist[i] = (-1) * math.log(R())/ l
    return dist 

# Нормальное распределение
def norm_dist(m: int, s:int, n: int) -> list:
    dist = [0] * n
    for i in range(n):
        R12 = 0
        for j in range(12):
            R12 += R()
        dist[i] = m + s * (R12 - 6)
    return dist 

# Мат ожидание и дисперсия
def M(dist: list) -> float:
    return np.mean(dist)

def D(dist: list) -> float:
    return np.var(dist, ddof=1)


if __name__ == '__main__':
    
    # Равномерное распределение
    a=1
    b=6
    d1 = even_dist(a, b, N)
    evenDist_hist1 = seaborn.displot(d1, bins=BINS)
    evenDist_hist2 = seaborn.displot(d1, kind='kde')
    evenDist_hist1.savefig("files/evenDist_hist1.png")
    evenDist_hist2.savefig("files/evenDist_hist2.png")
    # print(d1)

    # Экспотенциальное распределение
    l=1
    d2 = exp_dist(l, N)
    expDist_hist1 = seaborn.displot(d2, bins=BINS)
    expDist_hist2 = seaborn.displot(d2, kind='kde')
    expDist_hist1.savefig('files/expDist_hist1.png')
    expDist_hist2.savefig('files/expDist_hist2.png')

    # Нормальное распределение
    m=0
    s=3
    d3 = norm_dist(m, s, N)
    normDist_hist1 = seaborn.displot(d3, bins=BINS)
    normDist_hist2 = seaborn.displot(d3, kind='kde')
    normDist_hist1.savefig('files/normDist_hist1.png')
    normDist_hist2.savefig('files/normDist_hist2.png')

    # Генерируем все размеры выборок
    n = 10
    NS = [n]

    while (n < N):
        n *= 2
        NS.append(int(n))
        n *= 2.5
        NS.append(int(n))
        n*=2
        NS.append(int(n))

    dists = [d1, d2, d3]
    est = [[None for i in range(len(NS))] for j in range(len(dists) * 2)]

    for i in range(len(dists)):
        for j in range(len(NS)):
            est[2*i][j] = M(random.choices(dists[i], k=NS[j])) 
            est[2*i+1][j] = D(random.choices(dists[i], k=NS[j]))
            # random.choices(list, k) - выберет из list k элементов случайным образом (могут повторяться)

    true_est = [(a+b)/2, ((b-a)**2)/12, 1/l, 1/(l*l), m, s]

    # Вывод таблицы
    table = []
    dist_names = ['even', 'exp', 'norm']
    h = ["N(i)"] + NS + [True]

    for i in range(len(dists)):
        table.append([f'M({dist_names[i]}_dist)'] + est[2*i] + [true_est[2*i]])
        table.append([f'D({dist_names[i]}_dist)'] + est[2*i+1] + [true_est[2*i+1]])

    print(tabulate.tabulate(table, headers=h, tablefmt="fancy_grid", floatfmt=".3f"))

    # Подготовка данных для вывода графика зависимостей оценок от объема выборки

    nplist = []
    hh = [f'N{i+1}' for i in range(len(h)-1)]
    hh[-1] = 'N' # подписи для оси Х

    # загоняем все данные в датафрейм в ненормализованном виде
    for i in range(len(table)):
        for j in range(1, len(table[i])):
            nplist.append([table[i][0], hh[j-1], table[i][j]])
    # print(tabulate.tabulate(nplist))
    # tabulate(nplist)

    data = pd.DataFrame(np.array(nplist), columns=['est-dist', 'sample', 'value'])
    data = data.astype({'value': np.double }) # нужно дать нормальный тип столбцу, который будет вертикальной осью

    # # Все зависимости на одном графике
    # allInOnePlot = seaborn.lineplot(data, x='sample', y='value', hue='est-dist')
    # #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) # для вывода легенды справа
    # allInOnePlot.figure.figsize = (800, 500)
    # allInOnePlot.figure.savefig('files/allInOnePlot.png')

    # Каждая оценка на отдельном графике
    g = seaborn.FacetGrid(data, col='est-dist', col_wrap=2, height=5.5)
    g.map(seaborn.lineplot, 'sample', 'value')
    g.savefig('files/allPlots.png')

