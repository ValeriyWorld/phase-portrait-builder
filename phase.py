import math as m

import numpy as np
import matplotlib.pyplot as plt

# from test_initial_points import num_points, num_iter, x, y


# p = lambda x, y : -(x + y)**2 + 1

def p(x, y):
   return -(x + y)**2 + 1

# q = lambda x, y : -1 + x**2 + y

def q(x, y):
    return -1 + x**2 + y

def normalize(x, y):
    return x / m.hypot(x, y)

def length(x, y, k, i):
    return (
            m.hypot((x[k,i] - x[k,i-1]), (y[k,i] - y[k,i-1])) -
            m.hypot((x[k,i-1] - x[k,i-2]), (y[k,i-1] - y[k,i-2]))
           )

def length_increase(x, y, k, i):
    if length(x, y, k, i) > 0:
       return True

def length_decrease(x, y, k, i):
    if length(x, y, k, i) < 0:
       return True

def phase_plot_builder(step_type, num_points, x, y, num_iter,
                       alpha_x=0.001, alpha_y=0.001):
    coef1 = 10**(1)
    const1, const2 = alpha_x * coef1, alpha_y * coef1
    for k in range(num_points):
        for i in range(1, num_iter):
            x[k,i] = x[k,i-1] + alpha_x * normalize(p(x[k,i-1], y[k,i-1]),
                                                    q(x[k,i-1], y[k,i-1]))
            y[k,i] = y[k,i-1] + alpha_y * normalize(q(x[k,i-1], y[k,i-1]),
                                                    p(x[k,i-1], y[k,i-1]))
            #plt.plot([x[k,i-1], x[k,i]], [y[k,i-1], y[k,i]], 'g')
            if i > 1:
                if step_type == 'const':
                    pass
                elif step_type == 'linear':
                    if length_increase(x, y, k, i):
                       alpha_x += const1
                       alpha_y += const2
                    elif length_decrease(x, y, k, i):
                       alpha_x -= const1
                       alpha_y -= const2
                elif step_type == 'exp':
                    alpha_x *= np.exp(length(x, y, k, i))
                    alpha_y *= np.exp(length(x, y, k, i))
        plt.plot(x[k], y[k], 'g')
    plt.axis([-4, 4, -4, 4])
    plt.show()


step_type = input('Enter the step changing type: ')

num_points = int(input('Enter the number of initial points: '))
num_iter = int(input('Enter the number of iterations: '))

x = np.zeros((num_points, num_iter))
y = np.zeros((num_points, num_iter))

print('Enter the initial points: ')
for i in range(num_points):
    x[i,0] = float(input('x[{},0] = '.format(i)))
    y[i,0] = float(input('y[{},0] = '.format(i)))


phase_plot_builder(step_type, num_points, x, y, num_iter)
