import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

solver = caffe.SGDSolver("dummy_blstm_solver.prototxt")

data = solver.net.blobs['sum'].data
shape = data.shape

t = range(shape[0])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.set_ylabel('label prediction probabilities')
ax2.set_ylabel('gradients')
ax1.set_xlabel('Time')
ax2.set_xlabel('Time')
data_p = [ax1.plot(t, np.sin(t))[0] for _ in range(shape[2])]
diff_p = [ax2.plot(t, np.cos(t))[0] for _ in range(shape[2])]

def init():
    for i in range(shape[2]):
        data_p[i].set_ydata(np.ma.array(np.zeros(shape[0]), mask=True))
        diff_p[i].set_ydata(np.ma.array(np.zeros(shape[0]), mask=True))

    return data_p + diff_p

def update_plot(data, diff):
    for i in range(shape[2]):
        data_p[i].set_ydata(data[:,0,i])
        diff_p[i].set_ydata(diff[:,0,i])

    return data_p + diff_p


def animate(i):
    solver.step(100)
    data = solver.net.blobs['sum'].data
    diff = solver.net.blobs['sum'].diff
    ax1.relim()
    ax1.autoscale_view(True, True, True)
    ax2.relim()
    ax2.autoscale_view(True, True, True)
    return update_plot(data, diff)


ani = animation.FuncAnimation(fig, animate, np.arange(1, 100), init_func=init,
        interval=1000, blit=True)
plt.show()



