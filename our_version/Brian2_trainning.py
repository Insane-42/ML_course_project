import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy
import pickle
from struct import unpack
from brian2 import *
import brian2 
from brian2tools import *







def result_saving():
    if not is_test:
        t_saving()
    if not is_test:
        con_saving()
    else:
        np.save(data_path + 'activity/v4_2_resultPopVecs' + str(exam_num), result_moni)
        np.save(data_path + 'activity/v4_2_inputNumbers' + str(exam_num), input_numbers)

def result_plotting():
    if r_moni:
        brian2.figure(fig_num)
        fig_num += 1
        for i, name in enumerate(r_moni):
            brian2.subplot(len(r_moni), 1, 1+i)
            brian2.plot(r_moni[name].t/brian2.second, r_moni[name].rate, '.')
            brian2.title('Rates of population ' + name)
    if sp_moni:
        brian2.figure(fig_num)
        fig_num += 1
        for i, name in enumerate(sp_moni):
            brian2.subplot(len(sp_moni), 1, 1+i)
            brian2.plot(sp_moni[name].t/brian2.ms, sp_moni[name].i, '.')
            brian2.title('Spikes of population ' + name)
    if sp_num:
        brian2.figure(fig_num)
        fig_num += 1
        brian2.plot(sp_moni['Ae'].count[:])
        brian2.title('Spike count of population Ae')
    weight_plotting()
    plt.figure(5)
    subplot(3,1,1)
    brian_plot(connections['XeAe'].w)
    subplot(3,1,2)
    brian_plot(connections['AeAi'].w)
    subplot(3,1,3)
    brian_plot(connections['AiAe'].w)
    plt.figure(6)
    subplot(3,1,1)
    brian_plot(connections['XeAe'].delay)
    subplot(3,1,2)
    brian_plot(connections['AeAi'].delay)
    subplot(3,1,3)
    brian_plot(connections['AiAe'].delay)
    brian2.ioff()
    brian2.show()

def rank_getting(assi, sp_rate):
    rate_sum = [0] * 10
    assi_num = [0] * 10
    for i in range(10):
        assi_num[i] = len(np.where(assi == i)[0])
        if assi_num[i] > 0:
            rate_sum[i] = np.sum(sp_rate[assi == i]) / assi_num[i]
    return np.argsort(rate_sum)[::-1]

def assig_getting(results, in_num):
    assi = np.zeros(n_e)
    in_nums = np.asarray(in_num)
    rate_maxx = [0] * n_e
    for j in range(10):
        assi_num = len(np.where(in_nums == j)[0])
        if assi_num > 0:
            rate = np.sum(results[in_nums == j], axis = 0) / assi_num
        for i in range(n_e):
            if rate[i] > rate_maxx[i]:
                rate_maxx[i] = rate[i]
                assi[i] = j
    return assi
def weight_getting():
    name = 'XeAe'
    w_matrix = np.zeros((n_in, n_e))
    n_e_sq = int(np.sqrt(n_e))
    n_in_sq = int(np.sqrt(n_in))
    col_num = n_e_sq*n_in_sq
    row_num = col_num
    weight_new = np.zeros((col_num, row_num))
    con_matrix = np.zeros((n_in, n_e))
    con_matrix[connections[name].i, connections[name].j] = connections[name].w
    w_matrix = np.copy(con_matrix)

    for i in range(n_e_sq):
        for j in range(n_e_sq):
                weight_new[i*n_in_sq : (i+1)*n_in_sq, j*n_in_sq : (j+1)*n_in_sq] = \
                    w_matrix[:, i + j*n_e_sq].reshape((n_in_sq, n_in_sq))
    return weight_new


def weight_plotting():
    name = 'XeAe'
    weight = weight_getting()
    fig = brian2.figure(fig_num, figsize = (18, 18))
    im2 = brian2.imshow(weight, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    brian2.colorbar(im2)
    brian2.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig

def weight_updating(im, fig):
    weight = weight_getting()
    im.set_array(weight)
    fig.canvas.draw()
    return im

def weight_norming():
    for conn_name in connections:
        if conn_name[1] == 'e' and conn_name[3] == 'e':
            s_len = len(connections[conn_name].source)
            t_len = len(connections[conn_name].target)
            conn = np.zeros((s_len, t_len))
            conn[connections[conn_name].i, connections[conn_name].j] = connections[conn_name].w
            conn_new = np.copy(conn)
            col_sum = np.sum(conn_new, axis = 0)
            col_fac = weight['ee_input']/col_sum
            for j in range(n_e):
                conn_new[:,j] *= col_fac[j]
            connections[conn_name].w = conn_new[connections[conn_name].i, connections[conn_name].j]
def con_saving(suff = ''):
    for conns in save_conns:
        conn = connections[conns]
        list_conn = list(zip(conn.i, conn.j, conn.w))
        np.save(data_path + 'weights/v4_2_' + conns + suff, list_conn)
def t_saving(suff = ''):
    for pops in pop_name:
        np.save(data_path + 'weights/v4_2_theta_' + pops + suff, neu_group[pops + 'e'].theta)
def matrix_loading(file_name):
    loc = len(suff) + 4
    if file_name[-4-loc] == 'X':
        num_s = n_in
    else:
        if file_name[-3-loc]=='e':
            num_s = n_e
        else:
            num_s = n_i
    if file_name[-1-loc]=='e':
        num_t = n_e
    else:
        num_t = n_i
    m_data = np.load(file_name)
    v_arr = np.zeros((num_s, num_t))
    if not m_data.shape == (0,):
        v_arr[np.int32(m_data[:,0]), np.int32(m_data[:,1])] = m_data[:,2]
    return v_arr

def data_loading(file_name, is_train = True):
    if os.path.isfile('%s.pickle' % file_name):
        data = pickle.load(open('%s.pickle' % file_name,'rb'))
    else:
        if is_train:
            images = open('train-images.idx3-ubyte','rb')
            labels = open('train-labels.idx1-ubyte','rb')
        else:
            images = open('t10k-images.idx3-ubyte','rb')
            labels = open('t10k-labels.idx1-ubyte','rb')
        images.read(4)  
        image_num = unpack('>I', images.read(4))[0]
        r = unpack('>I', images.read(4))[0]
        c = unpack('>I', images.read(4))[0]
        labels.read(4)  
        N = unpack('>I', labels.read(4))[0]
        x = np.zeros((N, r, c), dtype=np.uint8)  
        y = np.zeros((N, 1), dtype=np.uint8)  
        for i in range(N):
            x[i] = [[unpack('>B', images.read(1))[0] for uc in range(c)]  for ur in range(r) ]
            y[i] = unpack('>B', labels.read(1))[0]
        data = {'x': x, 'y': y, 'rows': r, 'cols': c}
        pickle.dump(data, open("%s.pickle" % file_name, "wb"))
    return data

def neuron_eqs_init():
    e_neuron_eqs = '''
            dv/dt = ((e_v_rest - v) + (I_E+I_I) / nS) / (tau )  : volt (unless refractory)
            I_E = ge * nS * -v : amp
            I_I = gi * nS * (-100.*mV-v) : amp
            dge/dt = -ge/(1.0*ms) : 1
            dgi/dt = -gi/(2.0*ms) : 1
            tau = 100 * ms : second
            '''
    if is_test:
        e_neuron_eqs += '\n  theta      :volt'
    else:
        e_neuron_eqs += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
    e_neuron_eqs += '\n  dtimer/dt = 0.1  : second'

    i_neuron_eqs = '''
            dv/dt = ((i_v_rest - v) + (I_E+I_I) / nS) / (tau )  : volt (unless refractory)
            I_E = ge * nS *-v : amp
            I_I = gi * nS * (-85.*mV-v): amp
            dge/dt = -ge/(1.0*ms) : 1
            dgi/dt = -gi/(2.0*ms) : 1
            tau = 10 * ms : second
            '''
    return e_neuron_eqs, i_neuron_eqs

def get_stdp_ee():
    eqs_stdp_ee = '''
                    post2before : 1
                    dpre/dt   =   -pre/(tc_pre_ee) : 1 (event-driven)
                    dpost1/dt  = -post1/(tc_post_1_ee) : 1 (event-driven)
                    dpost2/dt  = -post2/(tc_post_2_ee) : 1 (event-driven)
                '''
    eqs_stdp_pre_ee = 'pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)'
    eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'
    return eqs_stdp_ee,eqs_stdp_pre_ee,eqs_stdp_post_ee

training = data_loading('training')
testing = data_loading('testing', is_train = False)

def neu_group_init():
    neu_group['e'] = brian2.NeuronGroup(n_e*len(pop_name), e_neuron_eqs, threshold= e_v_thresh_str, refractory= e_refrac, reset= scr_e, method='euler')
    neu_group['i'] = brian2.NeuronGroup(n_i*len(pop_name), i_neuron_eqs, threshold= i_v_thresh_str, refractory= i_refrac, reset= i_v_reset_str, method='euler')


def network_creating():
    for group_num, name in enumerate(pop_name):
        print( 'create neuron group', name)

        neu_group[name+'e'] = neu_group['e'][group_num*n_e:(group_num+1)*n_e]
        neu_group[name+'i'] = neu_group['i'][group_num*n_i:(group_num+1)*n_e]

        neu_group[name+'e'].v = e_v_rest - 40. * brian2.mV
        neu_group[name+'i'].v = i_v_rest - 40. * brian2.mV
        if is_test or weight_path[-8:] == 'weights/':
            neu_group['e'].theta = np.load(weight_path + 'theta_' + name + suff + '.npy') * brian2.volt
        else:
            neu_group['e'].theta = np.ones((n_e)) * 20.0*brian2.mV
        for conn_type in rec_conn_name:
            connName = name+conn_type[0]+name+conn_type[1]
            we_matrix = matrix_loading(weight_path + '../random/' + connName + suff + '.npy')
            model = 'w : 1'
            pre = 'g%s_post += w' % conn_type[0]
            post = ''
            if is_STDP:
                if 'ee' in rec_conn_name:
                    model += eqs_stdp_ee
                    pre += '; ' + eqs_stdp_pre_ee
                    post = eqs_stdp_post_ee
            connections[connName] = brian2.Synapses(neu_group[connName[0:2]], neu_group[connName[2:4]],
                                                        model=model, on_pre=pre, on_post=post)
            connections[connName].connect(True) 
            connections[connName].w = we_matrix[connections[connName].i, connections[connName].j]

        print( 'create monitors ', name)
        r_moni[name+'e'] = brian2.PopulationRateMonitor(neu_group[name+'e'])
        r_moni[name+'i'] = brian2.PopulationRateMonitor(neu_group[name+'i'])
        sp_num[name+'e'] = brian2.SpikeMonitor(neu_group[name+'e'])

def connection_creating():
    for i,name in enumerate(input_name):
        in_group[name+'e'] = brian2.PoissonGroup(n_in, 0*Hz)
        r_moni[name+'e'] = brian2.PopulationRateMonitor(in_group[name+'e'])

    for name in input_conn_name:
        print( 'create connections ', name[0], 'and', name[1])
        for conn_type in input_conn_names:
            connName = name[0] + conn_type[0] + name[1] + conn_type[1]
            we_matrix = matrix_loading(weight_path + connName + suff + '.npy')
            model = 'w : 1'
            pre = 'g%s_post += w' % conn_type[0]
            post = ''
            if is_STDP:
                model += eqs_stdp_ee
                pre += '; ' + eqs_stdp_pre_ee
                post = eqs_stdp_post_ee
            connections[connName] = brian2.Synapses(in_group[connName[0:2]], neu_group[connName[2:4]],
                                                        model=model, on_pre=pre, on_post=post)
            delay_minn = delay[conn_type][0]
            delay_maxx = delay[conn_type][1]
            delay_delta = delay_maxx - delay_minn
            connections[connName].connect(True)
            connections[connName].delay = 'delay_minn + rand() * delay_delta'
            connections[connName].w = we_matrix[connections[connName].i, connections[connName].j]

def trainning():

    network = Network()
    for obj_list in [neu_group, in_group, connections, r_moni,
            sp_moni, sp_num]:
        for key in obj_list:
            network.add(obj_list[key])

    pre_sp_num = np.zeros(n_e)
    assi = np.zeros(n_e)
    input_num = [0] * exam_num
    output_num = np.zeros((exam_num, 10))
    if not is_test:
        input_we_moni, fig_weights = weight_plotting()
        fig_num += 1
    for i,name in enumerate(input_name):
        in_group[name+'e'].rates = 0 * Hz
    network.run(0*second)
    j = 0
    while j < (int(exam_num)):
        if is_test:
            if testing_set_using:
                spike_rates = testing['x'][j%10000,:,:].reshape((n_in)) / input_dev *  input_int
            else:
                spike_rates = training['x'][j%60000,:,:].reshape((n_in)) / input_dev *  input_int
        else:
            weight_norming()
            spike_rates = training['x'][j%60000,:,:].reshape((n_in)) / input_dev *  input_int
        in_group['Xe'].rates = spike_rates * Hz
        network.run(exp_t, report='text')

        if j % interv == 0 and j > 0:
            assi = assig_getting(result_moni[:], input_num[j-interv : j])
        if j % weight_update_interval == 0 and not is_test:
            weight_updating(input_we_moni, fig_weights)
        if j % conn_interv == 0 and j > 0 and not is_test:
            con_saving(str(j))
            t_saving(str(j))

        now_sp_num = np.asarray(sp_num['Ae'].count[:]) - pre_sp_num
        pre_sp_num = np.copy(sp_num['Ae'].count[:])
        if np.sum(now_sp_num) < 5:
            input_int += 1
            for i,name in enumerate(input_name):
                in_group[name+'e'].rates = 0 * Hz
            network.run(rest_t)
        else:
            result_moni[j%interv,:] = now_sp_num
            if is_test and testing_set_using:
                input_num[j] = testing['y'][j%10000][0]
            else:
                input_num[j] = training['y'][j%60000][0]
            output_num[j,:] = rank_getting(assi, result_moni[j%interv,:])
            if j % 100 == 0 and j > 0:
                print( 'runs done:', j, 'of', int(exam_num))
            for i,name in enumerate(input_name):
                in_group[name+'e'].rates = 0 * Hz
            network.run(rest_t)
            input_int = input_int_st
            j += 1



is_test = True

np.random.seed(0)
data_path = './'
if is_test:
    weight_path = data_path + 'weights/'
    exam_num = 1000 * 1
    testing_set_using = True
    is_STDP = False
    interv = exam_num
else:
    weight_path = data_path + 'random/'
    exam_num = 6000 * 3
    testing_set_using = False
    is_STDP = True


suff = ''
n_in = 784
n_e = 400
n_i = n_e
exp_t =   0.35 * brian2.second 
rest_t = 0.15 * brian2.second
runtime = exam_num * (exp_t + rest_t)
if exam_num <= 10000:
    interv = exam_num
    weight_update_interval = 20
else:
    interv = 1000
    weight_update_interval = 100
if exam_num <= 60000:
    conn_interv = 1000
else:
    conn_interv = 1000
    interv = 1000

e_v_rest = -65. * brian2.mV
i_v_rest = -60. * brian2.mV
e_v_reset = -65. * brian2.mV
i_v_reset = -45. * brian2.mV
e_v_thresh = -52. * brian2.mV
i_v_thresh = -40. * brian2.mV
e_refrac = 5. * brian2.ms
i_refrac = 2. * brian2.ms

weight = {}
delay = {}
input_name = ['X']
pop_name = ['A']
input_conn_name = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input']
rec_conn_name = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*brian2.ms,10*brian2.ms)
delay['ei_input'] = (0*brian2.ms,5*brian2.ms)
input_int = 2.
input_dev = 8.
input_int_st = input_int

tc_pre_ee = 20*brian2.ms
tc_post_1_ee = 20*brian2.ms
tc_post_2_ee = 40*brian2.ms
nu_ee_pre =  0.0001      
nu_ee_post = 0.01       
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4


if is_test:
    scr_e = 'v = e_v_reset; timer = 0*ms'
else:
    tc_theta = 1e7 * brian2.ms
    theta_plus_e = 0.05 * brian2.mV
    scr_e = 'v = e_v_reset; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*brian2.mV
e_v_thresh_str = '(v>(theta - offset + e_v_reset)) and (timer>e_refrac)'
i_v_thresh_str = 'v>i_v_thresh'
i_v_reset_str = 'v=i_v_reset'

e_neuron_eqs, i_neuron_eqs = neuron_eqs_init()

eqs_stdp_ee,eqs_stdp_pre_ee,eqs_stdp_post_ee = get_stdp_ee()


brian2.ion()
fig_num = 1
neu_group = {}
in_group = {}
connections = {}
r_moni = {}
sp_moni = {}
sp_num = {}
result_moni = np.zeros((interv,n_e))
neu_group_init()


network_creating()

connection_creating()

trainning()

result_saving()

result_plotting()






