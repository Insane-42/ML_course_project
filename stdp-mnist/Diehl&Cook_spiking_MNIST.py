# coding=utf-8
'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

 
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cPickle as pickle
import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging 
import brian as b
from struct import unpack
from brian import *

# specify the location of the MNIST data
MNIST_data_path = ''

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]   # >I表示大端读字节，读unsigned int
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
    
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in xrange(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)]  for unused_row in xrange(rows) ]  # >B表示大端读字节，读unsigned char
            y[i] = unpack('>B', labels.read(1))[0]
            
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input                
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print readout.shape, fileName
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


def save_connections(ending = ''):
    print 'save connections'
    for connName in save_conns:
        connMatrix = connections[connName][:]
#         connListSparse = ([(i,j[0],j[1]) for i in xrange(connMatrix.shape[0]) for j in zip(connMatrix.rowj[i],connMatrix.rowdata[i])])
        connListSparse = ([(i,j,connMatrix[i,j]) for i in xrange(connMatrix.shape[0]) for j in xrange(connMatrix.shape[1]) ])
        np.save(data_path + 'weights/' + connName + ending, connListSparse)

def save_theta(ending = ''):
    print 'save theta'
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)

def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            connection = connections[connName][:]
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight['ee_input']/colSums
            for j in xrange(n_e):#
                connection[:,j] *= colFactors[j]


def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))         # 20
    n_in_sqrt = int(np.sqrt(n_input))    # 28
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = connections[name][:]    # connections['XeAe'] 见 L432-439
    weight_matrix = np.copy(connMatrix)

    # 分成多个28x28的小块，使用connMatrix为其赋值
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig
    
def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def plot_performance(fig_num):
    num_evaluations = int(num_examples/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = b.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    b.ylim(ymax = 100)
    b.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance
    
def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in xrange(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments
    
    
#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print 'time needed to load training set:', end - start
 
start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print 'time needed to load test set:', end - start


#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = True

b.set_global_preferences( 
                        defaultclock = b.Clock(dt=0.5*b.ms), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave = True, # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options = ['-ffast-math -march=native'],  # Defines the compiler switches passed to the gcc compiler. 
                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimizations are turned on 
                        usecodegen = True,  # Whether or not to use experimental code generation support.
                        usecodegenweave = True,  # Whether or not to use C with experimental code generation support.
                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
                        usecodegenthreshold = False,  # Whether or not to use experimental code generation support on thresholds.
                        usenewpropagate = True,  # Whether or not to use experimental new C propagation functions.
                        usecstdp = True,  # Whether or not to use experimental new C STDP.
                       ) 


np.random.seed(0)
data_path = './'
if test_mode:  # 如果是测试模式，就从weights/中加载训练好的权重，测试sample数为10000个
    weight_path = data_path + 'weights/'
    num_examples = 10000 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False
    update_interval = num_examples
else:          # 如果是训练模式，就从random/中加载随机初始化的权重，训练sample数为60000*3个(但文章中写的是60000)
    weight_path = data_path + 'random/'  
    num_examples = 60000 * 3
    use_testing_set = False
    do_plot_performance = True
    if num_examples <= 60000:    
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True


ending = ''
n_input = 784  # 每张图片像素为28x28
n_e = 400  # 激发神经元数量
n_i = n_e  # 抑制神经元数量
single_example_time = 0.35 * b.second  # 跑一个example的时间
resting_time = 0.15 * b.second  # 重置电位所需时间
runtime = num_examples * (single_example_time + resting_time)
if num_examples <= 10000:    
    update_interval = num_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
if num_examples <= 60000:    
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000

v_rest_e = -65. * b.mV  # 激发神经元的静息电位
v_rest_i = -60. * b.mV  # 抑制神经元的静息电位
v_reset_e = -65. * b.mV  # 激发神经元的重置电位
v_reset_i = -45. * b.mV  # 抑制神经元的重置电位
v_thresh_e = -52. * b.mV  # 激发神经元的激发阈值电位
v_thresh_i = -40. * b.mV  # 抑制神经元的激发阈值电位
refrac_e = 5. * b.ms  # 激发神经元不应期
refrac_i = 2. * b.ms  # 抑制神经元不应期

conn_structure = 'dense'
weight = {}
delay = {}
input_population_names = ['X']
population_names = ['A']
input_connection_names = ['XA']
save_conns = ['XeAe']
input_conn_names = ['ee_input'] 
recurrent_conn_names = ['ei', 'ie']
weight['ee_input'] = 78.
delay['ee_input'] = (0*b.ms, 10*b.ms)
delay['ei_input'] = (0*b.ms, 5*b.ms)
input_intensity = 2.
start_input_intensity = input_intensity

tc_pre_ee = 20*b.ms
tc_post_1_ee = 20*b.ms
tc_post_2_ee = 40*b.ms
nu_ee_pre = 0.0001      # learning rate
nu_ee_post = 0.01       # learning rate
wmax_ee = 1.0           # 应该是公式(3)(4)(6)中的w_max
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4

# 只有在训练的时候才适应性地改变v_threshold，测试的时候不改变theta，scr_e每次进行膜电位的重置，并提高theta
if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * b.ms
    theta_plus_e = 0.05 * b.mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*b.mV
v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

# 激发神经元的充放电公式，对应于文中的公式（1）和（2），不知道为啥激活神经元平衡电势是0，它就直接写了个-v[解答：Brian2中所有微分方程的初始条件默认为(0, 0)]
neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS *         -v                           : amp   
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

# 测试时theta不变，训练时theta在神经元受激活时增加，否则指数衰减
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
neuron_eqs_e += '\n  dtimer/dt = 100.0  : ms'

# 抑制神经元的充放电公式，对应于文中的公式（1）和（2），每行后面为单位，1表示无量纲
neuron_eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

# 三个微分方程表示突触前和突触后的trace随时间指数衰减，post1表示上一个突触后脉冲，post2表示下一个突触后脉冲，这里的"上一个"和"下一个"是相对于
# 目前的突触前脉冲而言的，所以假设发现了一个突触前脉冲，post1会减弱权值连接，post2会增强权值连接【有待细究】
eqs_stdp_ee = '''
                post2before                            : 1.0
                dpre/dt   =   -pre/(tc_pre_ee)         : 1.0
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1.0
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1.0
            '''
# stdp算法对权值的更新公式,突触后神经元发放脉冲时，需要计算对应的突触前脉冲在STDP学习窗口中获得的学习量，进而增大突触连接权值；当突触前神经元发放脉冲时，
# 突触连接权值会减小，这是它与上一个突触后脉冲相互作用的结果
eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post1'
eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post * pre * post2before; post1 = 1.; post2 = 1.'
    
b.ion()
fig_num = 1
neuron_groups = {}
input_groups = {}
connections = {}
stdp_methods = {}
rate_monitors = {}
spike_monitors = {}
spike_counters = {}
result_monitor = np.zeros((update_interval,n_e))

# 这里的population_names是'A'，应该代表的是神经元，'Ae'代表激活神经元，'Ai'代表抑制神经元
neuron_groups['e'] = b.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, 
                                   compile = True, freeze = True)
neuron_groups['i'] = b.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i, 
                                   compile = True, freeze = True)


#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------ 
for name in population_names:
    print 'create neuron group', name
    
    neuron_groups[name+'e'] = neuron_groups['e'].subgroup(n_e)  # neuron_groups['Ae']
    neuron_groups[name+'i'] = neuron_groups['i'].subgroup(n_i)  # neuron_groups['Ai']
    
    # 设置神经元的初始电位，以及load训练好的theta，不太明白为啥要减40mV
    neuron_groups[name+'e'].v = v_rest_e - 40. * b.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b.mV
    if test_mode or weight_path[-8:] == 'weights/':
        neuron_groups['e'].theta = np.load(weight_path + 'theta_' + name + ending + '.npy')
    else:
        neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b.mV

    # recurrent_conn_names = ['ei', 'ie']，分别代表的是e和i之间的连接、i和e之间的连接
    print 'create recurrent connections'
    for conn_type in recurrent_conn_names:
        connName = name+conn_type[0]+name+conn_type[1]  # 对应于'ei'和'ie'，connName分别就成了 'AeAi'和'AiAe'
        weightMatrix = get_matrix_from_file(weight_path + '../random/' + connName + ending + '.npy')
        # 以weightMatrix为权重矩阵，把'Ae'和'Ai'连接起来，conn_structure='dense'应该代表的是全连接，但是e、i之间的连接应该是不对称的？
        connections[connName] = b.Connection(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure, 
                                                    state = 'g'+conn_type[0])
        connections[connName].connect(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix)

    # 训练则ee_STDP_on=True, 测试则ee_STDP_on=False，但是recurrent_conn_names = ['ei', 'ie']，看起来进不了里面的if
    if ee_STDP_on:
        if 'ee' in recurrent_conn_names:
            stdp_methods[name+'e'+name+'e'] = b.STDP(connections[name+'e'+name+'e'], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, 
                                                           post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)

    # 监测一个runtime内的Ae、Ai的rate，以及Ae的脉冲计数
    print 'create monitors for', name
    rate_monitors[name+'e'] = b.PopulationRateMonitor(neuron_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)
    rate_monitors[name+'i'] = b.PopulationRateMonitor(neuron_groups[name+'i'], bin = (single_example_time+resting_time)/b.second)
    spike_counters[name+'e'] = b.SpikeCounter(neuron_groups[name+'e'])
    
    if record_spikes:
        spike_monitors[name+'e'] = b.SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = b.SpikeMonitor(neuron_groups[name+'i'])

if record_spikes:
    b.figure(fig_num)
    fig_num += 1
    b.ion()
    b.subplot(211)
    b.raster_plot(spike_monitors['Ae'], refresh=1000*b.ms, showlast=1000*b.ms)
    b.subplot(212)
    b.raster_plot(spike_monitors['Ai'], refresh=1000*b.ms, showlast=1000*b.ms)


#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------ 
pop_values = [0,0,0]
# input_population_names = ['X']，n_input = 784, 初始化输入层Xe
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = b.PoissonGroup(n_input, 0)
    rate_monitors[name+'e'] = b.PopulationRateMonitor(input_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)

# input_connection_names = ['XA']， input_conn_names = ['ee_input'], 构建input层和e层之间的连接(即Xe和Ae之间的连接)
for name in input_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]  # 也就是connName = XeAe
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        connections[connName] = b.Connection(input_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure, 
                                                    state = 'g'+connType[0], delay=True, max_delay=delay[connType][1])
        connections[connName].connect(input_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix, delay=delay[connType])
     
    if ee_STDP_on:
        print 'create STDP for connection', name[0]+'e'+name[1]+'e'
        stdp_methods[name[0]+'e'+name[1]+'e'] = b.STDP(connections[name[0]+'e'+name[1]+'e'], eqs=eqs_stdp_ee, pre = eqs_stdp_pre_ee, 
                                                       post = eqs_stdp_post_ee, wmin=0., wmax= wmax_ee)


#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))
if not test_mode:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1
if do_plot_performance:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
for i,name in enumerate(input_population_names):
    input_groups[name+'e'].rate = 0
b.run(0)
j = 0
while j < (int(num_examples)):
    # 获得输入的intensity（即rates）,输入到input_groups['Xe']中获得柏松分布的spike
    if test_mode:
        if use_testing_set:  # 这里逻辑很奇怪，根据前面的语句，test_mode = use_testing_set
            rates = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity   # 初始input_intensity=2，相当于除以4
        else:
            rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        normalize_weights()
        rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    input_groups['Xe'].rate = rates
#     print 'run number:', j+1, 'of', int(num_examples)
    b.run(single_example_time, report='text')

    # 间隔一定的sample数进行weight的更新和保存
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    if j % weight_update_interval == 0 and not test_mode:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        save_theta(str(j))
    
    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    if np.sum(current_spike_count) < 5:   # 如果激发的神经元过少，那么就增大input_intensity，以激发更多的e层神经元
        input_intensity += 1
        for i,name in enumerate(input_population_names): # input_population_names=['X']
            input_groups[name+'e'].rate = 0
        b.run(resting_time)
    else:
        result_monitor[j%update_interval,:] = current_spike_count
        if test_mode and use_testing_set:  # 得到该sample对应的label
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:]) # 得到预测结果
        if j % 100 == 0 and j > 0:
            print 'runs done:', j, 'of', int(num_examples)
        if j % update_interval == 0 and j > 0:
            if do_plot_performance:
                unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
                print 'Classification performance', performance[:(j/float(update_interval))+1]
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rate = 0
        b.run(resting_time)
        input_intensity = start_input_intensity
        j += 1


#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'
if not test_mode:
    save_theta()
if not test_mode:
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)
    

#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
if rate_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(rate_monitors):
        b.subplot(len(rate_monitors), 1, i)
        b.plot(rate_monitors[name].times/b.second, rate_monitors[name].rate, '.')
        b.title('Rates of population ' + name)
    
if spike_monitors:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        b.subplot(len(spike_monitors), 1, i)
        b.raster_plot(spike_monitors[name])
        b.title('Spikes of population ' + name)
        
if spike_counters:
    b.figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_counters):
        b.subplot(len(spike_counters), 1, i)
        b.plot(spike_counters['Ae'].count[:])
        b.title('Spike count of population ' + name)

plot_2d_input_weights()
b.ioff()
b.show()



