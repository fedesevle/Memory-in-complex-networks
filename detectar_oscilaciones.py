import matplotlib.pyplot as plt
import numpy as np
import random
import scipy as sc
from scipy import interpolate
import seaborn as sns
sns.set()
import pickle
import clasemotif
#%%
def lhs(n_variables,n_muestras,random_seed=0):
    random.seed(random_seed)
    np.random.seed(random_seed)
    p=1/n_muestras
    x = np.zeros([n_muestras,n_variables])
    for i_muestra in range(n_muestras):
        for i_variable in range(n_variables):
            x[i_muestra,i_variable]=(random.random()+i_muestra)*p
    for i_variable in range(n_variables):
        np.random.shuffle(x[:,i_variable])
    return x

def armar_lista_parametros(n_variables = 32,n_muestras = 10000):
    # tengo que generar los k y K de:
    # el Input (2)
    # cada interaccion dirigida entre A-B-C y consigo mismas (9 + 9)
    # cada activacion de background (prendida si nadie activa un nodo) (6)
    # cada inhibicion de background (prendida si nadie inhibe un nodo) (6)
    # param = [kI,ks,kE,kF,KI,Ks,KE,KF]

    parametros_LHS =  lhs(n_variables,n_muestras,random_seed=0)
    lista_parametros = []
    for i_muestra in range(n_muestras):
        ks = 10**((parametros_LHS[i_muestra,:16]-0.5)*2)
        Ks = 10**((parametros_LHS[i_muestra,16:]-0.5)*4)
        set_params = list(ks) + list(Ks)
    
        kI = set_params[0] # Estimulo
        ks = np.array(set_params[1:10]).reshape(3,3) # Interacciones
        kE = np.array(set_params[10:13]) # activacion background
        kF = np.array(set_params[13:16]) # inhibicion background
        # K
        KI = set_params[16] # Estimulo
        Ks = np.array(set_params[17:26]).reshape(3,3) # Interacciones
        KE = np.array(set_params[26:29]) # activacion background
        KF = np.array(set_params[29:]) # inhibicion background
        # A recibe el input entonces no hace falta la activacion background
        kE[0] = 0 
        lista_parametros.append([kI,np.transpose(ks),kE,kF,KI,np.transpose(Ks),KE,KF])
    return lista_parametros

def armar_lista_parametros_repressilator(n_variables = 18,n_muestras = 100000):
    # tengo que generar los k y K de:
    # el Input (2)
    # cada interaccion dirigida entre A-B-C y consigo mismas (9 + 9)
    # cada activacion de background (prendida si nadie activa un nodo) (6)
    # cada inhibicion de background (prendida si nadie inhibe un nodo) (6)
    # param = [kI,ks,kE,kF,KI,Ks,KE,KF]

    parametros_LHS =  lhs(n_variables,n_muestras,random_seed=0)
    lista_parametros = []
    for i_muestra in range(n_muestras):
        ks = 10**((parametros_LHS[i_muestra,:9]-0.5)*2)
        Ks = 10**((parametros_LHS[i_muestra,9:]-0.5)*4)
        set_params = list(ks) + list(Ks)
    
        kI = 1000 # Estimulo
        ks = np.array(set_params[0:9]).reshape(3,3) # Interacciones
        kE = np.array([100,100,100]) # activacion background
        kF = np.array([0,0,0]) # inhibicion background
        # K
        KI = 100 # Estimulo
        Ks = np.array(set_params[9:]).reshape(3,3) # Interacciones
        KE = np.array([100,100,100]) # activacion background
        KF = np.array([0.1,0.1,0.1]) # inhibicion background
        # A recibe el input entonces no hace falta la activacion background
        kE[0] = 0 
        lista_parametros.append([kI,np.transpose(ks),kE,kF,KI,np.transpose(Ks),KE,KF])
    return lista_parametros

def armar_lista_parametros_ppn(n_variables = 18,n_muestras = 100000):
    # tengo que generar los k y K de:
    # el Input (2)
    # cada interaccion dirigida entre A-B-C y consigo mismas (9 + 9)
    # cada activacion de background (prendida si nadie activa un nodo) (6)
    # cada inhibicion de background (prendida si nadie inhibe un nodo) (6)
    # param = [kI,ks,kE,kF,KI,Ks,KE,KF]

    parametros_LHS =  lhs(n_variables,n_muestras,random_seed=0)
    lista_parametros = []
    for i_muestra in range(n_muestras):
        ks = 10**((parametros_LHS[i_muestra,:9]-0.5)*2)
        Ks = 10**((parametros_LHS[i_muestra,9:]-0.5)*4)
        set_params = list(ks) + list(Ks)
    
        kI = 10 # Estimulo
        ks = np.array(set_params[0:9]).reshape(3,3) # Interacciones
        kE = np.array([0,0,0]) # activacion background
        kF = np.array([0,1,1]) # inhibicion background
        # K
        KI = .5 # Estimulo
        Ks = np.array(set_params[9:]).reshape(3,3) # Interacciones
        KE = np.array([.5,0.1,0.1]) # activacion background
        KF = np.array([0.1,.5,.5]) # inhibicion background
        # A recibe el input entonces no hace falta la activacion background
        kE[0] = 0 
        lista_parametros.append([kI,np.transpose(ks),kE,kF,KI,np.transpose(Ks),KE,KF])
    return lista_parametros

def crear_redes_I_A_B_C_O():
    
    # A, B y C pueden activarse o inhibirse entre si y a si mismos
    interacciones = [1,0,-1]
    # Armo todas las conmbinaciones, existen 19683 (3**9)
    redes = []
    for aa in interacciones:
        for ab in interacciones:
            for ac in interacciones:
                for ba in interacciones:
                    for bb in interacciones:
                        for bc in interacciones:
                            for ca in interacciones:
                                for cb in interacciones:
                                    for cc in interacciones:
                                        redes.append(np.transpose(np.array([aa,ab,ac,ba,bb,bc,ca,cb,cc]).reshape(3,3)))
    # Pensando que el input esta en A y el output en C
    # me quedo solo con las que conectan input-output (A con C), son 16038                                  
    redes_I_O = []
    for ired in range(len(redes)):
        red = redes[ired]
        if red[0,2]!=0 or (red[0,1]!=0 and red[1,2]!=0):
            redes_I_O.append(red)
    return redes_I_O

def calcular_memoria(I,Red,modelo,condiciones_iniciales, tiempo_max):
    
    I_int = I[0]*Red.param[0]

    _,variables_1,_ = integrar(modelo,Red,I_int,condiciones_iniciales, tiempo_max)
    tiempo_1,variables_1,transitorio = integrar(modelo,Red,I_int,variables_1[:,-1], tiempo_max)
    I_int = I[1]*Red.param[0]
    _,variables_2,_ = integrar(modelo,Red,I_int,variables_1[:,-1], tiempo_max)
    I_int = I[0]*Red.param[0]
    tiempo_2,variables_2,transitorio_2 = integrar(modelo,Red,I_int,variables_2[:,-1], tiempo_max)
    if not transitorio:
        tiempo_1 = np.append(tiempo_1,tiempo_2[-1])
        variables_1_extendido = np.zeros([len(condiciones_iniciales),len(variables_1[0,:])+1])
        variables_1_extendido[:,:-1] = variables_1
        variables_1_extendido[:,-1] = variables_1[:,-1]
        variables_1 = variables_1_extendido
    
    tiempo = np.linspace(0,np.min([tiempo_1[-1],tiempo_2[-1]]))
    var_1_interp = interpolate.interp1d(tiempo_1,variables_1, kind = 'linear')
    var_2_interp = interpolate.interp1d(tiempo_2,variables_2, kind = 'linear')
    variables_1 = var_1_interp(tiempo)
    variables_2 = var_2_interp(tiempo)
    memoria = np.linalg.norm(variables_2 - variables_1,axis=0)
    return tiempo,variables_1,variables_2, memoria*0.5773502691896258,transitorio_2

def calcular_memoria_integral(I,Red,modelo,condiciones_iniciales, tiempo_max):
    
    I_int = I[0]*Red.param[0]

    _,variables_1,_ = integrar(modelo,Red,I_int,condiciones_iniciales, tiempo_max)
    tiempo_1,variables_1,transitorio = integrar(modelo,Red,I_int,variables_1[:,-1], tiempo_max)
    I_int = I[1]*Red.param[0]
    _,variables_2,_ = integrar(modelo,Red,I_int,variables_1[:,-1], tiempo_max)
    I_int = I[0]*Red.param[0]
    tiempo_2,variables_2,transitorio_2 = integrar(modelo,Red,I_int,variables_2[:,-1], tiempo_max)
    if not transitorio:
        tiempo_1 = np.append(tiempo_1,tiempo_2[-1])
        variables_1_extendido = np.zeros([len(condiciones_iniciales),len(variables_1[0,:])+1])
        variables_1_extendido[:,:-1] = variables_1
        variables_1_extendido[:,-1] = variables_1[:,-1]
        variables_1 = variables_1_extendido
    
    tiempo_corto = np.argmin([tiempo_1[-1],tiempo_2[-1]]) #miro el tiempo mas corto
    largo_tiempo_corto = [len(tiempo_1),len(tiempo_2)][tiempo_corto] #defino el largo del tiempo mas corto
    
    tiempo = np.linspace(0,np.min([tiempo_1[-1],tiempo_2[-1]]),largo_tiempo_corto)
    var_1_interp = interpolate.interp1d(tiempo_1,variables_1, kind = 'linear')
    var_2_interp = interpolate.interp1d(tiempo_2,variables_2, kind = 'linear')
    variables_1 = var_1_interp(tiempo)
    variables_2 = var_2_interp(tiempo)
    memoria = np.linalg.norm(variables_2 - variables_1,axis=0)
    return tiempo,variables_1,variables_2, memoria*0.5773502691896258,transitorio_2

def modelo(Red,I, var, tiempo):
    in_var = 1-var
    # Escribo las derivadas a partir de las interacciones de la red
    derivadas = np.dot(var,Red.param_act*in_var/(in_var+Red.param[5])-Red.param_inh*var/(var+Red.param[5]))
    # Agrego el input en A
    derivadas[0] += I*in_var[0]/(in_var[0]+Red.param[4])
    # Agrego las activaciones/inhibiciones si no hay
    derivadas += Red.enzima_act*in_var/(in_var+Red.param[6])
    derivadas -= Red.enzima_inh*var/(var+Red.param[7])
    return derivadas

def integrar(modelo, Red,I, condiciones_iniciales, tiempo_max, tiempo_inicial = 0,
             error_min = 10**-6, error_max = 10**-4, max_iter=10000):
    
    cant_variables = len(condiciones_iniciales)
    
    dt_max = Red.escala_temporal
    dt_min = 0.001 * Red.escala_temporal

    variables = np.zeros([cant_variables,max_iter])
    variables[:,0] = condiciones_iniciales
    
    tiempo = np.ones(max_iter)*tiempo_inicial
    
    i = 0
    dt = dt_max
    transitorio = True
    while i<max_iter-1 and tiempo[i]<tiempo_max+1 and transitorio:
        if dt < dt_min:
            dt = dt_min
        elif dt > dt_max:
            dt = dt_max
        dk4,dk5 = rk4(modelo, Red,I, variables[:,i], tiempo, dt)
        error = np.sum(np.abs(dk4-dk5))
        if error > error_max and dt > dt_min:
            dt = 0.5*dt
        else:
            variables[:,i+1] = variables[:,i] + dk5
            tiempo[i+1] = tiempo[i] + dt
            i+=1
            if error < error_min:
                dt = 2 * dt
        if np.max(np.abs(variables[:,i]-variables[:,i-1]))<0.00001 and i>1:
            transitorio = False
    return tiempo[:i],variables[:,:i],transitorio

def integrar_completo(modelo, Red,I, condiciones_iniciales, tiempo_max, tiempo_inicial = 0,
             error_min = 10**-6, error_max = 10**-4, max_iter=100000):
    
    cant_variables = len(condiciones_iniciales)
    
    dt_max = 0.1
    dt_min = 0.001

    variables = np.zeros([cant_variables,max_iter])
    variables[:,0] = condiciones_iniciales
    
    tiempo = np.ones(max_iter)*tiempo_inicial
    
    i = 0
    t = tiempo_inicial
    dt = dt_max
    transitorio = True
    while i<max_iter-1 and t<tiempo_max+1 and transitorio:
        if dt < dt_min:
            dt = dt_min
        elif dt > dt_max:
            dt = dt_max
        dk4,dk5 = rk4(modelo, Red,I, variables[:,i], tiempo, dt)
        error = np.sum(np.abs(dk4-dk5))
        if error > error_max and dt > dt_min:
            dt = 0.5*dt
        else:
            variables[:,i+1] = variables[:,i] + dk5
            tiempo[i+1] = tiempo[i] + dt
            i+=1
            t+=dt
            if error < error_min:
                dt = 2 * dt
    return tiempo[:i],variables[:,:i]

def rk4(modelo, Red,I, variables, tiempo, dt):
    k_1 = dt * modelo(Red,I,variables, tiempo)
    k_2 = dt * modelo(Red,I,variables + 0.5 * k_1, tiempo)
    k_3 = dt * modelo(Red,I,variables + 0.75 * k_2, tiempo)
    d_rk3 = 0.2222222222222222 * k_1 + 0.3333333333333333 * k_2 + 0.4444444444444444 * k_3
    k_4 = dt * modelo(Red,I,variables + d_rk3 , tiempo)
    d_rk4 = 0.2916666666666667 * k_1 + 0.25 * k_2 + 0.3333333333333333 * k_3 + 0.125 * k_4
    return d_rk3,d_rk4

def graficar_dinamica(I,Red,modelo,condiciones_iniciales, tiempo_max):
    tiempo,variables_1,variables_2,memoria,_ = calcular_memoria(I,Red,modelo,condiciones_iniciales, tiempo_max)

    plt.figure()
    plt.subplot(131)
    plt.plot(tiempo, variables_1[0], label = 'A', lw = 5)
    plt.plot(tiempo, variables_1[1], label = 'B', lw = 5)
    plt.plot(tiempo, variables_1[2], label = 'C', lw = 5)
    plt.axis([min(tiempo),max(tiempo),0,1])
    plt.legend()
    plt.subplot(132)
    plt.plot(tiempo, variables_2[0], label = 'A', lw = 5)
    plt.plot(tiempo, variables_2[1], label = 'B', lw = 5)
    plt.plot(tiempo, variables_2[2], label = 'C', lw = 5)
    plt.axis([min(tiempo),max(tiempo),0,1])
    plt.legend()
    plt.subplot(133)
    plt.plot(tiempo, memoria, lw = 5, color = (0.2,0.2,0.2))
    plt.axis([min(tiempo),max(tiempo),0,1])

def graficar_dinamica_comun(I,Red,modelo,condiciones_iniciales, tiempo_max):
    tiempo_1,variables_1 = integrar_completo(modelo, Red,I[0]*Red.param[0], condiciones_iniciales, tiempo_max)
    condiciones_iniciales = variables_1[:,-1]
    tiempo_2,variables_2 = integrar_completo(modelo, Red,I[1]*Red.param[0], condiciones_iniciales, tiempo_max)
    condiciones_iniciales = variables_2[:,-1]
    tiempo_3,variables_3 = integrar_completo(modelo, Red,I[0]*Red.param[0], condiciones_iniciales, tiempo_max)
    tiempo = np.concatenate((tiempo_1,tiempo_1[-1]+tiempo_2,tiempo_1[-1]+tiempo_2[-1]+tiempo_3))
    variables = np.concatenate((variables_1,variables_2,variables_3),axis=1)
    estimulo = np.concatenate((I[0]*np.ones(len(tiempo_1)),I[1]*np.ones(len(tiempo_2)),I[0]*np.ones(len(tiempo_3))))
    plt.figure()
    plt.subplot(211)
    plt.plot(tiempo, variables[0], label = 'A', lw = 5)
    plt.plot(tiempo, variables[1], label = 'B', lw = 5)
    plt.plot(tiempo, variables[2], label = 'C', lw = 5)
    plt.legend()
    plt.axis([min(tiempo),max(tiempo),-0.1,1.1])
    plt.subplot(212)
    plt.plot(tiempo, estimulo, label = 'Estimulo', color = (0.3,0.3,0.3), lw = 5)
    plt.axis([min(tiempo),max(tiempo),0,10])
    plt.xlabel('Tiempo')
    plt.legend()

def graficar_dinamica_memoria(I,Red,modelo,condiciones_iniciales, tiempo_max):
    tiempo_1,variables_1 = integrar_completo(modelo, Red,I[0]*Red.param[0], condiciones_iniciales, tiempo_max)
    condiciones_iniciales = variables_1[:,-1]
    tiempo_2,variables_2 = integrar_completo(modelo, Red,I[1]*Red.param[0], condiciones_iniciales, tiempo_max)
    condiciones_iniciales = variables_2[:,-1]
    tiempo_3,variables_3 = integrar_completo(modelo, Red,I[0]*Red.param[0], condiciones_iniciales, tiempo_max)
    tiempo = np.concatenate((tiempo_1,tiempo_1[-1]+tiempo_2,tiempo_1[-1]+tiempo_2[-1]+tiempo_3))
    variables = np.concatenate((variables_1,variables_2,variables_3),axis=1)
    estimulo = np.concatenate((I[0]*np.ones(len(tiempo_1)),I[1]*np.ones(len(tiempo_2)),I[0]*np.ones(len(tiempo_3))))
    condiciones_iniciales = [0,0,0]
    tiempo_1,variables_1 = integrar_completo(modelo, Red,I[0]*Red.param[0], condiciones_iniciales, tiempo[-1])

    tiempo_memo = np.linspace(0,np.min([tiempo_1[-1],tiempo[-1]]))
    var_1_interp = interpolate.interp1d(tiempo_1,variables_1, kind = 'linear')
    var_2_interp = interpolate.interp1d(tiempo,variables, kind = 'linear')
    variables_1_memo = var_1_interp(tiempo_memo)
    variables_2_memo = var_2_interp(tiempo_memo)
    memoria = np.linalg.norm(variables_2_memo - variables_1_memo,axis=0)

    plt.figure()
    plt.subplot(221)
    plt.plot(tiempo_1, variables_1[0], label = 'A', lw = 5)
    plt.plot(tiempo_1, variables_1[1], label = 'B', lw = 5)
    plt.plot(tiempo_1, variables_1[2], label = 'C', lw = 5)
    plt.legend()
    plt.axis([min(tiempo),max(tiempo),-0.1,1.1])
    plt.subplot(223)
    plt.plot(tiempo, variables[0], label = 'A', lw = 5)
    plt.plot(tiempo, variables[1], label = 'B', lw = 5)
    plt.plot(tiempo, variables[2], label = 'C', lw = 5)
    plt.xlabel('Tiempo')
    plt.legend()
    plt.axis([min(tiempo),max(tiempo),-0.1,1.1])
    plt.subplot(222)
    plt.plot(tiempo, estimulo, label = 'Estimulo', color = (0.3,0.3,0.3), lw = 5)
    plt.axis([min(tiempo),max(tiempo),0,10])
    plt.legend()
    plt.subplot(224)
    plt.plot(tiempo_memo, memoria, label = 'Memoria', color = (0.4,0.2,0.4), lw = 5)
    plt.axis([min(tiempo),max(tiempo),0,1])
    plt.xlabel('Tiempo')
    plt.legend()

def buscar_id_Red(matriz,matrices_redes):
    for i_matrices in range(len(matrices_redes)):
        if np.sum(matrices_redes[i_matrices]==matriz)==9:
            break
    return i_matrices

def buscar_set_con_memoria(id_Red,lista_parametros,memorias):
    redes = []
    for i_memo in range(len(memorias)):
        Red = clasemotif.Motif(matrices_redes[id_Red], lista_parametros[i_memo])
        if memorias[i_memo] > 0.7 and np.log10(Red.param[1][2,0]/Red.param[5][2,0])>1 and np.log10(Red.param[1][0,0]/Red.param[5][0,0])>1:
            redes.append(i_memo)
    return redes
 
def integrar_red(matriz_red, lista_parametros, condiciones_iniciales = [0,0,0], tiempo_max = 10000, I = [.1,5]):
    n_sets=len(lista_parametros)
    memorias = np.zeros(n_sets)
    transitorios = np.zeros(n_sets)
    for i_param in range(n_sets):
        Red = clasemotif.Motif(matriz_red, lista_parametros[i_param])
        _,_,_, memoria, transitorio = calcular_memoria(I,Red,modelo,condiciones_iniciales, tiempo_max)
        memorias[i_param] = memoria[-1]
        transitorios[i_param] = transitorio
    return memorias, transitorios     

def crear_diccionario_matriz_id_red(matrices_redes):
    matrices_a_id_redes = {}
    for id_Red in range(len(matrices_redes)):
        matrices_a_id_redes.update({str(matrices_redes[id_Red]):id_Red})
    return matrices_a_id_redes      
#%% Creo la lista de parametros con Latin Hypercube Sampling
lista_parametros = armar_lista_parametros()
#%% Creo la lista de parametros con Latin Hypercube Sampling
lista_parametros_respres = armar_lista_parametros_repressilator()
#%% Creo la lista de parametros con Latin Hypercube Sampling
lista_parametros_ppn = armar_lista_parametros_ppn()
#%% Creo la lista de las redes con todas las combinaciones posibles
matrices_redes = crear_redes_I_A_B_C_O()
#%% Creo diccionario con key = matriz, value = id_Red, para buscarlas facil
matrices_a_id_redes = crear_diccionario_matriz_id_red(matrices_redes)
#%% Definir una red con una matriz
matriz = np.array([[0,-1,0],[0,0,-1],[-1,0,0]])
id_Red = matrices_a_id_redes[str(matriz)]
#%% Setear un set particular de parametros
parametros = [100, # k input
 np.array([[0. , 20, 0. ],
        [0., 0., 20],
        [20, 0. , 0.]]), # k interacciones
 np.array([100 , 100, 100]), # k enzimas act
 np.array([0., 0., 0.]), # k enzimas inh
 100, # K input
 np.array([[0.1, .01, 0.1], 
        [0.1, 0.1, .01],
        [0.01, 0.1, 0.1]]), # K interacciones
 np.array([100, 100, 100]), # K enzimas act
 np.array([0.1, 0.1, 0.1])] # K enzimas inh
#%% Redes interesantes
id_Red = 1234 # DFN DFP
id_Red = 7828 # Red lineal
# id_Red = 3076 # NFBL
# id_Red = 2482 # Red lineal + A se autoprende
id_Red = 8800 #Feedback positivo indirecto de a 3
id_Red = 3301
id_Red = 8803 # Repressilator
id_Red = 8422 # Positivo Positivo Negativo

#%% Graficar un set de una red y su dinamica
I = .1;condiciones_iniciales=[0,0,0];tiempo_max=100
i_set=623
i_set=597
i_set=220
# i_set=set_oscilan[358]
i_set=70843
Red = clasemotif.Motif(matrices_redes[id_Red], lista_parametros_respres[i_set])
tiempo,variables = integrar_completo(modelo, Red,I*Red.param[0], condiciones_iniciales, tiempo_max)
plt.figure()
plt.plot(tiempo, variables[0], label = 'A', lw = 5)
plt.plot(tiempo, variables[1], label = 'B', lw = 5)
plt.plot(tiempo, variables[2], label = 'C', lw = 5)

#%% Graficar un set de una red y su dinamica
tiempo_equi = np.linspace(0,tiempo[-1],len(tiempo)*10)
variables_interp = interpolate.interp1d(tiempo, variables, kind='cubic')
variables_interp = variables_interp(tiempo_equi)
plt.figure()
plt.plot(tiempo_equi,variables_interp[0])
dt = tiempo_equi[1]-tiempo_equi[0]


largo_tiempo = len(tiempo_equi)
frecuencias = np.linspace(0.0, 1.0/(2.0*dt), largo_tiempo//2)

transformada = sc.fft.fft(variables_interp[0])
transformada = 2.0/largo_tiempo * np.abs(transformada[0:largo_tiempo//2])
plt.figure()
plt.plot(frecuencias, transformada)

picos = sc.signal.find_peaks(transformada,threshold=0.0001)
plt.plot(frecuencias[picos[0]], transformada[picos[0]],'o')
#%% Cargar 
nombre_archivo = 'sets_oscilan'
with open(nombre_archivo+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
    oscilan = pickle.load(f)

#%% 

# Veo si una Red con su set definido oscila
def es_oscilador(Red):
    I = .1;condiciones_iniciales=[0,0,0];tiempo_max=100
    tiempo,variables = integrar_completo(modelo, Red,I*Red.param[0], condiciones_iniciales, tiempo_max)
    tiempo_equi = np.linspace(0,tiempo[-1],len(tiempo)*10)
    variables_interp = interpolate.interp1d(tiempo, variables, kind='cubic')
    variables_interp = variables_interp(tiempo_equi)
    largo_tiempo = len(tiempo_equi)
    transformada = sc.fft.fft(variables_interp[0])
    transformada = 2.0/largo_tiempo * np.abs(transformada[0:largo_tiempo//2])
    picos = sc.signal.find_peaks(transformada,threshold=0.001)
    return len(picos[0])>0

# Calculo la integral de la memoria para los últimos puntos para una lista de sets
def calcular_memoria_sets(lista_sets,id_Red):
    I = [.1,5];condiciones_iniciales=[0,0,0];tiempo_max=100
    memoria_set = np.zeros(len(lista_sets))
    i=0
    for i_set in lista_sets:
        Red = clasemotif.Motif(matrices_redes[id_Red], lista_parametros_ppn[i_set])
        tiempo,variables_1,variables_2, memoria,transitorio = calcular_memoria_integral(I,Red,modelo,condiciones_iniciales, tiempo_max)   
        memoria_integral = np.sum(memoria[int(len(memoria)*9/10):])
        memoria_set[i] = memoria_integral
        i+=1
    return memoria_set

# Me fijo cuáles de los 10000 sets oscilan        
def detectar_sets_que_oscilan(id_Red,lista_parametros):
    N_sets = len(lista_parametros)
    oscilan = np.zeros(N_sets) 
    for i_set in range(N_sets):
        Red = clasemotif.Motif(matrices_redes[id_Red], lista_parametros[i_set])
        oscilan[i_set] = es_oscilador(Red)
        if i_set%10==0:
            print(i_set)
    return sc.signal.find_peaks(oscilan,threshold=0.001)[0]

# Armo lista de los que NO oscilan
def armar_lista_sets_que_NO_oscilan(N_sets,set_oscilan):
    lista_sets = list(range(N_sets))
    for i_set in range(len(set_oscilan)-1,-1,-1):
        lista_sets.pop(set_oscilan[i_set])
    return lista_sets

def definir_estacionario(id_Red,set_oscilan):
    set_oscilan_estacionario = np.zeros(len(set_oscilan))
    i = 0
    for i_set in set_oscilan:
        Red = clasemotif.Motif(matrices_redes[id_Red], lista_parametros_ppn[i_set])
        tiempo,variables = integrar_completo(modelo, Red,I*Red.param[0], condiciones_iniciales, tiempo_max)
        if np.abs(variables[0][-1]-variables[0][len(variables[0])-2])<0.0001:
            set_oscilan_estacionario[i] = 1
        print(i)
        i+=1
    return set_oscilan_estacionario
                                          

# lista_parametros_ppn = lista_parametros_ppn[:1000]
N_sets = len(lista_parametros_ppn)
set_oscilan = detectar_sets_que_oscilan(id_Red,lista_parametros_ppn)
set_NO_oscilan = armar_lista_sets_que_NO_oscilan(N_sets,set_oscilan)

set_oscilan_estacionario = definir_estacionario(id_Red,set_oscilan)

N_rep = 100
sets_por_grupo = len(set_oscilan)
lista_memorias_sets_random = []
for i_rep in range(N_rep):
    lista_sets_random = random.sample(set_NO_oscilan,sets_por_grupo)
    lista_memorias_sets_random.append(calcular_memoria_sets(lista_sets_random,id_Red))
    print(i_rep)

lista_memorias_sets_oscilan=[]
lista_memorias_sets_oscilan.append(calcular_memoria_sets(set_oscilan,id_Red))

plt.figure()
hist = plt.hist(lista_memorias_sets_oscilan[0][set_oscilan_estacionario==0],20)
plt.hist(lista_memorias_sets_oscilan[0][set_oscilan_estacionario==1],hist[1])
histo = plt.hist(lista_memorias_sets_random)
plt.yscale('log');#plt.xscale('log')
plt.axis([-3,70,0.2,1000])
plt.axis([0,0.09,0.2,1000])





#%% Guardar
nombre_archivo = 'oscilan_repress'
with open(nombre_archivo+'.pkl', 'wb') as f:
    pickle.dump([set_oscilan,set_NO_oscilan,lista_memorias_sets_oscilan,lista_memorias_sets_random], f)
#%% Guardar sets que oscilan
nombre_archivo = 'sets_oscilan_ppn'
with open(nombre_archivo+'.pkl', 'wb') as f:
    pickle.dump([set_oscilan,set_NO_oscilan,lista_memorias_sets_oscilan,lista_memorias_sets_random], f)
#%% Graficar un set de una red y su dinamica INPUT Escalón
Red = clasemotif.Motif(matrices_redes[id_Red], parametros)
I = [.1,1];condiciones_iniciales=[0,0,0];tiempo_max=5
graficar_dinamica_comun(I,Red,modelo,condiciones_iniciales, tiempo_max)
Red.graficar()
