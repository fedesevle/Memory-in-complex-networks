import lhsmdu
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import clasemotif
from scipy import interpolate
import time
import seaborn as sns
sns.set()
import pickle

#%%
def buscar_id_Red(matriz,matrices_redes):
    for i_matrices in range(len(matrices_redes)):
        if np.sum(matrices_redes[i_matrices]==matriz)==9:
            break
    return i_matrices

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

def graficar_histograma_memorias(id_Red,lista_memorias,lista_transitorios):
    memorias = lista_memorias[id_Red]
    transitorios = lista_transitorios[id_Red]
    plt.figure()
    plt.hist([memorias[transitorios==False],memorias[transitorios==True]],50,stacked=True)
    plt.yscale('log')
    plt.xlabel('Memoria')
    plt.ylabel('Frecuencia')
    plt.axis([0,1,0.5,len(memorias)])
    
def crear_diccionario_matriz_id_red(matrices_redes):
    matrices_a_id_redes = {}
    for id_Red in range(len(matrices_redes)):
        matrices_a_id_redes.update({str(matrices_redes[id_Red]):id_Red})
    return matrices_a_id_redes

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
#%% Cargar variables 
nombre_archivo = '16038redes_100sets'
with open(nombre_archivo+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
    lista_memorias, lista_transitorios, id_Redes = pickle.load(f)
    
#%% Creo la lista de parametros con Latin Hypercube Sampling
lista_parametros = armar_lista_parametros()
#%% Creo la lista de las redes con todas las combinaciones posibles
matrices_redes = crear_redes_I_A_B_C_O()
#%% Creo diccionario con key = matriz, value = id_Red, para buscarlas facil
matrices_a_id_redes = crear_diccionario_matriz_id_red(matrices_redes)
#%% Buscar redes con memoria significativa
n_sets = 100; n_redes = 16038
id_Redes_memoria = np.zeros(n_redes)
for i_red in range(len(lista_memorias)):
    memorias = lista_memorias[i_red]
    transitorios = lista_transitorios[i_red]
    if sum(memorias[transitorios==False]>0.5)>0.02*n_sets:
        id_Redes_memoria[i_red] = 1
#%%
id_Red = 7234
Red = clasemotif.Motif(matrices_redes[id_Red], lista_parametros[0])
Red.graficar(grosor_parametros = False)

def deletar_matrices(id_Red,matrices_redes):
    id_redes_deletadas = []
    matriz = matrices_redes[id_Red].copy()
    for i in range(3):
        for j in range(3):
            if matriz[(i,j)] != 0:
                interaccion = matriz[(i,j)]
                matriz[(i,j)] = 0
                try:
                    id_redes_deletadas.append(matrices_a_id_redes[str(matriz)])
                except KeyError:
                    pass
                matriz[(i,j)] = interaccion
    return np.array(id_redes_deletadas)

def buscar_redes_deletadas_con_memoria(id_Red, id_Redes_memoria):
    if id_Redes_memoria[id_Red] == 1:
        # Armo todas las combinaciones de deleciones unicas
        id_redes_deletadas = deletar_matrices(id_Red,matrices_redes)
        # Me quedo con las que tienen memoria
        id_redes_deletadas = id_redes_deletadas[id_Redes_memoria[id_redes_deletadas]==1]
    return id_redes_deletadas

redes_con_memoria =  np.array(range(16038))
redes_con_memoria =  redes_con_memoria[id_Redes_memoria[redes_con_memoria]==1]
motifs_memoria = []
t0=time.time()
for id_Red in redes_con_memoria:
    if len(buscar_redes_deletadas_con_memoria(id_Red, id_Redes_memoria)) == 0:
        motifs_memoria.append(id_Red)
tardo = time.time()-t0

resultados = [3301,8800,9787,9814,11816]

for id_Red in motifs_memoria:
    Red = clasemotif.Motif(matrices_redes[id_Red], lista_parametros[0])
    Red.graficar(grosor_parametros = False)
       
id_redes_deletadas = buscar_redes_deletadas_con_memoria(id_Red, id_Redes_memoria)

for i in matrices_deletadas:
    Red = clasemotif.Motif(matrices_redes[i], lista_parametros[0])
    Red.graficar()

matrices_nuevas = np.array(matrices_nuevas)
matrices_con_memoria = np.zeros(len(matrices_nuevas))
for i_matriz_nueva in range(len(matrices_nuevas)):
    matrices_con_memoria = id_Redes_memoria[buscar_id_Red(matrices_nuevas[i_matriz_nueva],matrices_redes)]
        
    


#%% Graficar redes encontradas
for i_red in range(len(id_Redes)):
    if id_Redes_memoria[i_red] == 1:
        Red = clasemotif.Motif(matrices_redes[id_Redes[i_red]], lista_parametros[0])
        Red.graficar(grosor_parametros=False)