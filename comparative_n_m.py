import matplotlib.pyplot as plt
import numpy as np
import random
import imp
import circuit_class
import scipy as sc
from scipy import signal
import seaborn as sns
sns.set()
import pickle
import pandas as pd
#%%
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

def buscar_redes_deletadas_con_memoria(id_Red, conjunto_id_redes):
    # Armo todas las combinaciones de deleciones unicas
    id_redes_deletadas = deletar_matrices(id_Red,matrices_redes)
    # Me quedo con las que tienen memoria
    id_redes_deletadas = conjunto_id_redes & set(id_redes_deletadas)
    return id_redes_deletadas

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

def armar_lista_memoria_vectores(directorio,rango):
    memoria_vector = []
    for id_Red in rango:
        nombre_archivo = 'Corrida_prueba_%d' %id_Red
        with open(directorio + nombre_archivo + '.pkl','rb') as f:  # Python 3: open(..., 'rb')
            output = pickle.load(f)
            memoria_vector.append(output[1]-output[0])
    return memoria_vector

def graficar_distribucion_cubo(id_Red):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = memoria_vector[id_Red][0]
    y = memoria_vector[id_Red][1]
    z = memoria_vector[id_Red][2]
    ax.scatter(x,y,z,alpha=0.1)
    ax.set_xlabel('A');ax.set_ylabel('B');ax.set_zlabel('C')
    ax.plot([-1,1],[-1,1],[-1,1],'.',alpha=0)
    ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([-1, 0, 1])
    ax.view_init(elev=16, azim=-38)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    plt.show()

def mas_memoria(memoria_vector):
    suma_memoria=np.zeros(len(memoria_vector))
    for id_Red in range(len(memoria_vector)):
        suma_memoria[id_Red]=sum(np.abs(memoria_vector[id_Red][np.abs(memoria_vector[id_Red])<1]))
    # redes_mas_memoria = np.zeros(len(memoria_vector))
    # suma_memoria_int = list(suma_memoria)
    # for id_Red in range(len(memoria_vector)):
    #     redes_mas_memoria[id_Red] = np.argmax(suma_memoria_int)
    #     print(redes_mas_memoria[id_Red])
    #     suma_memoria_int.pop(int(redes_mas_memoria[id_Red]))
    return suma_memoria/10000

def maxima_memoria(memoria_vector):
    max_memoria = np.zeros(len(memoria_vector))
    set_max_memoria = np.zeros(len(memoria_vector))
    for id_Red in range(len(memoria_vector)):
        max_memoria[id_Red]=np.max(np.abs(memoria_vector[id_Red][np.abs(memoria_vector[id_Red])<1]))
        set_max_memoria[id_Red]=np.argmax(np.abs(memoria_vector[id_Red][np.abs(memoria_vector[id_Red])<1]))%10000
    return max_memoria,set_max_memoria

# Acá era más estricto porque tomaba más cajitas
def detectar_picos_memoria(memoria_vector):
    posicion_picos = np.zeros(len(memoria_vector))
    for id_Red in range(len(memoria_vector)):
        histo = np.histogram(np.abs(memoria_vector[id_Red][np.abs(memoria_vector[id_Red])<1]),100)
        histo0 = np.concatenate((histo[0],np.array([0])))
        pico = sc.signal.find_peaks(histo0,threshold=20)[0]
        if len(pico)>0:
            posicion_picos[id_Red] = np.max(pico)
    return posicion_picos

def detectar_picos_memoria(memoria_vector):
    posicion_picos = np.zeros(len(memoria_vector))
    for id_Red in range(len(memoria_vector)):
        histo = np.histogram(np.abs(memoria_vector[id_Red][np.abs(memoria_vector[id_Red])<1]),10)
        histo0 = np.concatenate((histo[0],np.array([0])))
        pico = sc.signal.find_peaks(histo0,threshold=20)[0]
        if len(pico)>0:
            posicion_picos[id_Red] = np.max(pico)
    return posicion_picos

def contar_sets_memoria_C(memoria_vector):
    sets_C = np.zeros(len(memoria_vector))
    for id_Red in range(len(memoria_vector)):
        sets_C[id_Red] = np.sum((np.abs(memoria_vector[id_Red][2,:])<1) * (np.abs(memoria_vector[id_Red][2,:])>0.8))
    return sets_C    

def graficar_histo_picos(id_Red,memoria_vector):
    histo = plt.hist(np.abs(memoria_vector[id_Red][np.abs(memoria_vector[id_Red])<1]),100)
    histo0 = np.concatenate((histo[0],np.array([0])))
    pico = sc.signal.find_peaks(histo0,threshold=20)[0]
    plt.plot(0.5*(histo[1][pico]+histo[1][pico+1]),histo[0][pico],'o')
    plt.yscale('log')

def buscar_motifs_principales(conjunto_id_redes):
    motifs = []
    for id_Red in conjunto_id_redes:
        if len(buscar_redes_deletadas_con_memoria(id_Red, conjunto_id_redes)) == 0:
            motifs.append(id_Red)
    return motifs

def graficar_filtro(A,B,C):
    ms=40;
    plt.figure(figsize=(15,4))
    plt.subplot(131)
    if A:
        ms_int=ms*0.2
    else:
        ms_int=ms
    plt.plot(1, 1,'o', label = 'A', ms=ms_int, color = (42/255, 157/255, 143/255))    
    if B:
        ms_int=ms*0.2
    else:
        ms_int=ms
    plt.plot(0, 0,'o', label = 'B', ms=ms_int, color = (38/255, 70/255, 83/255))
    if C:
        ms_int=ms*0.2
    else:
        ms_int=ms
    plt.plot(2, 0,'o', label = 'C', ms=ms_int, color = (244/255, 162/255, 97/255))
    plt.axis('off')
    plt.axis([-1,3,-0.5,1.5])
    
    plt.subplot(133)
    if A:
        ms_int=ms
    else:
        ms_int=ms*0.2
    plt.plot(1, 1,'o', label = 'A', ms=ms_int, color = (42/255, 157/255, 143/255))    
    if B:
        ms_int=ms
    else:
        ms_int=ms*0.2
    plt.plot(0, 0,'o', label = 'B', ms=ms_int, color = (38/255, 70/255, 83/255))
    if C:
        ms_int=ms
    else:
        ms_int=ms*0.2
    plt.plot(2, 0,'o', label = 'C', ms=ms_int, color = (244/255, 162/255, 97/255))
    plt.axis('off')
    plt.axis([-1,3,-0.5,1.5])
    
    plt.subplot(132)
    plt.plot(np.array([0,1,1,2,2,3]), np.array([0,0,1,1,0,0]), color= (233/255, 196/255, 106/255), lw = 5)
    plt.arrow(0.5, 2, 2, 0, color= (233/255, 196/255, 106/255), lw = 5, head_width=.1)
    plt.axis([-1,4,-2,4])
    plt.axis('off')
    
def filtrar_redes_sin_PFL(matrices_redes):
    redes_sin_fp = []
    for id_Red in range(16038):
        # Saco las autoregulaciones positivas
        if matrices_redes[id_Red][(0,0)]!=1:
            if matrices_redes[id_Red][(1,1)]!=1:
                if matrices_redes[id_Red][(2,2)]!=1:
                    # Saco los dobles positivos
                    if not(matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
                        if not(matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
                            if not(matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,1)]==1):
                                # Saco los dobles negativos
                                if not(matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
                                    if not(matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
                                        if not(matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1):
                                            # Saco los triples positivos antihorario y horario
                                            if not(matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
                                                if not(matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
                                                    # Saco los triples positivo - negativo - negativo antihorario
                                                    if not(matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
                                                        if not(matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==-1):
                                                            if not(matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==1):
                                                                # Saco los triples positivo - negativo - negativo antihorario
                                                                if not(matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
                                                                    if not(matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==-1):
                                                                        if not(matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==1):
                                                                            redes_sin_fp.append(id_Red)
    return redes_sin_fp

def filtrar_redes_con_PFL(matrices_redes):
    redes_sin_fp = []
    for id_Red in range(16038):
        bandera=False
        # Saco las autoregulaciones positivas
        if matrices_redes[id_Red][(0,0)]==1:
            bandera=True
        if matrices_redes[id_Red][(1,1)]==1:
            bandera=True
        if matrices_redes[id_Red][(2,2)]==1:
            bandera=True
        # Saco los dobles positivos
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,1)]==1):
            bandera=True
        # Saco los dobles negativos
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1):
            bandera=True
        # Saco los triples positivos antihorario y horario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            bandera=True
        # Saco los triples positivo - negativo - negativo antihorario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==1):
            bandera=True
        # Saco los triples positivo - negativo - negativo antihorario
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==1):
            bandera=True
        if bandera:
            redes_sin_fp.append(id_Red)
    return redes_sin_fp

def filtrar_redes_con_PFL_A(matrices_redes):
    redes_sin_fp = []
    for id_Red in range(16038):
        bandera=False
        # Saco las autoregulaciones positivas
        if matrices_redes[id_Red][(0,0)]==1:
            bandera=True
        if bandera:
            redes_sin_fp.append(id_Red)
    return redes_sin_fp

def filtrar_redes_con_PFL_B(matrices_redes):
    redes_sin_fp = []
    for id_Red in range(16038):
        bandera=False
        # Saco las autoregulaciones positivas
        if matrices_redes[id_Red][(1,1)]==1:
            bandera=True
        if bandera:
            redes_sin_fp.append(id_Red)
    return redes_sin_fp

def filtrar_redes_con_PFL_C(matrices_redes):
    redes_sin_fp = []
    for id_Red in range(16038):
        bandera=False
        # Saco las autoregulaciones positivas
        if matrices_redes[id_Red][(2,2)]==1:
            bandera=True
        if bandera:
            redes_sin_fp.append(id_Red)
    return redes_sin_fp  

def contar_PFL(matrices_redes):
    redes_fp_0 = []
    redes_fp_1 = []
    redes_fp_2 = []
    redes_fp_3 = []
    redes_fp_4 = []
    redes_fp_5 = []
    redes_fp_6 = []
    redes_fp_7 = []
    redes_fp_8 = []
    redes_fp_9 = []    
    for id_Red in range(16038):
        bandera=0
        # Saco las autoregulaciones positivas
        if matrices_redes[id_Red][(0,0)]==1:
            bandera+=1
        if matrices_redes[id_Red][(1,1)]==1:
            bandera+=1
        if matrices_redes[id_Red][(2,2)]==1:
            bandera+=1
        # Saco los dobles positivos
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            bandera+=1
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            bandera+=1
        if (matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,1)]==1):
            bandera+=1
        # Saco los dobles negativos
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera+=1
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera+=1
        if (matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1):
            bandera+=1
        # Saco los triples positivos antihorario y horario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            bandera+=1
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            bandera+=1
        # Saco los triples positivo - negativo - negativo antihorario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera+=1
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera+=1
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==1):
            bandera+=1
        # Saco los triples positivo - negativo - negativo antihorario
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera+=1
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera+=1
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==1):
            bandera+=1
        if bandera==0:
            redes_fp_0.append(id_Red)
        elif bandera==1:
            redes_fp_1.append(id_Red)
        elif bandera==2:
            redes_fp_2.append(id_Red)
        elif bandera==3:
            redes_fp_3.append(id_Red)
        elif bandera==4:
            redes_fp_4.append(id_Red)
        elif bandera==5:
            redes_fp_5.append(id_Red)
        elif bandera==6:
            redes_fp_6.append(id_Red)
        elif bandera==7:
            redes_fp_7.append(id_Red)
        elif bandera==8:
            redes_fp_8.append(id_Red)
        else:
            redes_fp_9.append(id_Red)
    return [redes_fp_0,redes_fp_1,redes_fp_2,redes_fp_3,redes_fp_4,redes_fp_5,redes_fp_6,redes_fp_7,redes_fp_8,redes_fp_9]                                                     

def filtrar_redes_con_doble_negativo(matrices_redes):
    redes_sin_fp = []
    for id_Red in range(16038):
        bandera=False
        # Saco los dobles negativos
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1):
            bandera=True
        if bandera:
            redes_sin_fp.append(id_Red)
    return redes_sin_fp

def filtrar_redes_con_doble_positivo(matrices_redes):
    redes_sin_fp = []
    for id_Red in range(16038):
        bandera=False
        # Saco los dobles positivos
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,1)]==1):
            bandera=True
        if bandera:
            redes_sin_fp.append(id_Red)
    return redes_sin_fp

def filtrar_redes_con_NFL(matrices_redes):
    redes_sin_fp = []
    for id_Red in range(16038):
        bandera=False
        # Saco los triples positivo - negativo - negativo antihorario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==1):
            bandera=True
        # Saco los triples positivo - negativo - negativo antihorario
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==1):
            bandera=True
        if bandera:
            redes_sin_fp.append(id_Red)
    return redes_sin_fp

def filtrar_redes_con_PFL_sin_NFL(matrices_redes):
    redes_sin_fn = []
    redes_con_fn = []
    for id_Red in range(16038):
        bandera=False
        # Saco las autoregulaciones positivas
        if matrices_redes[id_Red][(0,0)]==1:
            bandera=True
        if matrices_redes[id_Red][(1,1)]==1:
            bandera=True
        if matrices_redes[id_Red][(2,2)]==1:
            bandera=True
        # Saco los dobles positivos
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,1)]==1):
            bandera=True
        # Saco los dobles negativos
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1):
            bandera=True
        # Saco los triples positivos antihorario y horario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            bandera=True
        # Saco los triples positivo - negativo - negativo antihorario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==1):
            bandera=True
        # Saco los triples positivo - negativo - negativo antihorario
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==-1):
            bandera=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==1):
            bandera=True
        if not bandera:
            bandera=False
            # Saco los triples positivo - negativo - negativo antihorario
            if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==-1):
                bandera=True
            if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
                bandera=True
            if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==1):
                bandera=True
            # Saco los triples positivo - negativo - negativo antihorario
            if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==-1):
                bandera=True
            if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
                bandera=True
            if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==1):
                bandera=True
            if bandera:
                redes_con_fn.append(id_Red)
            else:
                redes_sin_fn.append(id_Red)
    return redes_sin_fn,redes_con_fn

def agregar_booleanos_feedbacks(MEMORIA,matrices_redes):
    
    # Feedbacks Positivos
    # Agrego las autoregulaciones positivas
    MEMORIA['FP_A']=False
    MEMORIA['FP_B']=False
    MEMORIA['FP_C']=False
    
    # Agrego los doble positivos
    MEMORIA['FP_DP_AB']=False
    MEMORIA['FP_DP_BC']=False
    MEMORIA['FP_DP_CA']=False
    
    # Agrego los dobles negativos
    MEMORIA['FP_DN_AB']=False
    MEMORIA['FP_DN_BC']=False
    MEMORIA['FP_DN_CA']=False
    
    # Agrego los triples positivos antihorario y horario
    MEMORIA['FP_PPP_ABC']=False    
    MEMORIA['FP_PPP_ACB']=False

    # Agrego los triples positivo - negativo - negativo antihorario
    MEMORIA['FP_PNN_ABC']=False
    MEMORIA['FP_PNN_BCA']=False
    MEMORIA['FP_PNN_CAB']=False
    
    # Agrego los triples positivo - negativo - negativo horario
    MEMORIA['FP_PNN_ACB']=False
    MEMORIA['FP_PNN_CBA']=False
    MEMORIA['FP_PNN_BAC']=False
    
    # Feedbacks negativos
    # Agrego las autoregulaciones negativas
    MEMORIA['FN_A']=False
    MEMORIA['FN_B']=False
    MEMORIA['FN_C']=False
    
    # Agrego los positivo-negativos
    MEMORIA['FN_AB']=False
    MEMORIA['FN_BC']=False
    MEMORIA['FN_CA']=False
    MEMORIA['FN_BA']=False
    MEMORIA['FN_CB']=False
    MEMORIA['FN_AC']=False
    
    # Agrego los triples positivo - positivo - negativo antihorario
    MEMORIA['FN_PPN_ABC']=False
    MEMORIA['FN_PPN_BCA']=False
    MEMORIA['FN_PPN_CAB']=False
    
    # Agrego los triples positivo - positivo - negativo horario
    MEMORIA['FN_PPN_ACB']=False    
    MEMORIA['FN_PPN_CBA']=False
    MEMORIA['FN_PPN_BAC']=False

    # Agrego los triples negativo - negativo - negativo antihorario
    MEMORIA['FN_NNN_ABC']=False
    MEMORIA['FN_NNN_CBA']=False
    
    for id_Red in range(16038):
        
        # Agrego las autoregulaciones positivas
        if matrices_redes[id_Red][(0,0)]==1:
            MEMORIA.at[id_Red,'FP_A']=True
        if matrices_redes[id_Red][(1,1)]==1:
            MEMORIA.at[id_Red,'FP_B']=True
        if matrices_redes[id_Red][(2,2)]==1:
            MEMORIA.at[id_Red,'FP_C']=True
            
        # Agrego los doble positivos
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            MEMORIA.at[id_Red,'FP_DP_AB']=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            MEMORIA.at[id_Red,'FP_DP_BC']=True
        if (matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,1)]==1):
            MEMORIA.at[id_Red,'FP_DP_CA']=True
            
        # Agrego los dobles negativos
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            MEMORIA.at[id_Red,'FP_DN_AB']=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
            MEMORIA.at[id_Red,'FP_DN_BC']=True
        if (matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1):
            MEMORIA.at[id_Red,'FP_DN_CA']=True
            
        # Agrego los triples positivos antihorario y horario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            MEMORIA.at[id_Red,'FP_PPP_ABC']=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            MEMORIA.at[id_Red,'FP_PPP_ACB']=True
            
        # Agrego los triples positivo - negativo - negativo antihorario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==-1):
            MEMORIA.at[id_Red,'FP_PNN_ABC']=True
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==-1):
            MEMORIA.at[id_Red,'FP_PNN_BCA']=True
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==1):
            MEMORIA.at[id_Red,'FP_PNN_CAB']=True
            
        # Agrego los triples positivo - negativo - negativo horario
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            MEMORIA.at[id_Red,'FP_PNN_ACB']=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==-1):
            MEMORIA.at[id_Red,'FP_PNN_CBA']=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==1):
            MEMORIA.at[id_Red,'FP_PNN_BAC']=True

        # Feedbacks negativos
        # Agrego las autoregulaciones negativas
        if matrices_redes[id_Red][(0,0)]==-1:
            MEMORIA.at[id_Red,'FN_A']=True
        if matrices_redes[id_Red][(1,1)]==-1:
            MEMORIA.at[id_Red,'FN_B']=True
        if matrices_redes[id_Red][(2,2)]==-1:
            MEMORIA.at[id_Red,'FN_C']=True
            
        # Agrego los positivo-negativos
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,0)]==-1):
            MEMORIA.at[id_Red,'FN_AB']=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,0)]==-1):
            MEMORIA.at[id_Red,'FN_BC']=True
        if (matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,1)]==-1):
            MEMORIA.at[id_Red,'FN_CA']=True
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,0)]==1):
            MEMORIA.at[id_Red,'FN_BA']=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,0)]==1):
            MEMORIA.at[id_Red,'FN_CB']=True
        if (matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,1)]==1):
            MEMORIA.at[id_Red,'FN_AC']=True

        # Agrego los triples positivo - positivo - negativo antihorario
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==-1):
            MEMORIA.at[id_Red,'FN_PPN_ABC']=True
        if (matrices_redes[id_Red][(0,1)]==-1 and matrices_redes[id_Red][(1,2)]==1 and matrices_redes[id_Red][(2,0)]==1):
            MEMORIA.at[id_Red,'FN_PPN_BCA']=True
        if (matrices_redes[id_Red][(0,1)]==1 and matrices_redes[id_Red][(1,2)]==-1 and matrices_redes[id_Red][(2,0)]==1):
            MEMORIA.at[id_Red,'FN_PPN_CAB']=True

        # Agrego los triples positivo - positivo - negativo horario
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==-1):
            MEMORIA.at[id_Red,'FN_PPN_ACB']=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==1 and matrices_redes[id_Red][(1,0)]==1):
            MEMORIA.at[id_Red,'FN_PPN_CBA']=True
        if (matrices_redes[id_Red][(0,2)]==1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==1):
            MEMORIA.at[id_Red,'FN_PPN_BAC']=True
            
        # Agrego los triples negativo - negativo - negativo
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            MEMORIA.at[id_Red,'FN_NNN_ABC']=True
        if (matrices_redes[id_Red][(0,2)]==-1 and matrices_redes[id_Red][(2,1)]==-1 and matrices_redes[id_Red][(1,0)]==-1):
            MEMORIA.at[id_Red,'FN_NNN_CBA']=True
            
def graficar_histogramas(MEMORIA,columna,subconjuntos,labels,todas=True,densidad=False):
    plt.close('all')
    plt.figure(0)
    histograma = plt.hist(MEMORIA[columna],100,density=False)
    plt.figure(1)
    bineado = np.zeros(len(histograma[1])-1)
    for i in range(len(bineado)):
        bineado[i] = (histograma[1][i]+histograma[1][i+1])*0.5
    if todas:
        plt.subplot(121)
        plt.fill_between(bineado,histograma[0],color=(0,0,0),alpha=0.25, label='Todas')
        plt.subplot(122)
        plt.fill_between(bineado,histograma[0],color=(0,0,0),alpha=0.25, label='Todas')
    histograma_i = []
    for i in range(len(subconjuntos)):
        plt.figure(0)
        histograma_i=plt.hist(MEMORIA[columna][subconjuntos[i]],histograma[1],density=densidad)
        plt.figure(1)
        plt.subplot(121)
        plt.fill_between(bineado,histograma_i[0],alpha=0.5,label=labels[i])
        plt.subplot(122)
        plt.fill_between(bineado,histograma_i[0],alpha=0.5,label=labels[i])
    plt.yscale('log');plt.legend();sns.set_style(style='white')
    plt.legend();sns.set_style(style='white');plt.show()
    plt.close(0)

def graficar_id(id_Red):
    Red = clasemotif.Motif(matrices_redes[id_Red], lista_parametros[0])
    Red.graficar(grosor_parametros = False)

def graficar_matriz(matriz):
    id_Red = matrices_a_id_redes[matriz]
    Red = clasemotif.Motif(matrices_redes[id_Red], lista_parametros[0])
    Red.graficar(grosor_parametros = False)
#%% Cargar DataFrame MEMORIA
nombre_archivo = 'MEMORIA'
with open(nombre_archivo+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
    MEMORIA = pickle.load(f)
#%% Creo la lista de parametros con Latin Hypercube Sampling
lista_parametros = armar_lista_parametros()
#%% Creo la lista de las redes con todas las combinaciones posibles
matrices_redes = crear_redes_I_A_B_C_O()
#%% Creo diccionario con key = matriz, value = id_Red, para buscarlas facil
matrices_a_id_redes = crear_diccionario_matriz_id_red(matrices_redes)
#%%
# Positivos
Filtro_P = MEMORIA['FP_A'] | MEMORIA['FP_B'] | MEMORIA['FP_C']
Filtro_PP = MEMORIA['FP_DP_AB'] | MEMORIA['FP_DP_BC'] | MEMORIA['FP_DP_CA']
Filtro_NN = MEMORIA['FP_DN_AB'] | MEMORIA['FP_DN_BC'] | MEMORIA['FP_DN_CA']
Filtro_PPP = MEMORIA['FP_PPP_ABC'] | MEMORIA['FP_PPP_ACB']
Filtro_PNN = MEMORIA['FP_PNN_ABC'] | MEMORIA['FP_PNN_BCA']  | MEMORIA['FP_PNN_CAB'] | MEMORIA['FP_PNN_ACB'] | MEMORIA['FP_PNN_CBA']  | MEMORIA['FP_PNN_BAC']
# Negativos
Filtro_N = MEMORIA['FN_A'] | MEMORIA['FN_B'] | MEMORIA['FN_C']
Filtro_PN = MEMORIA['FN_AB'] | MEMORIA['FN_BC'] | MEMORIA['FN_CA'] | MEMORIA['FN_BA'] | MEMORIA['FN_CB'] | MEMORIA['FN_AC']
Filtro_PPN = MEMORIA['FN_PPN_ABC'] | MEMORIA['FN_PPN_BCA'] | MEMORIA['FN_PPN_CAB'] | MEMORIA['FN_PPN_ACB'] | MEMORIA['FN_PPN_CBA'] | MEMORIA['FN_PPN_BAC']
Filtro_NNN = MEMORIA['FN_NNN_ABC'] | MEMORIA['FN_NNN_CBA']
# Todes
Filtro_FP = Filtro_P | Filtro_PP | Filtro_NN | Filtro_PPP | Filtro_PNN
Filtro_no_FP = Filtro_FP==False


#%% Armo un dataframe con las redes
def filtrar_redes_vecinas_sacando_enlaces(lista_enlaces_motif, Filtro):
    redes_vecinas = []
    redes_centrales = []
    for id_Red in range(16038):
        # if Filtro[id_Red]:
            matriz = matrices_redes[id_Red].copy()
            agregar_red = len(lista_enlaces_motif)
            diferencia_enlaces = 0
            for enlace in lista_enlaces_motif:
                if matriz[enlace[0]] == enlace[1]:
                    agregar_red -= 1
                diferencia_enlaces += np.abs(matriz[enlace[0]] - enlace[1])
            if agregar_red==1 and diferencia_enlaces==1:
                redes_vecinas.append(id_Red)
                for enlace in lista_enlaces_motif:
                    matriz[enlace[0]] = enlace[1]
                redes_centrales.append(matrices_a_id_redes[str(matriz)])
    return redes_vecinas,redes_centrales

def armar_dataframe_vecinos_motifs(enlaces_motifs):
    diferencias = pd.DataFrame()
    controles = pd.DataFrame()
    for motifs in enlaces_motifs.keys():
        redes_vecinas,redes_centrales = filtrar_redes_vecinas_sacando_enlaces(enlaces_motifs[motifs],Filtro_no_FP)
        diferencias_i = pd.DataFrame()
        controles_i = pd.DataFrame()
        diferencias_i[motifs] = np.array(MEMORIA['Sets C'][redes_centrales])-np.array(MEMORIA['Sets C'][redes_vecinas])
        diferencias = pd.concat([diferencias,diferencias_i], ignore_index=True, axis=1)
        controles_i[motifs] = controles_vecinos_motifs(diferencias_i[motifs].copy())
        controles = pd.concat([controles,controles_i], ignore_index=True, axis=1)

    diferencias.columns = enlaces_motifs.keys()
    return diferencias,controles

def controles_vecinos_motifs(diferencias_i):
    N_boot = 10000
    medias = np.zeros(N_boot)
    for i_boot in range(N_boot):
        cambios = random.sample(range(len(diferencias_i)),int(len(diferencias_i)*0.5))
        diferencias_i[cambios] = -diferencias_i[cambios]
        medias[i_boot] = diferencias_i[cambios].mean()
    return medias

def graficar_diferencias_controles(diferencias,controles):
    # plt.figure()
    # p = plt.plot(range(len(controles.median())),np.array(controles.median()))
    # plt.fill_between(list(range(len(controles.median()))),np.percentile(controles,25,axis=0),np.percentile(controles,75,axis=0),alpha=0.25, color=p[0].get_color())
    # plt.fill_between(list(range(len(controles.median()))),np.percentile(controles,2.5,axis=0),np.percentile(controles,97.5,axis=0),alpha=0.25, color=p[0].get_color())
    # plt.plot(diferencias.mean(),'o')
    # plt.xticks(rotation=45)    
    # plt.axis([-1,len(controles.median()),-150,150])
    
    plt.figure()
    p = plt.plot(range(len(controles.median())),np.zeros(len(controles.median())))
    plt.fill_between(list(range(len(controles.median()))),(np.percentile(controles,25,axis=0)-controles.median())/np.std(controles),(np.percentile(controles,75,axis=0)-controles.median())/np.std(controles),alpha=0.25, color=p[0].get_color())
    plt.fill_between(list(range(len(controles.median()))),(np.percentile(controles,2.5,axis=0)-controles.median())/np.std(controles),(np.percentile(controles,97.5,axis=0)-controles.median())/np.std(controles),alpha=0.25, color=p[0].get_color())
    plt.plot((np.array(diferencias.mean())-controles.median())/np.std(controles),'o')
    plt.xticks(list(range(len(controles.median()))),labels=list(diferencias.columns),rotation=45)    
    plt.axis([-1,len(controles.median()),-30,30])

enlaces_FP = {}
enlaces_FP.update({'FP_A':[[(0,0),1]]})
enlaces_FP.update({'FP_B':[[(1,1),1]]})
enlaces_FP.update({'FP_C':[[(2,2),1]]})
enlaces_FP.update({'FP_PP_AB':[[(0,1),1],[(1,0),1]]})
enlaces_FP.update({'FP_PP_BC':[[(1,2),1],[(2,1),1]]})
enlaces_FP.update({'FP_PP_CA':[[(0,2),1],[(2,0),1]]})
enlaces_FP.update({'FP_NN_AB':[[(0,1),-1],[(1,0),-1]]})
enlaces_FP.update({'FP_NN_BC':[[(1,2),-1],[(2,1),-1]]})
enlaces_FP.update({'FP_NN_CA':[[(0,2),-1],[(2,0),-1]]})
enlaces_FP.update({'FP_ABC':[[(0,1),1],[(1,2),1],[(2,0),1]]})
enlaces_FP.update({'FP_ACB':[[(0,2),1],[(2,1),1],[(1,0),1]]})
enlaces_FP.update({'FP_PNN_ABC':[[(0,1),1],[(1,2),-1],[(2,0),-1]]})
enlaces_FP.update({'FP_PNN_BCA':[[(0,1),-1],[(1,2),1],[(2,0),-1]]})
enlaces_FP.update({'FP_PNN_CAB':[[(0,1),-1],[(1,2),-1],[(2,0),1]]})
enlaces_FP.update({'FP_PNN_ACB':[[(0,2),1],[(2,1),-1],[(1,0),-1]]})
enlaces_FP.update({'FP_PNN_CBA':[[(0,2),-1],[(2,1),1],[(1,0),-1]]})
enlaces_FP.update({'FP_PNN_BAC':[[(0,2),-1],[(2,1),-1],[(1,0),1]]})

diferencias_FP,controles_FP = armar_dataframe_vecinos_motifs(enlaces_FP)
graficar_diferencias_controles(diferencias_FP,controles_FP)

enlaces_FN = {}
enlaces_FN.update({'N_A':[[(0,0),-1]]})
enlaces_FN.update({'N_B':[[(1,1),-1]]})
enlaces_FN.update({'N_C':[[(2,2),-1]]})
enlaces_FN.update({'PN_AB':[[(0,1),1],[(1,0),-1]]})
enlaces_FN.update({'PN_BC':[[(1,2),1],[(2,1),-1]]})
enlaces_FN.update({'PN_CA':[[(0,2),1],[(2,0),-1]]})
enlaces_FN.update({'PN_BA':[[(0,1),-1],[(1,0),1]]})
enlaces_FN.update({'PN_CB':[[(1,2),-1],[(2,1),1]]})
enlaces_FN.update({'PN_AC':[[(0,2),-1],[(2,0),1]]})
enlaces_FN.update({'PPN_ABC':[[(0,1),1],[(1,2),1],[(2,0),-1]]})
enlaces_FN.update({'PPN_BCA':[[(0,1),-1],[(1,2),1],[(2,0),1]]})
enlaces_FN.update({'PPN_CAB':[[(0,1),1],[(1,2),-1],[(2,0),1]]})
enlaces_FN.update({'PPN_ACB':[[(0,2),1],[(2,1),1],[(1,0),-1]]})
enlaces_FN.update({'PPN_CBA':[[(0,2),-1],[(2,1),1],[(1,0),1]]})
enlaces_FN.update({'PPN_BAC':[[(0,2),1],[(2,1),-1],[(1,0),1]]})
enlaces_FN.update({'NNN_ABC':[[(0,1),-1],[(1,2),-1],[(2,0),-1]]})
enlaces_FN.update({'NNN_ACB':[[(0,2),-1],[(2,1),-1],[(1,0),-1]]})

diferencias_FN, control_FN = armar_dataframe_vecinos_motifs(enlaces_FN)
graficar_diferencias_controles(diferencias_FN,control_FN)

enlaces_IFFL = {}
enlaces_IFFL.update({'P_AC_PN_ABC':[[(0,2),1],[(0,1),1],[(1,2),-1]]})
enlaces_IFFL.update({'P_AC_NP_ABC':[[(0,2),1],[(0,1),-1],[(1,2),1]]})
enlaces_IFFL.update({'N_AC_PP_ABC':[[(0,2),-1],[(0,1),1],[(1,2),1]]})
enlaces_IFFL.update({'N_AC_NN_ABC':[[(0,2),-1],[(0,1),-1],[(1,2),-1]]})

diferencias_IFFL, control_IFFL = armar_dataframe_vecinos_motifs(enlaces_IFFL)
graficar_diferencias_controles(diferencias_IFFL,control_IFFL)

enlaces_CFFL = {}
enlaces_CFFL.update({'P_AC_PP_ABC':[[(0,2),1],[(0,1),1],[(1,2),1]]})
enlaces_CFFL.update({'P_AC_NN_ABC':[[(0,2),1],[(0,1),-1],[(1,2),-1]]})
enlaces_CFFL.update({'N_AC_PN_ABC':[[(0,2),-1],[(0,1),1],[(1,2),-1]]})
enlaces_CFFL.update({'N_AC_NP_ABC':[[(0,2),-1],[(0,1),-1],[(1,2),1]]})

diferencias_CFFL, control_CFFL = armar_dataframe_vecinos_motifs(enlaces_CFFL)
graficar_diferencias_controles(diferencias_CFFL,control_CFFL)

enlaces_BI = {}
enlaces_BI.update({'N_CA':[[(2,0),-1]]})
enlaces_BI.update({'N_AC':[[(0,2),-1]]})
enlaces_BI.update({'NN_AC':[[(0,2),-1],[(2,0),-1]]})

diferencias_BI, control_BI = armar_dataframe_vecinos_motifs(enlaces_BI)
graficar_diferencias_controles(diferencias_BI,control_BI)

enlaces_multi = {}
enlaces_multi.update({'NN_AC':[[(0,2),-1],[(2,0),-1]]})
enlaces_multi.update({'NN_AB':[[(0,1),-1],[(1,0),-1]]})
enlaces_multi.update({'NN_BC':[[(1,2),-1],[(2,1),-1]]})
enlaces_multi.update({'NN_AC_AB':[[(0,2),-1],[(2,0),-1],[(0,1),-1],[(1,0),-1]]})
enlaces_multi.update({'NN_AC_BC':[[(0,2),-1],[(2,0),-1],[(1,2),-1],[(2,1),-1]]})
enlaces_multi.update({'NN_AB_BC':[[(0,1),-1],[(1,0),-1],[(1,2),-1],[(2,1),-1]]})
enlaces_multi.update({'NN_AB_BC_AC':[[(0,1),-1],[(1,0),-1],[(1,2),-1],[(2,1),-1],[(0,2),-1],[(2,0),-1]]})

diferencias_multi, control_multi = armar_dataframe_vecinos_motifs(enlaces_multi)
graficar_diferencias_controles(diferencias_multi,control_multi)
#%% Grafico Distribución con autoregulacion positiva A/B/C
n = 1700
graficar_id(redes_vecinas[n])
graficar_id(redes_centrales[n])

#%%
def graficar_motifs(enlaces_motifs, guardar = False):
    for motif in enlaces_motifs.keys():
        matriz = np.array([[0,0,0],[0,0,0],[0,0,0]])
        for enlace in enlaces_motifs[motif]:
            matriz[enlace[0]] = enlace[1]
        Red = clasemotif.Motif(matriz,lista_parametros[0])
        Red.graficar(grosor_parametros=False, enzimas=False)
        if guardar:
            plt.savefig(motif + '.png')
        
graficar_motifs(enlaces_IFFL, guardar = True)
graficar_motifs(enlaces_FP, guardar = True)
graficar_motifs(enlaces_FN, guardar = True)

#%%
def graficar_motifs_labels(enlaces_motifs, guardar = False):
    N = len(enlaces_motifs.keys())
    plt.figure()
    i_motif = 0
    for motif in enlaces_motifs.keys():
        i_motif += 1
        plt.subplot(1,N,i_motif)
        matriz = np.array([[0,0,0],[0,0,0],[0,0,0]])
        for enlace in enlaces_motifs[motif]:
            matriz[enlace[0]] = enlace[1]
        Red = clasemotif.Motif(matriz,lista_parametros[0])
        Red.graficar(grosor_parametros=False, nueva_fig = False, enzimas=False)
        # if guardar:
        #     plt.savefig(motif + '.png')
        
graficar_motifs_labels(enlaces_IFFL, guardar = True)
graficar_motifs_labels(enlaces_FP, guardar = True)
graficar_motifs_labels(enlaces_FN, guardar = True)
