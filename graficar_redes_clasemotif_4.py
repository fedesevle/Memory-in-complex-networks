import matplotlib.pyplot as plt
import numpy as np
import random
import clasemotif_4
from scipy import interpolate
import time
import seaborn as sns
sns.set()
import pickle
sns.set_style("white")
#%%
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

#%% Creo la lista de las redes con todas las combinaciones posibles
matrices_redes = crear_redes_I_A_B_C_O()
#%% Redes interesantes
# id_Red = 1234 # DFN DFP
id_Red = 0 # Red lineal
id_Red = 16037 # Red lineal
# id_Red = 3076 # NFBL
# id_Red = 2482 # Red lineal + A se autoprende
# id_Red = 6676 # Red con doble negativo y positivo, mucha memoria
#%% Graficar un set de una red y su dinamica

params = [0.1,
 np.array([[0.0001, 1, 1], #el feedback lo puse en 2 para el FP y en 0.0001 para sin FP
        [1, 1, 1],
        [1, 1, 1]]),
 np.array([1 , 1, 1]),
 np.array([1, 1, 1]),
 1,
 0.5*np.array([[1, 1, 1],
        [1 , 1, 1],
        [1, 1, 1]]),
 0.5*np.array([ 1,  1, 1 ]),
 0.5*np.array([ 1,  1, 1])]

Red = clasemotif_4.Motif(matrices_redes[id_Red], params)
Red.graficar()

# Do the plot code

#%% Guardar redes random
N_redes = 80
for i in range(N_redes):
    id_Red = random.randint(0,16038)
    Red = clasemotif_4.Motif(matrices_redes[id_Red], params)
    Red.graficar()
    plt.savefig('C:\\Users\\fede_\\OneDrive\\Escritorio\\Figuras\\Figu 3\\redes\\Red'+str(i)+'.eps', format='eps')
    plt.close('all')