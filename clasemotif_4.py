import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


class Motif:

    def __init__(self, matriz, param):

        self.matriz = matriz
        self.activaciones = matriz>0
        self.inhibiciones = matriz<0
        self.param = param
        self.param_act = param[1]*self.activaciones
        self.param_inh = param[1]*self.inhibiciones
        self.nodos_activar = np.sum(matriz[:,:]==1,axis=0)==0
        self.nodos_activar[0] = 0
        self.nodos_inhibir = np.sum(matriz[:,:]==-1,axis=0)==0
        self.enzima_act =  self.param[2] * self.nodos_activar
        self.enzima_inh =  self.param[3] * self.nodos_inhibir
        
        self.escala_temporal = 1/np.max([self.param[0],np.max(self.param_act + self.param_inh),np.max([self.enzima_act,self.enzima_inh])])


    def graficar(self, grosor_parametros = False, nueva_fig = True, enzimas = True):
        
        red = self.matriz
        param = self.param
        
        if nueva_fig:
            plt.figure()
            
        ms=40;lw=5;L=100
        
        color_A = (0.265625, 0.84375 , 0.265625)
        color_B = (0.3984375 , 0.609375  , 0.86328125)
        color_C = (0.859375 , 0.390625 , 0.8828125)
        x_A = 70.711; y_A = 70.711
        x_B = 0; y_B = 0
        x_C = 96.593; y_C = -25.882
        
        color_negro = (0.21875, 0.21875 , 0.21875)
        color_gris = (0.45703125, 0.45703125, 0.45703125)
        color_input = (0.94921875, 0.5078125 , 0.1953125)
        
        xx=np.linspace(-3,5,1000)
        R = 1.75
        plt.fill_between(xx, 0.2 + (R**2 - (xx-1)**2)**0.5*0.75, 0.2 - (R**2 - (xx-1)**2)**0.5*0.5, color = color_gris)        
        plt.plot(1, 1,'o', label = 'A', ms=ms, mec=color_negro, mew=2, color = color_A)
        plt.plot(0, 0,'o', label = 'B', ms=ms, mec=color_negro, mew=2, color = color_B)
        plt.plot(2, 0,'o', label = 'C', ms=ms, mec=color_negro, mew=2, color = color_C)
        
        color_act = (0.265625, 0.84375 , 0.265625) # verde
        style_act = patches.ArrowStyle.Simple(tail_width=0.5, head_width=13, head_length=13)
        color_inh = (0.90625, 0.46875, 0.46875) # rojo
        style_inh = patches.ArrowStyle.BarAB(widthA=0, angleA=None, widthB=6.0, angleB=None)
        
        #Grafico el grosor de acuerdo a k/K, que va entre 0.001 y 1000
        ancho_I = (np.log10(param[0]/param[4])+3)/6 * lw
        ancho_interacciones = (np.log10(param[1]/param[5])+3)/6 * lw
        ancho_E = (np.log10(self.enzima_act/param[6])+3)/6 * lw
        ancho_F = (np.log10(self.enzima_inh/param[7])+3)/6 * lw
        if not grosor_parametros:
            ancho_I = ancho_I*0+lw
            ancho_interacciones = ancho_interacciones*0+lw
            ancho_E = ancho_E*0+lw
            ancho_F = ancho_F*0+lw   
        

        # Input
        if enzimas:
            flecha = patches.FancyArrowPatch(posA=[0,1], posB=[1-0.25,1],connectionstyle="arc3,rad=.0", arrowstyle=style_act,color=color_input, lw = ancho_I)
            plt.gca().add_patch(flecha)
        
        if red[0,0]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[1+0.1,1+0.1], posB=[1-0.15,1+0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_interacciones[0,0])
            plt.gca().add_patch(flecha)  
        elif red[0,0]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[1+0.1,1+0.1], posB=[1-0.15,1+0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_interacciones[0,0])
            plt.gca().add_patch(flecha)  
        
        if red[1,0]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[0,0], posB=[1-0.1,1-0.15],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[1,0])
            plt.gca().add_patch(flecha)  
        elif red[1,0]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[0,0], posB=[1-0.1,1-0.15],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[1,0])
            plt.gca().add_patch(flecha)  
        
        if red[2,0]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[2,0], posB=[1+0.18,1-0.1],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[2,0])
            plt.gca().add_patch(flecha)
        elif red[2,0]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[2,0], posB=[1+0.18,1-0.1],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[2,0])
            plt.gca().add_patch(flecha)
        
        if red[0,1]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[1,1], posB=[0.05,0.16],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[0,1])
            plt.gca().add_patch(flecha)
        elif red[0,1]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[1,1], posB=[0.05,0.16],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[0,1])
            plt.gca().add_patch(flecha)
        
        if red[1,1]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[-0.2,0], posB=[-0.1,-0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_interacciones[1,1])
            plt.gca().add_patch(flecha)
        elif red[1,1]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[-0.2,0], posB=[-0.1,-0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_interacciones[1,1])
            plt.gca().add_patch(flecha)
        
        if red[2,1]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[2,0], posB=[0.25,0.03],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[2,1])
            plt.gca().add_patch(flecha)
        elif red[2,1]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[2,0], posB=[0.25,0.03],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[2,1])
            plt.gca().add_patch(flecha)
        
        if red[0,2]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[1,1], posB=[2-0.15,0.12],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[0,2])
            plt.gca().add_patch(flecha)
        elif red[0,2]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[1,1], posB=[2-0.15,0.12],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[0,2])
            plt.gca().add_patch(flecha)
        
        if red[1,2]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[0,0], posB=[2-0.23,-0.05],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[1,2])
            plt.gca().add_patch(flecha)
        elif red[1,2]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[0,0], posB=[2-0.23,-0.05],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_interacciones[1,2])
            plt.gca().add_patch(flecha)
        
        if red[2,2]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[2-0.1,0-0.1], posB=[2+0.1,0-0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_interacciones[2,2])
            plt.gca().add_patch(flecha)
        elif red[2,2]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[2-0.1,0-0.1], posB=[2+0.1,0-0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_interacciones[2,2])
            plt.gca().add_patch(flecha)
        
        # if enzimas:
        #     style = style_act;color = color_act
        #     if self.nodos_activar[1]:
        #         flecha = patches.FancyArrowPatch(posA=[-0.5,0.5], posB=[-0.15,0.15],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_E[1])
        #         plt.gca().add_patch(flecha)
        #     if self.nodos_activar[2]:
        #         style = style_act;color = color_act
        #         flecha = patches.FancyArrowPatch(posA=[2+0.5,0.5], posB=[2+0.15,0.15],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_E[2])
        #         plt.gca().add_patch(flecha)
            
        #     style = style_inh;color = color_inh
        #     if self.nodos_inhibir[0]:
        #         flecha = patches.FancyArrowPatch(posA=[2,1], posB=[1+0.25,1],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_F[0])
        #         plt.gca().add_patch(flecha)
        #     if self.nodos_inhibir[1]:
        #         flecha = patches.FancyArrowPatch(posA=[-0.75,0], posB=[-0.25,0],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_F[1])
        #         plt.gca().add_patch(flecha)
        #     if self.nodos_inhibir[2]:
        #         flecha = patches.FancyArrowPatch(posA=[2.75,0], posB=[2+0.25,0],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_F[2])
        #         plt.gca().add_patch(flecha)
        
        plt.axis('off')
        plt.axis([-1,3,-0.5,1.5])
        
    def dinamica(self, tiempo, variables, Input, activaciones, inhibiciones, enzimas_act, enzimas_inh):
        
        red = self.matriz
        
        ms=40;lw=10;
        
        
        plt.plot(1, 1,'o', label = 'A', ms=variables[0]*ms)
        plt.plot(0, 0,'o', label = 'B', ms=variables[1]*ms)
        plt.plot(2, 0,'o', label = 'C', ms=variables[2]*ms)
        
        color_act = (0.35,0.7,0.35) # verde
        style_act = patches.ArrowStyle.Simple(tail_width=0.5, head_width=13, head_length=13)
        color_inh = (0.65,0.4,0.4) # rojo
        style_inh = patches.ArrowStyle.BarAB(widthA=0, angleA=None, widthB=6.0, angleB=None)
        
        #Grafico el grosor de acuerdo a k/K, que va entre 0.001 y 1000
        ancho_I = Input*0.5 * lw
        ancho_activaciones = activaciones*0.5 * lw
        ancho_inhibiciones = inhibiciones*0.5 * lw
        ancho_E = enzimas_act*0.5 * lw
        ancho_F = enzimas_inh*0.5 * lw

        # #Grafico el grosor de acuerdo a k/K, que va entre 0.001 y 1000
        # ancho_I = np.log10(Input) * lw
        # ancho_activaciones = np.log10(activaciones) * lw
        # ancho_inhibiciones = np.log10(inhibiciones) * lw
        # ancho_E = np.log10(enzimas_act) * lw
        # ancho_F = np.log10(enzimas_inh) * lw        
        
        # Input
        flecha = patches.FancyArrowPatch(posA=[0,1], posB=[1-0.25,1],connectionstyle="arc3,rad=.0", arrowstyle=style_act,color=color_act, lw = ancho_I)
        plt.gca().add_patch(flecha)
        
        if red[0,0]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[1+0.1,1+0.1], posB=[1-0.15,1+0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_activaciones[0,0])
            plt.gca().add_patch(flecha)  
        elif red[0,0]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[1+0.1,1+0.1], posB=[1-0.15,1+0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_inhibiciones[0,0])
            plt.gca().add_patch(flecha)  
        
        if red[1,0]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[0,0], posB=[1-0.1,1-0.15],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_activaciones[1,0])
            plt.gca().add_patch(flecha)  
        elif red[1,0]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[0,0], posB=[1-0.1,1-0.15],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_inhibiciones[1,0])
            plt.gca().add_patch(flecha)  
        
        if red[2,0]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[2,0], posB=[1+0.18,1-0.1],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_activaciones[2,0])
            plt.gca().add_patch(flecha)
        elif red[2,0]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[2,0], posB=[1+0.18,1-0.1],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_inhibiciones[2,0])
            plt.gca().add_patch(flecha)
        
        if red[0,1]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[1,1], posB=[0.05,0.16],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_activaciones[0,1])
            plt.gca().add_patch(flecha)
        elif red[0,1]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[1,1], posB=[0.05,0.16],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_inhibiciones[0,1])
            plt.gca().add_patch(flecha)
        
        if red[1,1]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[-0.2,0], posB=[-0.1,-0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_activaciones[1,1])
            plt.gca().add_patch(flecha)
        elif red[1,1]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[-0.2,0], posB=[-0.1,-0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_inhibiciones[1,1])
            plt.gca().add_patch(flecha)
        
        if red[2,1]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[2,0], posB=[0.25,0.03],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_activaciones[2,1])
            plt.gca().add_patch(flecha)
        elif red[2,1]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[2,0], posB=[0.25,0.03],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_inhibiciones[2,1])
            plt.gca().add_patch(flecha)
        
        if red[0,2]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[1,1], posB=[2-0.15,0.12],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_activaciones[0,2])
            plt.gca().add_patch(flecha)
        elif red[0,2]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[1,1], posB=[2-0.15,0.12],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_inhibiciones[0,2])
            plt.gca().add_patch(flecha)
        
        if red[1,2]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[0,0], posB=[2-0.23,-0.05],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_activaciones[1,2])
            plt.gca().add_patch(flecha)
        elif red[1,2]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[0,0], posB=[2-0.23,-0.05],connectionstyle="arc3,rad=.2", arrowstyle=style,color=color, lw = ancho_inhibiciones[1,2])
            plt.gca().add_patch(flecha)
        
        if red[2,2]==1:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[2-0.1,0-0.1], posB=[2+0.1,0-0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_activaciones[2,2])
            plt.gca().add_patch(flecha)
        elif red[2,2]==-1:
            style = style_inh;color = color_inh
            flecha = patches.FancyArrowPatch(posA=[2-0.1,0-0.1], posB=[2+0.1,0-0.15],connectionstyle="arc3,rad=3", arrowstyle=style,color=color, lw = ancho_inhibiciones[2,2])
            plt.gca().add_patch(flecha)
        
        style = style_act;color = color_act
        if self.nodos_activar[1]:
            flecha = patches.FancyArrowPatch(posA=[-0.5,0.5], posB=[-0.15,0.15],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_E[1])
            plt.gca().add_patch(flecha)
        if self.nodos_activar[2]:
            style = style_act;color = color_act
            flecha = patches.FancyArrowPatch(posA=[2+0.5,0.5], posB=[2+0.15,0.15],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_E[2])
            plt.gca().add_patch(flecha)
        
        style = style_inh;color = color_inh
        if self.nodos_inhibir[0]:
            flecha = patches.FancyArrowPatch(posA=[2,1], posB=[1+0.25,1],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_F[0])
            plt.gca().add_patch(flecha)
        if self.nodos_inhibir[1]:
            flecha = patches.FancyArrowPatch(posA=[-0.75,0], posB=[-0.25,0],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_F[1])
            plt.gca().add_patch(flecha)
        if self.nodos_inhibir[2]:
            flecha = patches.FancyArrowPatch(posA=[2.75,0], posB=[2+0.25,0],connectionstyle="arc3,rad=.0", arrowstyle=style,color=color, lw = ancho_F[2])
            plt.gca().add_patch(flecha)
        
        plt.axis('off')
        # plt.axis([-1,3,-0.5,1.5])

        plt.axis([0,2,0.5,1.5])
