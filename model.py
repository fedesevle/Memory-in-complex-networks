import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle
from scipy import interpolate
import random 
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
import pickle

plt.close('all')

#%%
def Gillespie_FN_wo(cond_iniciales,t_max,I,N):
    A = [cond_iniciales[0]]
    B = [cond_iniciales[1]]
    C = [cond_iniciales[2]]
    
    E = 1 * N
    
    k_I   = 1
    k_AB  = 10
    k_BC  = 10
    k_CA  = 10
    k_EA  = 1  
    k_EB  = 1  
    k_EC  = 1  
    
    K_I   = 1   *N
    K_AB  = .01 *N
    K_BC  = .01 *N
    K_CA  = .1  *N
    K_EA  = .03 *N
    K_EB  = .01 *N
    K_EC  = .03 *N
    
    i = 0
    t = [0]
    while t[-1]<t_max:
        d_I  = I   *k_I *(N-A[i])/(N-A[i]+K_I )
        
        d_AB = A[i]*k_AB*  B[i]  /(B[i]  +K_AB)
        d_BC = B[i]*k_BC*(N-C[i])/(N-C[i]+K_BC)
        d_CA = C[i]*k_CA*(N-A[i])/(N-A[i]+K_CA)
        
        d_EA = E   *k_EA*  A[i]  /(A[i]  +K_EA)
        d_EB = E   *k_EB*(N-B[i])/(N-B[i]+K_EB)
        d_EC = E   *k_EC*  C[i]  /(C[i]  +K_EC)
        
        dR = d_I + d_AB+d_BC+d_CA + d_EA+d_EB+d_EC
        dt = np.random.exponential(1/dR)
        p  = np.random.rand()

        if 0<p and p<d_I/dR:
            A.append(A[i]+1)
            B.append(B[i])
            C.append(C[i])
        elif d_I/dR<p and p<(d_I+d_AB)/dR:
            A.append(A[i])
            B.append(B[i]-1)
            C.append(C[i])
        elif (d_I+d_AB)/dR<p and p<(d_I+d_AB+d_BC)/dR:
            A.append(A[i])
            B.append(B[i])
            C.append(C[i]+1)
        elif (d_I+d_AB+d_BC)/dR<p and p<(d_I+d_AB+d_BC+d_CA)/dR:
            A.append(A[i]+1)
            B.append(B[i])
            C.append(C[i])
        elif (d_I+d_AB+d_BC+d_CA)/dR<p and p<(d_I+d_AB+d_BC+d_CA+d_EA)/dR:
            A.append(A[i]-1)
            B.append(B[i])
            C.append(C[i])
        elif (d_I+d_AB+d_BC+d_CA+d_EA)/dR<p and p<(d_I+d_AB+d_BC+d_CA+d_EA+d_EB)/dR:
            A.append(A[i])
            B.append(B[i]+1)
            C.append(C[i])
        else:
            A.append(A[i])
            B.append(B[i])
            C.append(C[i]-1)
            
        t.append(t[i]+dt)
        i+=1
        
    return np.array(A)/N,np.array(B)/N,np.array(C)/N,t

def Gillespie_FN_w (cond_iniciales,t_max,I_l,I_h,t0,tf,N):
    A = [cond_iniciales[0]]
    B = [cond_iniciales[1]]
    C = [cond_iniciales[2]]
    
    E = 1 * N
    
    k_I   = 1
    k_AB  = 10
    k_BC  = 10
    k_CA  = 10
    k_EA  = 1  
    k_EB  = 1  
    k_EC  = 1  
    
    K_I   = 1   *N
    K_AB  = .01 *N
    K_BC  = .01 *N
    K_CA  = .1  *N
    K_EA  = .03 *N
    K_EB  = .01 *N
    K_EC  = .03 *N
    
    i = 0
    t = [0]
    while t[-1]<t_max:
        I = pulse(t0,tf,I_l,I_h,t[i])
        
        d_I  = I   *k_I *(N-A[i])/(N-A[i]+K_I )
        
        d_AB = A[i]*k_AB*  B[i]  /(B[i]  +K_AB)
        d_BC = B[i]*k_BC*(N-C[i])/(N-C[i]+K_BC)
        d_CA = C[i]*k_CA*(N-A[i])/(N-A[i]+K_CA)
        
        d_EA = E   *k_EA*  A[i]  /(A[i]  +K_EA)
        d_EB = E   *k_EB*(N-B[i])/(N-B[i]+K_EB)
        d_EC = E   *k_EC*  C[i]  /(C[i]  +K_EC)
        
        dR = d_I + d_AB+d_BC+d_CA + d_EA+d_EB+d_EC
        dt = np.random.exponential(1/dR)
        p  = np.random.rand()

        if 0<p and p<d_I/dR:
            A.append(A[i]+1)
            B.append(B[i])
            C.append(C[i])
        elif d_I/dR<p and p<(d_I+d_AB)/dR:
            A.append(A[i])
            B.append(B[i]-1)
            C.append(C[i])
        elif (d_I+d_AB)/dR<p and p<(d_I+d_AB+d_BC)/dR:
            A.append(A[i])
            B.append(B[i])
            C.append(C[i]+1)
        elif (d_I+d_AB+d_BC)/dR<p and p<(d_I+d_AB+d_BC+d_CA)/dR:
            A.append(A[i]+1)
            B.append(B[i])
            C.append(C[i])
        elif (d_I+d_AB+d_BC+d_CA)/dR<p and p<(d_I+d_AB+d_BC+d_CA+d_EA)/dR:
            A.append(A[i]-1)
            B.append(B[i])
            C.append(C[i])
        elif (d_I+d_AB+d_BC+d_CA+d_EA)/dR<p and p<(d_I+d_AB+d_BC+d_CA+d_EA+d_EB)/dR:
            A.append(A[i])
            B.append(B[i]+1)
            C.append(C[i])
        else:
            A.append(A[i])
            B.append(B[i])
            C.append(C[i]-1)
            
        t.append(t[i]+dt)
        i+=1
        
    return np.array(A)/N,np.array(B)/N,np.array(C)/N,t

def pulse(t0,tf,I_l,I_h,t):
    I = I_l
    if t>t0 and t<tf:
        I = I_h
    return I
    
#%%
N = 1000
I_l = .1 * N
I_h = 10 * N
t0 = 3
tf = 6
cond_iniciales = [0,0,0]
t_max = 100

A,B,C,t         = Gillespie_FN_wo(cond_iniciales,t_max,I_l,N)
#A_s,B_s,C_s,t_s = Gillespie_FN_w (cond_iniciales,t_max,I_l,I_h,t0,tf,N)


plt.figure()
plt.plot(t,A)
plt.plot(t,B)
plt.plot(t,C)

#%%
def gen_curves_wo(N_rep,N,cond_iniciales,t_max):
    I_l = .1 * N
    A_list = []
    t_list = []
    for i in range(N_rep):
        A,_,_,t         = Gillespie_FN_wo(cond_iniciales,t_max,I_l,N)
        A_list.append(A)
        t_list.append(t)
    return A_list,t_list

def get_t_peaks(A,t,b,a):
    smooth = filtfilt(b, a, A)
    argpeaks = find_peaks(smooth,height=.1,distance=4*N)[0]
    t0 = t[argpeaks[0]]
    t_peaks = []
    for peak in argpeaks:
        t_peaks.append(t[peak]-t0)
    return t_peaks, argpeaks

def gen_t_peaks_matrix(t_peaks_list):
    N_rep = len(t_peaks_list)
    N_peaks = np.zeros(N_rep)
    for i in range(N_rep):
        N_peaks[i] = len(t_peaks_list[i])
    N_peaks = int(min(N_peaks))
    peaks_times = np.zeros(N_rep*N_peaks).reshape(N_rep,N_peaks)
    for i in range(N_rep):
        peaks_times[i,:] = t_peaks_list[i][:N_peaks]
    return peaks_times,N_peaks
    
def gen_peak_list(A_list,t_list,N):
    N_rep = len(A_list)
    b, a = butter(4, 1/400*1000/N, btype='lowpass')
    t_peaks_list = []; argpeaks_list = []
    for i in range(N_rep):
        t_peaks, argpeaks = get_t_peaks(A_list[i],t_list[i],b,a)
        t_peaks_list.append(t_peaks)
        argpeaks_list.append(argpeaks)
    return t_peaks_list, argpeaks_list

#%%
np.logspace(2,4,5)

cond_iniciales = [0,0,0]
t_max          = 50
N_rep          = 1000
N              = 10000

A_list,t_list               = gen_curves_wo(N_rep,N,cond_iniciales,t_max)
t_peaks_list, argpeaks_list = gen_peak_list(A_list,t_list,N)
peaks_times,N_peaks         = gen_t_peaks_matrix(t_peaks_list)
T = peaks_times[:,-1].mean()/(N_peaks-1)
peaks_times_norm = peaks_times/T

A_list_s,t_list_s               = gen_curves_wo(N_rep,N,cond_iniciales,t_max)
t_peaks_lis_s, argpeaks_list_s  = gen_peak_list(A_list_s,t_list_s,N)
peaks_times_s,N_peaks_s         = gen_t_peaks_matrix(t_peaks_lis_s)
T_s = peaks_times_s[:,-1].mean()/(N_peaks_s-1)
peaks_times_norm_s = peaks_times_s/T_s

#to_save = [peaks_times_norm,peaks_times_norm_s,A_list,t_list,A_list_s,t_list_s]
to_save = [peaks_times_norm,peaks_times_norm_s]
# Save the workspace
with open(str(N)+".pkl", "wb") as f:
    pickle.dump(to_save, f)
    
#%%
A = A_list[0]
t = t_list[0]
argpeaks = argpeaks_list[0]
plt.figure()
plt.plot(t,A)
for arg in argpeaks:
    plt.plot(t[arg],A[arg],'ko')

#%%
def violins(peaks_times_norm):
    N_rep   = peaks_times_norm.shape[0]
    N_peaks = peaks_times_norm.shape[1]

    cmap = cm.get_cmap('tab20', N_peaks)
    plt.figure()
    violin_parts  = plt.violinplot(peaks_times_norm, vert=False,
                                   showmeans=False, showmedians=True,
                                   showextrema=False, widths=1)
    plt.yticks(ticks=np.arange(0, N_peaks), labels=np.arange(0, N_peaks))
    boxplot  = plt.boxplot(peaks_times_norm, vert=False, patch_artist=True,
                           widths=.3, showfliers=False)
    plt.yticks(ticks=np.arange(1, N_peaks+1), labels=np.arange(0, N_peaks))
    for i, body in enumerate(violin_parts['bodies']):
        body.set_facecolor(cmap(i))  # Cycle through colors if needed
        body.set_edgecolor('black')  # Optional: add a border color
        body.set_alpha(0.8)  # Set transparency
    violin_parts['cmedians'].set_color('black')  # Set the color of the median line
    violin_parts['cmedians'].set_linewidth(2)
    for box, color in zip(boxplot['boxes'], cmap(range(N_peaks))):
        box.set_facecolor(color)
    for median in boxplot['medians']:
        median.set_color('k')
        median.set_linewidth(2) 

def strip(peaks_times_norm,N_points):
    N_rep   = peaks_times_norm.shape[0]
    N_peaks = peaks_times_norm.shape[1]
    cmap = cm.get_cmap('tab20', N_peaks)
    plt.figure()
    for i in range(N_peaks):
        color = cmap(i)
        mean = peaks_times_norm[:,i].mean()
        plt.plot(peaks_times_norm[:N_points,i],np.linspace(0,1,N_points),'o',
                 color=color)
        plt.plot([mean,mean],[0,1], '--', color=color)
    plt.axis([-1,N_peaks+1,-0.1,1.1])

#%%
N_rep   = peaks_times_norm.shape[0]
N_peaks = peaks_times_norm.shape[1]
N_peaks_s = peaks_times_norm_s.shape[1]

N_peaks = min(N_peaks,N_peaks_s)
sens = .25

p_w_resp     = np.zeros(N_peaks-1)
p_wo_resp    = np.zeros(N_peaks-1)
for N_peak in range(N_peaks-1):
    p_w_resp[N_peak]   += sum(sum(np.abs(peaks_times_norm_s[:,:]+.5-(peaks_times_norm_s[:,N_peak].mean()+.5))<sens))
    p_wo_resp[N_peak]  += sum(sum(np.abs(peaks_times_norm[:,:]     -(peaks_times_norm_s[:,N_peak].mean()+.5))<sens))
    
p_w_resp  *= 100/N_rep
p_wo_resp *= 100/N_rep
#%%
plt.figure()
plt.plot(range(N_peaks-1),p_w_resp,'green')
plt.plot(range(N_peaks-1),p_wo_resp,'red')
plt.plot(range(N_peaks-1),np.zeros(N_peaks-1),   '--',color='grey')
plt.plot(range(N_peaks-1),np.ones(N_peaks-1)*100,'--',color='grey')