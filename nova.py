import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import seaborn as sns
import os

matplotlib.rcParams.update({'font.size': 18}) 
plt.figure(figsize=(8,6))

#cifar10
xaxis = [10,20,50,100]
fedavg = [40.45, 43.83, 48.66, 54.03]
fedavg_error = [0.57, 0.36, 0.17, 0.51]

fednova = [34.78, 40.15, 50.05, 54.60]
fednova_error = [0.32, 0.41, 0.13, 0.42]

ccfl = [40.32, 44.78, 50.21, 53.42]
ccfl_error = [0.28, 0.38, 0.08, 0.38]




x = np.arange(4)

total_width, n = 0.8, 4
width = total_width / n
#x = x - (total_width - width) / 2


plt.bar(x - width, fedavg,  width=width, yerr=fedavg_error, label='FedAvg(full)', color='#FF8C00')
plt.bar(x, fednova, width=width, yerr=fednova_error, label='FedNova', color='#81B8DF')
plt.bar(x + width, ccfl, width=width, yerr=ccfl_error, label='CC-FedAvg',color='#A6BCBC')
plt.xticks(x, ['10', '20', '50', '100'])
plt.ylim(20, 60)
plt.xlabel('E (# steps of local iteration in each round)')
plt.ylabel('accuracy')
plt.legend()

save=True
if save==True:
    #plt.savefig(os.path.join('C:\\Users\\zhh\\Documents\\mind\\computation_reduction\\ieee\\figs', 'nova_cifar10.eps'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(r'C:\\Users\\zhh\\Nutstore\\1\\mind\\面上\\NSFC-application-template-latex-main\\fig', '7-2.png'), dpi=600)
plt.show()


#cifar100
matplotlib.rcParams.update({'font.size': 18}) 
plt.figure(figsize=(8,6))

#fedavg = [26.24,	36.38,	48.50,	51.23] #,	51.31]
#fedavg_error = [0.52, 0.67, 0.45, 0.38] #, 0.58]
#xaxis = [10,20,50,100] # ,200]
#fednova = [18.13,	29.21,	42.40,	48.21] #,	51.00]
#fednova_error = [0.27, 0.56, 0.42, 0.49] #, 0.47]
#ccfl = [19.30,	34.73,	45.13,	48.53] #,	47.41]
#ccfl_error = [0.32, 0.42, 0.19, 0.36] #, 0.44]


xaxis = [10,20,50,100 ,200]
fedavg = [26.24,	36.38,	48.50,	51.23,	51.31]
fedavg_error = [0.52, 0.67, 0.45, 0.38, 0.58]
fednova = [18.13,	29.21,	42.40,	48.21,	51.00]
fednova_error = [0.27, 0.56, 0.42, 0.49, 0.47]
ccfl = [19.30,	34.73,	45.13,	48.53,	47.41]
ccfl_error = [0.32, 0.42, 0.19, 0.36, 0.44]

x = np.arange(5)

total_width, n = 0.8, 4
width = total_width / n
#x = x - (total_width - width) / 2


plt.bar(x - width, fedavg,  width=width, yerr=fedavg_error, label='FedAvg(full)', color='#FF8C00')
plt.bar(x, fednova, width=width, yerr=fednova_error, label='FedNova', color='#81B8DF')
plt.bar(x + width, ccfl, width=width, yerr=ccfl_error, label='CC-FedAvg',color='#A6BCBC')
plt.xticks(x, ['10', '20', '50', '100', '200'])
plt.ylim(15, 60)
plt.xlabel('E (# steps of local iteration in each round)')
plt.ylabel('accuracy')
plt.legend()

save=True
if save==True:
    #plt.savefig(os.path.join('C:\\Users\\zhh\\Documents\\mind\\computation_reduction\\ieee\\figs', 'nova_cifar100.eps'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(r'C:\\Users\\zhh\\Nutstore\\1\\mind\\面上\\NSFC-application-template-latex-main\\fig', '7-3.png'), dpi=600)
plt.show()





# Suppose variable `reward_sum` is a list containing all the reward summary scalars
def plot_with_variance(reward_mean, reward_std, xaxis, label, color='blue'):
    """plot_with_variance
        reward_mean: typr list, containing all the means of reward summmary scalars collected during training
        reward_std: type list, containing all variance
        savefig_dir: if not None, this must be a str representing the directory to save the figure
    """

    lower = [x - y for x, y in zip(reward_mean, reward_std)]
    upper = [x + y for x, y in zip(reward_mean, reward_std)]

    plt.plot(xaxis, reward_mean, '.-', color=color, label=label)
    plt.fill_between(xaxis, lower, upper, color=color, alpha=0.2)

matplotlib.rcParams.update({'font.size': 18}) 
plt.figure(figsize=(8,6))
#					10(round 500)	10(round 1000)	10(round 1500)	10(round 2000)			
#fedavg				42.50+0.14		47.85+0.46		51.98+0.21		54.13+0.42
#fednova				36.50+0.28		42.06+0.54		47.58+0.67		50.59+0.62
#ccfl				42.25+0.09		47.21+0.35		50.36+.018		52.87+0.24
xaxis = [500, 1000, 1500, 2000]
fedavg = [42.50, 47.85, 51.98, 54.13]
fedavg_error = [0.14, 0.46, 0.21, 0.42]

fednova = [36.50, 42.06, 47.58, 50.59]
fednova_error = [0.28, 0.54, 0.67, 0.62]

ccfl = [42.25, 47.21, 50.36, 52.87]
ccfl_error = [0.09, 0.35, 0.18, 0.24]

#plt.figure()
#plot_with_variance(fedavg, fedavg_error, xaxis, color='blue', label='FedAvg(full)')
#plot_with_variance(fednova, fednova_error, xaxis,  color='green', label='FedNova')
#plot_with_variance(ccfl, ccfl_error, xaxis,  color='red', label='EA-FedAvg')

plt.errorbar(xaxis, fedavg, yerr=fedavg_error, color='blue',  label='FedAvg(full)')
plt.errorbar(xaxis, fednova, yerr=fednova_error, color='green', label='FedNova')
plt.errorbar(xaxis, ccfl, yerr=ccfl_error, color='red', label='CC-FedAvg')
#plt.grid()
plt.legend()
plt.xlabel('T(# number of round)')
plt.ylabel('Accuracy')   


save=False
if save==True:
    plt.savefig(os.path.join('C:\\Users\\zhh\\Documents\\mind\\computation_reduction\\ieee\\figs', 'nova_local10.eps'), dpi=600, bbox_inches='tight')
plt.show()
