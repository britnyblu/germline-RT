import pybedtools
import itertools
import pandas as pd
import ToR_functions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
from pyliftover import LiftOver
from sklearn.neighbors import NearestNeighbors
from itertools import combinations, groupby,permutations
from operator import itemgetter
from scipy.stats.stats import pearsonr, spearmanr, ttest_ind
import statsmodels.formula.api as sm
from scipy.stats import ks_2samp, norm, skew, linregress
import scikits.bootstrap as bootstrap
from decimal import Decimal
import matplotlib as mpl

##General files regarding replication timing
gil_rt_switching_labels='/mnt/lustre/hms-01/fs01/britnyb/lab_files/gil_tor_data/RT_bin_labels_mouse/Supp-RT-mouse-40kb-bins_lables_switching_100kb.bed'
gil_rt_constant_labels='/mnt/lustre/hms-01/fs01/britnyb/lab_files/gil_tor_data/RT_bin_labels_mouse/Supp-RT-mouse-40kb-bins_lables_constant_100kb.bed'

##plot settings
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['axes.facecolor']='white'
plt.tick_params(direction="inout",bottom="on",left="on",width=1,length=6)
plt.rcParams['axes.linewidth']=1

##SIZE=20
##plt.rc('font', size=SIZE)  # controls default text sizes
##plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
##plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
##plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
##plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
##plt.rc('legend', fontsize=SIZE)  # legend fontsize
##plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title

class my_experiment_data:
    '''This stores all data and associated methods for a given ToR experiment
    combined_names=sample names
    exp=experiment name'''
    mouse_chroms=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','X']
    human_chroms=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
    chicken_chroms=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','32','W','Z']
    ToR_dir="/cs/icore/britnyb/lab"
    exps_file=ToR_dir+"/scripts/tor/ToR_exps_info.txt"
    output_dir="/mnt/lustre/hms-01/fs01/britnyb/lab_files/ToR_data"
    existing_exps=pd.read_csv(exps_file,sep="\t")["exp"].tolist()
    print("to make a new profile insert load,exp_name,genome,sorted_path,sample_names,reps,extra_names,pool,sep,async")
    print("already existing:"+str(existing_exps))
    sep="."
    main_seq_dir='/mnt/lustre/hms-01/fs01/britnyb/lab_files/tor_seq_data/'
    gil_dict={'CH12':'52673_CH12_ave','D3_ESC':'RT_D3_ESC_All.txt','MEF_F':'F_RT_MEF_MEF_All.txt','D3_NPC':'RT_D3_NPC_EBM9_All.txt','MEF_M':'M_RT_MEF_MEF_All.txt','EpiSC5':'RT_EpiSC5_EpiSC_All.txt',
              '46c_ESC':'RT_46C_ESC_All.txt','EpiSC7':'RT_EpiSC7_EpiSC_All.txt','46C_NPC':'RT_46C_NPC_ASd6_All.txt','endoderm':'RT_GscSox17-EB5_Endoderm_d6G+S+_All.txt',
              'mesoderm':'RT_GscSox17-EB5_Mesoderm_d6G+S-_All.txt','myo':'RT_J185a_Myoblast_All.txt',
              'L1210':'RT_L1210_Lymphoblastoid_All.txt','mammary':'RT_C127_mammary','MEL':'RT_MEL_Erythroleukemia','D3_EDM3':'RT_D3_EBM3_All.txt','trophoblast':'RT_TSC_Trophoblast',
              'D3_EBM6':'RT_D3_EBM6_All.txt','ipsc':'RT_iPSC_iPSC_All.txt','D3_EPL9':'RT_D3_EPL9_All.txt','CD4':'RT_Tc1_CD4+_CD4+_Peripheral_All.txt','Rif1_MEF':'RT_Rif1++1.txt',
              'Rif2_MEF':'RT_Rif1++2.txt','RT_REH_1':'RT_REH_1.txt','RT_REH_2':'RT_REH_2.txt','MCF7':'MCF7.txt'}
    def __init__(self,load,exp_name,genome=None,sorted_path=None,sample_names=[],reps=0,extra_names=None,pool=None,sep='.',async=None):
        self.updated_existing_exps=pd.read_csv(my_experiment_data.exps_file,sep="\t")["exp"].tolist()
        self.data=pd.read_csv(my_experiment_data.exps_file,sep="\t")

        if load==True:
            self.load=load
            if exp_name not in self.updated_existing_exps:
                raise NameError("experiment does not exist")
            self.genome=self.data["genome"][self.data["exp"]==exp_name].values[0]
            self.sorted=self.data["sorted_path"][self.data["exp"]==exp_name].values[0]
            self.reps=self.data["reps"][self.data["exp"]==exp_name].values[0]
            self.exp=exp_name 
            self.combined_names=self.data["sample_names"][self.data["exp"]==exp_name].values[0].replace(" ","").split(",")
            self.pool=self.data["pooled_G1"][self.data["exp"]==exp_name].values[0]
            self.sep=self.data["sep"][self.data["exp"]==exp_name].values[0]
            self.async=self.data["async"][self.data["exp"]==exp_name].values[0]
            if self.pool=='None':
                self.pool=None
            if self.sep=='None':
                self.sep=None
            if self.async=='None':
                self.async=None
        else:
            self.load=load
            if exp_name  in self.updated_existing_exps:
                raise NameError("experiment already exists")
            self.genome=genome #example: human
            if async==None:
                self.sorted=my_experiment_data.main_seq_dir+exp_name+'/sorted/'+sorted_path #example: %(exp)s-%(phase)s/%(exp)s-%(phase)s.chr%(chro)s.sorted'
            else:
                self.sorted=my_experiment_data.main_seq_dir+exp_name+'/gsnap/'+sorted_path
            self.names=sample_names # SMC1, WT, HU
            self.reps=reps #example: 3
            self.exp=exp_name #example: 'SMC1 vs other types' - to be used as the folder name in the main tor folder
            self.all_names=list(itertools.product(self.names,range(1,reps+1)))
            self.sep=sep
            print(self.all_names)
            self.combined_names=[b+sep+str(c) for (b,c) in self.all_names]
            if pool==None:
                self.pool=None
            else:
                self.pool=my_experiment_data.main_seq_dir+exp_name+'/sorted/'+pool #needs regex for exp and chro
            self.async=async

        if "hg" in self.genome:
            self.chroms=my_experiment_data.human_chroms
        elif "mm" in self.genome:
            self.chroms=my_experiment_data.mouse_chroms
        elif "gal" in self.genome:
            self.chroms=my_experiment_data.chicken_chroms
        if extra_names:
            self.combined_names.extend(extra_names)
            
    def manual_save(self):
        '''save the data for the experiment'''
        with open(my_experiment_data.exps_file,'a') as f:
            f.write(self.exp+"\t"+self.genome+"\t"+self.sorted+"\t"+", ".join(self.combined_names)+"\t"+str(self.reps)+"\t"+str(self.pool)+"\t"+str(self.sep)+"\n")

    def temp_compare_g1_bins(self,smoothing=10e-16,window_size=200,step=100000):
        for combo in self.combined_names:
            split_combo=combo.rsplit('.',1)
            name=split_combo[0]
            if len(split_combo)<2:
                rep=''
            else:
                rep=combo.rsplit('.',1)[-1]
            ToR_functions.windows_genome_temp(name,rep,smoothing,window_size,step,name=self.exp,chroms=self.chroms,genome=self.genome,sorted_path=self.sorted)
            break

    def create_ToR_gilbert(self,smoothing=10e-16,window_size=200,step=100000,pool_win_size=600):
        sorted_path='/mnt/lustre/hms-01/fs01/britnyb/lab_files/gil_tor_data/ToR/'
        for name in self.combined_names:
            file_name=my_experiment_data.gil_dict[name]
            ToR_functions.gilbert_windows_genome(exp=name,smoothing=smoothing,window_size=window_size,step=step,name=self.exp,chroms=self.chroms,genome=self.genome,file_name=file_name,sorted_path=sorted_path)

        ToR_functions.multi_interp(self.exp,self.combined_names,normalized=True)
        if self.load==False:
            with open(my_experiment_data.exps_file,'a') as f:
                f.write(self.exp+"\t"+self.genome+"\t"+self.sorted+"\t"+", ".join(self.combined_names)+"\t"+str(self.reps)+"\t"+str(self.pool)+"\t"+str(self.sep)+"\t"+str(self.async)+"\n")
        
    def add_in_sample(self,smoothing=10e-16,window_size=200,step=100000,pool_win_size=600):
        """Create ToR profile"""
        #this would be pretty useful
        pass        

    def create_ToR_normal(self,smoothing=10e-16,window_size=200,step=100000,pool_win_size=600,min_len=15,loess=False):
        """Create ToR profile"""
        for combo in self.combined_names:
            split_combo=combo.rsplit(self.sep,1)
            name=split_combo[0]
            if len(split_combo)<2:
                rep=''
            else:
                rep=combo.rsplit(self.sep,1)[-1]
            if self.pool:
                pooled_f=self.pool
                ToR_functions.windows_genome(name,rep,smoothing,pool_win_size,step,name=self.exp,chroms=self.chroms,genome=self.genome,sorted_path=self.sorted,pooled_file=pooled_f,sep=self.sep,min_len=min_len)
            elif self.async:
                ToR_functions.windows_genome(name,rep,smoothing,window_size,step,name=self.exp,chroms=self.chroms,genome=self.genome,sorted_path=self.sorted,sep=self.sep,allele='C57BL',min_len=min_len,loess=loess)
                ToR_functions.windows_genome(name,rep,smoothing,window_size,step,name=self.exp,chroms=self.chroms,genome=self.genome,sorted_path=self.sorted,sep=self.sep,allele='CAST',min_len=min_len,loess=loess)
            else:
                ToR_functions.windows_genome(name,rep,smoothing,window_size,step,name=self.exp,chroms=self.chroms,genome=self.genome,sorted_path=self.sorted,sep=self.sep,min_len=min_len,loess=loess)
        if self.async:
            self.combined_names=[b+self.sep+'C57BL' for b in self.combined_names]+[b+self.sep+'CAST' for b in self.combined_names]
        ToR_functions.multi_interp(name=self.exp,exp_names=self.combined_names)
        ToR_functions.multi_interp(self.exp,self.combined_names,normalized=True)
        
        if self.load==False:
            with open(my_experiment_data.exps_file,'a') as f:
                f.write(self.exp+"\t"+self.genome+"\t"+self.sorted+"\t"+", ".join(self.combined_names)+"\t"+str(self.reps)+"\t"+str(self.pool)+"\t"+str(self.sep)+"\t"+str(self.async)+"\n")        


    def mask_track(self,win_size='100'):
        for sample in self.combined_names:
            if self.async:
                allele=sample.rsplit('.',1)[1]
                sample=sample.rsplit('.',1)[0]
                if '.' in sample:
                    rep=sample.rsplit('.',1)[1]
                    sample=sample.rsplit('.',1)[0]
                else:
                    rep=None
                ToR_functions.create_mask(sample,self.exp,self.sorted,self.chroms,allele,rep,win_size='100')
            else:
                allele=None
                if '.' in sample:
                    rep=sample.rsplit('.',1)[1]
                    sample=sample.rsplit('.',1)[0]
                else:
                    rep=None
                ToR_functions.create_mask(sample,self.exp,self.sorted,self.chroms,allele,rep,win_size='100')
                 
               
            
    def create_ToR_sliding(self,smoothing=10e-16,window_size=250000,step=50000,min_win=3):
        """Create ToR profile
        this uses sliding windows - the step is the size it slides by and window_size is the window size"""
        for combo in self.combined_names:
            split_combo=combo.rsplit(self.sep,1)
            name=split_combo[0]
            if len(split_combo)<2:
                rep=''
            else:
                rep=combo.rsplit(self.sep,1)[-1]
            if self.pool:
                pooled_f=self.pool
                ToR_functions.windows_genome_sliding_window(name,rep,smoothing,pool_win_size,step,chroms=self.chroms,name=self.exp,genome=self.genome,sorted_path=self.sorted,pooled_file=pooled_f,sep=self.sep,min_win=min_win)
            elif self.async:
                ToR_functions.windows_genome_sliding_window(name,rep,smoothing,window_size,step,chroms=self.chroms,name=self.exp,genome=self.genome,sorted_path=self.sorted,sep=self.sep,allele='C57BL',min_win=min_win)
                ToR_functions.windows_genome_sliding_window(name,rep,smoothing,window_size,step,chroms=self.chroms,name=self.exp,genome=self.genome,sorted_path=self.sorted,sep=self.sep,allele='CAST',min_win=min_win)
            else:
                ToR_functions.windows_genome_sliding_window(name,rep,smoothing,window_size,step,chroms=self.chroms,name=self.exp,genome=self.genome,sorted_path=self.sorted,sep=self.sep,min_win=min_win)
        if self.async:
            self.combined_names=[b+self.sep+'C57BL' for b in self.combined_names]+[b+self.sep+'CAST' for b in self.combined_names]
        ToR_functions.multi_interp_sliding(name=self.exp,exp_names=self.combined_names)
        
        if self.load==False:
            with open(my_experiment_data.exps_file,'a') as f:
                f.write(self.exp+"\t"+self.genome+"\t"+self.sorted+"\t"+", ".join(self.combined_names)+"\t"+str(self.reps)+"\t"+str(self.pool)+"\t"+str(self.sep)+"\t"+str(self.async)+"\n")        


    def graph_ToR(self,chro,colors=[],saving_only=False,win_size='100',SIZE=8,norm=True,samples=[],save_figure_data=[],zoom=[]):
        """graph a ToR Profile """

        if len(samples)==0:
            rel_file=ToR_functions.multi_exp_file%{'name':self.exp,'win_size':win_size}
            with open(rel_file,'r') as f1:
                print(f1.readline().split('\t'))
            samples=[a+'.'+self.exp for a in input("copy the names of the data you want to compare separated by a comma\n")]
        ToR_functions.graph(self.exp,chro,samples,colors,saving_only=saving_only,win_size=win_size,SIZE=SIZE,norm=norm,save_figure_data=save_figure_data,zoom=zoom)

        
    def graph_with_other_project(self,chro,other_projects,colors=[],SIZE=8,samples=[],include_gc=False,include_mut=False,star_gc=False,win_size='100',norm=True,frame=[],save_figure_data=[],zoom=[]):
        """ Graph with data from another sequencing project
        If this will be used often it is ideal to just merge the data?"""
        if len(samples)==0:
            rel_file=ToR_functions.multi_exp_file%{'name':self.exp,'win_size':win_size}
            with open(rel_file,'r') as f1:
                print(f1.readline().split('\t'))
            samples=[a+'.'+self.exp for a in list(input("copy the names of the data you want to compare separated by a comma\n"))]            
            for name in other_projects:
                other_file=ToR_functions.multi_exp_file%{'name':name,'win_size':win_size}
                with open(other_file,'r') as f1:
                    print(f1.readline().split('\t'))
                samples.extend([a+'.'+name for a in list(input("copy the names of the data you want to compare separated by a comma\n"))])

        data=pd.read_csv(my_experiment_data.exps_file,sep="\t") #to check genomes for liftover
        other_genomes=[]
        for name in other_projects:
            other_genome=data["genome"][data["exp"]==name].values[0]
            other_genomes.append(other_genome)

        ToR_functions.graph_with_other(self.exp,chro,samples,colors,other_projects,[self.genome]+other_genomes,SIZE=SIZE,star_gc=star_gc,include_mut=include_mut,norm=norm,include_gc=include_gc,win_size=win_size,frame=frame,save_figure_data=save_figure_data,zoom=zoom)


    def simple_delta(self,other_project_names=None,win_size='100'):
        """ This takes as input pairs of reps in an experiment and plots the delta histogram"""

        ToR_functions.simple_delta(self.exp,other_project_names,win_size=win_size)
        

    def show_stats(self):
        '''This is likely no longer relevant '''
        ToR_functions.find_stats(name=self.exp,samples=self.names) ##old and less relevant
        print(ToR_functions.stat_file%{'name':self.exp})
        ToR_functions.find_stats(self.exp,self.names)
        with open(ToR_functions.stat_file%{'name':self.exp},'r') as f1:
            print(f1.readline().split('\t'))
        try:
            samples=input("copy the names of the data you want to compare separated by a comma\n")
        except SyntaxError:
            samples=[]
        cutoff=ToR_functions.show_stats(self.exp,samples)


    def differential_regions(self,other_project_name=None,chro='1',major=1,minor=.5, save=False,win_size='100'):
        """Also less relevant
        Create a file with differential regions
        Also plots a scatter plot showing data captured"""
 #       ToR_functions.determine_different(self.exp,sample_names,stds,cutoff_extend)
        rel_file=ToR_functions.multi_exp_file%{'name':self.exp,'win_size':win_size}
        with open(rel_file,'r') as f1:
            print(f1.readline().split('\t'))
        samples=[a+'.'+self.exp for a in list(input("copy the names of the data you want to compare separated by a comma\n"))]
        
        if other_project_name:
            rel_file=ToR_functions.multi_exp_file%{'name':other_project_name,'win_size':win_size}
            with open(rel_file,'r') as f1:
                print(f1.readline().split('\t'))
            samples.extend([a+'.'+other_project_name for a in list(input("copy the names of the data you want to compare separated by a comma\n"))])

        ToR_functions.determine_different_simple(self.exp,samples,chro,major,minor,other_project_name,save,win_size=win_size)

    def sequential_differential_regions(self,chro='1',samples=[],win_size='100'):
        if len(samples)==0:
            rel_file=ToR_functions.multi_exp_file%{'name':self.exp,'win_size':win_size}
            with open(rel_file,'r') as f1:
                print(f1.readline().split('\t'))
            samples=list(input("copy the names of the data you want to compare separated by a comma\n"))
        ToR_functions.determine_different_sequential(self.exp,samples,chro,win_size=win_size)
           

    def plot_stats_n(self,chro=1,n=6):
        """finds differential functions """
        ToR_functions.plot_stats_by_n(self.exp)
        ToR_functions.differential_based_on_stats_by_n(self.exp,n=n,chro=chro)

    def make_heatmap(self,other_projects=None,abc=False,method='spearman',filt=None,all_samps=False,manual_order=False,scale=None,samples=[],win_size='100',remove_x=False,save_figure_data=[]):
        """ A heatmap of correlations between different ToR profiles"""
        if len(samples)==0:
            rel_file=ToR_functions.multi_exp_file%{'name':self.exp,'win_size':win_size}
            with open(rel_file,'r') as f1:
                s=f1.readline().strip().split('\t')
            if all_samps==True:
                samples=[a+'.'+self.exp for a in s if "tor" in a]
            else:
                print(s)
                samples=[a+'.'+self.exp for a in list(input("copy the names of the data you want to compare separated by a comma\n"))]
            
            if other_projects!=None:
                for other_project in other_projects:
                    other_file=ToR_functions.multi_exp_file%{'name':other_project,'win_size':win_size}

                    with open(other_file,'r') as f1:
                        s=f1.readline().strip().split('\t')
                    if all_samps==True:
                        samples.extend([a+'.'+other_project for a in s if "tor" in a])
                    else:
                        print(s)
                        samples.extend([a+'.'+other_project for a in list(input("copy the names of the data you want to compare separated by a comma\n"))])
        other_genome=self.genome #place holder
  #      for other_project in other_projects:
    #        #prepare for theoretical liftover
      #      data=pd.read_csv(my_experiment_data.exps_file,sep="\t")
        #    other_genome=data["genome"][data["exp"]==other_project].values[0]
        ToR_functions.heatmap(self.exp,samples,other_projects,genomes=[self.genome,other_genome],abc=abc,method=method,filt=filt,manual_order=manual_order,scale=scale,win_size=win_size,remove_x=False,save_figure_data=save_figure_data)
        
    def plot_raw(self,chro,samples=[],win_size='100'):
        """ plot raw and smooth data together"""
        if len(samples)==0:
            rel_file=ToR_functions.multi_exp_file%{'name':self.exp,'win_size':win_size}
            with open(rel_file,'r') as f1:
                print([a.strip().split('_tor_')[-1] for a in f1.readline().split('\t')])
            samples=list(input("copy the names of the data you want to compare separated by a comma\n"))

        ToR_functions.plot_raw(self.exp,samples,chro,win_size=win_size)

    def chro_avgs(self,samples=[],win_size='100'):
        """Prints a list of the average ToR of each chromosome
        Works on the list of columns copied"""
        rel_file=ToR_functions.multi_exp_file%{'name':self.exp,'win_size':win_size}
        with open(rel_file,'r') as f1:
            print(f1.readline().split('\t'))
        if len(samples)==0:
            samples=list(input("copy the names of the data you want to compare separated by a comma\n"))
        data=pd.read_csv(rel_file,sep='\t',na_values='nan')
        chro_max=input('how many chromosomes are in your sample discluding X?')
        for sample in samples:
            print(sample+':')
            for chro in range(1,chro_max)+['X']:
                chro=str(chro)
                avg=data[sample][data['#1_chr']=='chr'+chro].mean()
                print(chro+' '+str(avg))

    def show_ttrs(self,chro='1',win_size='100'):
        """show a graph with ttrs marked"""
        rel_file=ToR_functions.multi_exp_file%{'name':self.exp,'win_size':win_size}
        with open(rel_file,'r') as f1:
            print(f1.readline().split('\t'))
        sample=input("copy the name of the data you want to view\n")

        ToR_functions.ttr_define(self.exp,sample,chro,win_size=win_size)

        if os.path.isfile(directory + '/'+self.exp+'/'+ name + '.pkl'):
           with open(directory + '/'+self.exp+'/'+ name + '.pkl', 'rb') as f:
            self.dict= pickle.load(f)

    def is_normal(self):
        ToR_functions.is_normal(self.exp)

    def quality_control(self,method='spearman',exps=[],all_samps=True,filt=gil_rt_constant_labels):
        print('assessing G1 coverage correlations')
        if exps==[]:
            rel_exps=[]
            for exp in self.existing_exps:
                if exp==self.exp:
                    continue
                if exp=='Gil_mm9':
                    continue
                if my_experiment_data(True,exp).genome==self.genome:
                    rel_exps.append(exp)
#           data=one_phase_cor_matrix(plot=True,exps=rel_exps,method=method)
        else:
            rel_exps=exps
        print('assessing standard tor')
        self.make_heatmap(other_projects=rel_exps,method=method,filt=filt,all_samps=all_samps)


####################### simple helpers ###################################
#helper function to append exp name to list for function input
def append(listname,prefix):
    return([a+'.'+prefix for a in listname])

def colnames(exp,win_size='100'):
    rel_file=ToR_functions.multi_exp_file%{'name':exp.exp,'win_size':win_size}
    with open(rel_file,'r') as f1:
        line=f1.readline().strip().split('\t')
    return(line)


######################### COVERAGE ################################
def coverage_bed(phase='G1',exps=[]):
    hund_kb_windows=1
    for exp in exps:
        exp_object=my_experiment_data(True,exp)
        g1_file=pd.read_csv(exp_object.sorted%{'exp':name,'rep':rep,'phase':phase,'chro':chro},header=None, names=['chro','loc'],sep='\t')



def one_phase_cor_matrix(cutoff=600,phase='G1',method='spearman',plot=True,exps=['SMC1_project_2015_nowt2','Small_cell_number_2016'],bin_size=100000,win_size='100',chroms=[],samples=[]):
    '''Create a correlation matrix comparing the coverage of a specific phase's reads in bins of a given size
    make sure to remove outliers - outliers to the high coverage side and areas where all samples have zero reads (repetitive regions)'''
    all_frames=[]
    for exp in exps:
        exp_object=my_experiment_data(True,exp)
        prev_samp='' #to use to prevent issues with pooled g1 data
        if len(samples)==0:
            samples=exp_object.combined_names
        for sample in samples:
            if chroms==[]:
                chroms=exp_object.chroms
            frames=[]
            if exp_object.sep==None:
                name=sample
                rep=''
            else:
                name=sample.rsplit(exp_object.sep,1)[0]
                rep=sample.rsplit(exp_object.sep,1)[-1]
            print(name,rep)

            #clause to prevent redundancy of pooled G1
            if exp_object.pool!=None and phase=='G1':
                if name==prev_samp: #we already did this pool
                    prev_samp=name
                    continue
                prev_samp=name

            for chro in chroms:
                if exp_object.pool!=None and phase=='G1':
                    g1_file=pd.read_csv(exp_object.pool%{'exp':name,'phase':phase,'chro':chro},header=None, names=['chro','loc'],sep='\t')
                    if chro=='1':
                        print(exp_object.pool%{'exp':name,'phase':phase,'chro':chro})
                else:
                    g1_file=pd.read_csv(exp_object.sorted%{'exp':name,'rep':rep,'phase':phase,'chro':chro},header=None, names=['chro','loc'],sep='\t')
                    if chro=='1':
                        print(exp_object.sorted%{'exp':name,'rep':rep,'phase':phase,'chro':chro})
                starts=range(0,g1_file['loc'].tail(1).values[0],bin_size)
                cur_data=pd.DataFrame({'count.'+sample+'.'+exp:g1_file.groupby(pd.cut(g1_file['loc'],starts)).size().values.tolist()})
                cur_data['chro']='chr'+chro
                cur_data['start']=starts[:len(cur_data)]
                cur_data['end']=starts[1:]
                frames.append(cur_data[['chro','start','end','count.'+sample+'.'+exp]])
            all_frames.append(pd.concat(frames,ignore_index=True))
    final_frame=reduce(lambda left,right: pd.merge(left,right,on=['chro','start','end'],how='outer'), all_frames)
    if plot==True:
        df=final_frame
        final_frame=final_frame[[col for col in final_frame.columns if 'count' in col and 'all' not in col]]
        final_frame=final_frame.dropna()
        final_frame=final_frame[(final_frame < cutoff).all(1)]
        for col in [a for a in final_frame.columns]:
            final_frame[col]=final_frame[col]/(final_frame[col].mean())
        print(len(final_frame))
        correlation_matrix=final_frame.corr(method=method)
        plt.pcolor(correlation_matrix,cmap='bwr')
        plt.yticks(np.arange(0.5, len(correlation_matrix.index), 1), correlation_matrix.index)
        plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns)
        plt.colorbar()
        plt.show()
    else:
        final_frame.to_csv('temp_cor_matrix_'+phase,sep='\t',na_rep='nan',index=None)
        return(final_frame)

def global_corr_matrices(exps=['SMC1_project_2015_nowt2','Small_cell_number_2016'],method='spearman'):
    """use the g1_cor_matrix function to make basic g1 or s data from a given list of experiments
    this creates global variables which can be used in further analyses"""
    global g1
    global s
    s=one_phase_cor_matrix(phase='S',plot=False,exps=exps,method=method)
    g1=one_phase_cor_matrix(phase='G1',plot=False,exps=exps,method=method)


def plot_coverage(s_data,g1_data,chro='1',outlier_cutoff=5):
    """This pipeline takes s and g phase coverage data and plots it"""
    #remove empty
    g1_data=g1_data.dropna()
    s_data=s_data.dropna()

    #remove outlier bins
    cutoff=outlier_cutoff
    for col in [a for a in g1_data.columns if 'count' in a]:
        g1_data[col]=g1_data[col].clip(upper=g1_data[col].mean()*cutoff)
    for col in [a for a in s_data.columns if 'count' in a]:
        s_data[col]=s_data[col].clip(upper=s_data[col].mean()*cutoff)

    #normalize
    for col in [a for a in s_data.columns if 'count' in a]:
        s_data[col]=s_data[col]/(s_data[col].mean())
    for col in [a for a in g1_data.columns if 'count' in a]:
        g1_data[col]=g1_data[col]/(g1_data[col].mean())

    print(g1_data.columns.values.tolist())
    g1_samples=list(input("copy the names of the g1 samples you want to compare separated by a comma\n"))
    s_samples=list(input("copy the names of the s samples you want to compare separated by a comma\n"))
    if chro=='all':
        for sample in s_samples:
            plt.plot(range(len(s_data)),s_data[sample],label=sample+'_S')
        for sample in g1_samples:
            plt.plot(range(len(g1_data)),g1_data[sample],label=sample+'_G1')
    else:
        for sample in s_samples:
            plt.plot(s_data['start'][s_data['chro']=='chr'+chro],s_data[sample][s_data['chro']=='chr'+chro],label=sample+'_S')
        for sample in g1_samples:
            plt.plot(g1_data['start'][g1_data['chro']=='chr'+chro],g1_data[sample][g1_data['chro']=='chr'+chro],label=sample+'_G1')
    plt.legend()
    plt.show()

def plot_coverage_single(g1_data,colors=[],chro='1',outlier_cutoff=5):
    """This pipeline takes s and g phase coverage data and plots it"""
    #remove empty
    g1_data=g1_data.dropna()

    #remove outlier bins
    cutoff=outlier_cutoff
    for col in [a for a in g1_data.columns if 'count' in a]:
        g1_data[col]=g1_data[col].clip(upper=g1_data[col].mean()*cutoff)

    #normalize
    for col in [a for a in g1_data.columns if 'count' in a]:
        g1_data[col]=g1_data[col]/(g1_data[col].mean())
    plt.rc('axes', color_cycle=ToR_functions.color_cycler(colors))
    print(g1_data.columns.values.tolist())
    g1_samples=list(input("copy the names of the g1 samples you want to compare separated by a comma\n"))
    if chro=='all':
        for sample in g1_samples:
            plt.plot(range(len(g1_data)),g1_data[sample],label=sample+'_G1')
    else:
        for sample in g1_samples:
            plt.plot(g1_data['start'][g1_data['chro']=='chr'+chro],g1_data[sample][g1_data['chro']=='chr'+chro],
                     label=ToR_functions.rename_frame([sample.split('count.')[1]])[0]+' G1 coverage')
    ax=plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def delta_reps(pattern,gen='mm9',win_size='100'):
    """this plots kde plots of replicates from experiments """
    rel_exps=[a for a in my_experiment_data.existing_exps if my_experiment_data(True,a).genome==gen and 'Gil' not in a]
    data=get_merged_data(rel_exps,win_size=win_size)
    print(data.columns.values.tolist())
    cols=list(input("choose two columns to compare or insert ''\n"))
    plt.rc('axes', color_cycle=ToR_functions.color_cycler(pattern))

    while len(cols)!=0:
        sns.kdeplot(data[cols[0]]-data[cols[1]],label=str(cols[0])+' '+str(cols[1]))
        cols=list(input("choose two columns to compare or insert ''\n"))
    plt.legend()
    plt.show()

def get_merged_data(exps,others=[],others_names=[],win_size='100',clean=False,norm=True,megabase=False,ops=None,filter_data=False):
    """given a list of experiments this merges them into one cohesive dataframe
    also accepts other files in which you must input data regarding the file names chosen"""
    print(exps[0])
    if norm==True or 'Gil' in exps[0]:
        path=ToR_functions.multi_exp_file%{'name':exps[0],'win_size':win_size}
    else:
        path=ToR_functions.multi_exp_file%{'name':exps[0],'win_size':'_not_normalized'+win_size}
    merge_cols=['#1_chr','2_start','3_end','4_mid']

    our_data=pd.read_csv(path,sep='\t',na_values='nan')
    our_data.columns=[a+'.'+exps[0] if a not in merge_cols else a for a in our_data.columns]

    rel_columns=[a for a in our_data.columns if 'tor' in a]
#    our_data=our_data[["#1_chr","4_mid"]+rel_columns]

    for exp in exps[1:]:
        print(exp)
        if norm==True or 'Gil' in exp:
            path=ToR_functions.multi_exp_file%{'name':exp,'win_size':win_size}
        else:
            path=ToR_functions.multi_exp_file%{'name':exp,'win_size':'_not_normalized'+win_size}
        new_data=pd.read_csv(path,sep='\t',na_values='nan')
        new_data.columns=[a+'.'+exp if a not in merge_cols else a for a in new_data.columns]
        new_data=new_data[[a for a in new_data.columns if 'tor' in a]+[a for a in merge_cols]]
        our_data=pd.merge(our_data,new_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False)
    for other in others:
        new_data=pd.read_csv(other,sep='\t',na_values='nan')
        cur_merge=[a for a in merge_cols if a in new_data.columns.values.tolist()]
        our_data=pd.merge(our_data,new_data,how='inner',left_on=cur_merge, right_on=cur_merge,sort=False)
    if filter_data:
        gil_data=pd.read_csv(filter_data,sep='\t')
        semi_merge=['#1_chr','4_mid']
        gil_data=gil_data[semi_merge]
        our_data=pd.merge(our_data,gil_data,how='inner',left_on=semi_merge, right_on=semi_merge,sort=False) 
      
    print(len(our_data))
    print('not removing na rows')
    if clean:
        columns=our_data.columns.values.tolist()
        for col in [a for a in our_data.columns.values.tolist() if 'tor' in a]:
            name=col.rsplit('.',1)[1]
            exp=col.rsplit('.',1)[0]
            exp=exp.split('tor_',1)[1]
            filt_f=ToR_functions.counts_file_clean%{'name':name,'exp':exp}
            filt=pd.read_csv(filt_f,sep='\t')
  #          print(our_data[col].isnull().sum())
            our_data=our_data.merge(filt,left_index=[0,1,2,3],right_index=[1,2,3,4],indicator=True,how='outer',suffixes=['','_y'])
            our_data[col][our_data['_merge']=='left_only']=np.nan
            our_data=our_data[columns]
  #          print(our_data[col].isnull().sum())

    if megabase:
        our_data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/temp_pre_mega.bed',index=False)
        our_data=our_data.dropna(axis=0,how='any')


        megabase=ToR_functions.output_dir+'features/mega_windows.bed'
        mega_bed=pybedtools.bedtool.BedTool(megabase)
        frame_bed=pybedtools.bedtool.BedTool.from_dataframe(our_data)
        mega_bed=mega_bed.intersect(frame_bed,wa=True,wb=True)
        all_cols=['#1_chr','2_start','3_end','4_mid']+[x for x in our_data.columns.values.tolist() if 'tor' in x or x in others_names]
        if ops:
            operations=['mean']+['mean']*len([i for i,x in enumerate(our_data.columns.values.tolist()) if 'tor' in x])+ops
        else:
            operations=['mean']+['mean']*len([i for i,x in enumerate(our_data.columns.values.tolist()) if 'tor' in x])+['sum']*len([i for i,x in enumerate(our_data.columns.values.tolist()) if x in others_names])
        col_nums=[4]+[5+a for a in [i for i,x in enumerate(our_data.columns.values.tolist()) if 'tor' in x]]
        col_nums=col_nums+[5+a for a in [i for i,x in enumerate(our_data.columns.values.tolist()) if x in others_names]]
        all_cols=['#1_chr','2_start','3_end','4_mid']+[our_data.columns.values.tolist()[i-5] for i in col_nums[1:]]
        non_loc_cols=[our_data.columns.values.tolist()[i-5] for i in col_nums[1:]]
 #       return(mega_bed,all_cols,col_nums,operations)
        mega_bed=mega_bed.merge(d=-1,c=col_nums,o=operations)
        print(len(all_cols),len(mega_bed[0].fields))
        our_data=mega_bed.to_dataframe(names=all_cols)
        our_data=our_data.dropna(axis=0,how='any')
        for col in non_loc_cols:
            our_data[col]=our_data[col].rolling(window=5).mean()

        our_data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/temp_post_mega.bed',index=False)
    ##clause to make evolution project easier
    if '5_gene_count' in our_data.columns.values:
        our_data['5_gene_count']=our_data['5_gene_count']*100
    if 'mutation_rate' in our_data.columns.values:
        our_data['mutation_rate']=our_data['mutation_rate']*100
    if '6_pct_gc' in our_data.columns.values:
        our_data['6_pct_gc']=our_data['6_pct_gc']*100
    
    return(our_data)


    our_data=our_data.dropna(axis=0,how='any')
    print('removed chr19 and x')
    our_data=our_data[our_data["#1_chr"]!='chrX']
    print(len(our_data))

    return(our_data)

#################### feature analysis ##########################
merge_cols=['#1_chr','4_mid']
cluster_order_removed=['15_tor_trophoblast.Gil_mm9', '9_tor_PGC-all.PGC', '9_tor_Sperm.all.PGC_nov16', '19_tor_ipsc.Gil_mm9', '5_tor_46c_ESC.Gil_mm9', '21_tor_D3_ESC.Gil_mm9', '6_tor_D3_EPL9.Gil_mm9',
               '16_tor_D3_EDM3.Gil_mm9', '7_tor_mesoderm.Gil_mm9', '18_tor_endoderm.Gil_mm9', '11_tor_EpiSC5.Gil_mm9', '14_tor_EpiSC7.Gil_mm9', '8_tor_46C_NPC.Gil_mm9', '9_tor_D3_EBM6.Gil_mm9',
               '23_tor_D3_NPC.Gil_mm9', '20_tor_MEL.Gil_mm9', '13_tor_CH12.Gil_mm9', '17_tor_L1210.Gil_mm9', '25_tor_CD4.Gil_mm9', '8_tor_Act-24h.all.CD8_nov16',
               '12_tor_mammary.Gil_mm9', '16_tor_1X10_5_all.small_cell_number', '10_tor_myo.Gil_mm9', '26_tor_Rif2_MEF.Gil_mm9', '27_tor_Rif1_MEF.Gil_mm9', '22_tor_MEF_M.Gil_mm9', '24_tor_MEF_F.Gil_mm9']
cluster_order=['22_tor_MEF_M.Gil_mm9', '14_tor_1X10_3_all.small_cell_number', '16_tor_1X10_5_all.small_cell_number', '15_tor_1X10_4_all.small_cell_number', '12_tor_Act-48h.all.CD8_nov16', '25_tor_CD4.Gil_mm9',
               '8_tor_Act-24h.all.CD8_nov16', '9_tor_PGC-all.PGC', '9_tor_Sperm.all.PGC_nov16']
def find_nearest(group, match, groupname):
    match = match[match[groupname] == group.name]
    nbrs = NearestNeighbors(1).fit(match['4_mid'].values[:, None])
    dist, ind = nbrs.kneighbors(group['4_mid'].values[:, None])

    group['4_mid_closest'] = group['4_mid']
    group['4_mid'] = match['4_mid'].values[ind.ravel()]
    return group

def distribution_correlation_gc(data_file=ToR_functions.gc_data_mm9,method='spearman',start=0,exps=['PGC', 'PGC_nov16', 'CD8_nov16', 'small_cell_number','Gil_mm9'],filter_data=gil_rt_switching_labels,data_col='',win_size='100',rel_cols=None):
    """Usually we do correlation which is generally not ideal
    Here I will try to do what Amnon did in his article of tor and mutations which is to show that the distribution
    or snp-tors is enriched with late tor as opposed to the general tor distribution"""
    frame=get_merged_data(exps,win_size=win_size)
    frame=frame.dropna(axis=0,how='any')
    data=pd.read_csv(data_file,sep='\t',na_values='nan',index_col=False)
    print(data.head())
    if len(data)>100000:
        frame=frame[frame['#1_chr']=='chr1']
        data=data[data['d_chro']=='chr1']
       
    print(len(data))
    print(len(frame))
    f=pybedtools.bedtool.BedTool.from_dataframe(frame)
    d=pybedtools.bedtool.BedTool.from_dataframe(data)
    d=d.intersect(f,wb=True)
    data_names=data.columns.values.tolist()
    print(data.columns.values.tolist())
    rel=input('choose the column you care about')
    cutoff=float(input('choose a cutoff'))
    frame_names=frame.columns.values.tolist()
    data=d.to_dataframe(names=data_names+frame_names)
    data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/tmp.bed',sep="\t",index=False)
    data=data[data[rel]>cutoff]
    data=data[rel_cols]
    if not rel_cols:
        rel_cols=cluster_order
        rel_cols=[a for a in frame.columns.values.tolist() if 'tor' in a]
    frame=frame[rel_cols]
    lab=input('label your feature ex: snps,peaks')
    for col in rel_cols:
        print(col)
        print(ks_2samp(frame[col].values.tolist(),data[col].values.tolist()))
        y,binEdges=np.histogram(frame[col].values.tolist(),bins=30,normed=True)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,label='ToR distrib')
        y,binEdges=np.histogram(data[col].values.tolist(),bins=30,normed=True)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(bincenters,y,label=lab+' distrib')
        plt.legend()
        plt.title(col)
        plt.ylim(0,1.7)
        plt.show()



def distribution_correlation(data_file=ToR_functions.snp_list,method='spearman',start=0,
                             exps=['PGC', 'PGC_nov16', 'CD8_nov16', 'small_cell_number','Gil_mm9'],
                             filter_data=gil_rt_switching_labels,lab=None,win_size='100',rel_cols=''):
    """Usually we do correlation which is generally not ideal
    Here I will try to do what Amnon did in his article of tor and mutations which is to show that the distribution
    or snp-tors is enriched with late tor as opposed to the general tor distribution"""
    frame=get_merged_data(exps,win_size=win_size)
    frame=frame.dropna(axis=0,how='any')
    data=pd.read_csv(data_file,sep='\t',na_values='nan',index_col=False,names=['d_chro','d_start','d_end'])
    print(data.head())
    if filter_data:
        print(len(frame))
        gil_data=pd.read_csv(filter_data,sep='\t')
        frame=pd.merge(frame,gil_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False)

    if len(data)>100000:
        frame=frame[frame['#1_chr']=='chr1']
        data=data[data['d_chro']=='chr1']
       
    print(len(data))
    print(len(frame))
    print(frame.head())
    f=pybedtools.bedtool.BedTool.from_dataframe(frame)
    d=pybedtools.bedtool.BedTool.from_dataframe(data)
    d=d.intersect(f,wb=True)
    data_names=['d_chro','d_start','d_end']
    frame_names=frame.columns.values.tolist()
    data=d.to_dataframe(names=data_names+frame_names)
    data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/tmp.bed',sep="\t",index=False)
    data=data[frame_names]
    if rel_cols=='':
        rel_cols=cluster_order
        rel_cols=[a for a in frame.columns.values.tolist() if 'tor' in a]
    data=data[rel_cols]
    frame=frame[rel_cols]
    if not lab:
        lab=input('label your feature ex: snps,peaks')
    for col in rel_cols:
        print(col)
        print(ks_2samp(frame[col].values.tolist(),data[col].values.tolist()))
##        y,binEdges=np.histogram(frame[col].values.tolist(),bins=30,normed=True)
##        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
##        plt.plot(bincenters,y,label='real_distrib')
##        y,binEdges=np.histogram(data[col].values.tolist(),bins=30,normed=True)
##        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
##        plt.plot(bincenters,y,label=lab+' distrib')
##        plt.legend()
##        plt.title(col)
##        plt.show()
        sns.distplot(frame[col].values.tolist(),label='all RT bins')
        sns.distplot(data[col].values.tolist(),label=lab)
        plt.title(ToR_functions.rename_frame([col])[0]+' RT distribution')
        plt.legend()
        plt.show()


def distribution_boxplot(data_file=ToR_functions.snp_list,method='spearman',start=0,exps=['PGC', 'PGC_nov16', 'CD8_nov16', 'small_cell_number','Gil_mm9'],filter_data=gil_rt_switching_labels,data_col='',win_size='100',rel_cols=''):
    """Usually we do correlation which is generally not ideal
    boxplot of distributions of dif columns for a certain feature aka snp sites or chipseq peaks for rami"""
    frame=get_merged_data(exps,win_size=win_size)
    frame=frame.dropna(axis=0,how='any')
    data=pd.read_csv(data_file,sep='\t',na_values='nan',index_col=False,names=['d_chro','d_start','d_end'])
    print(data.head())
    if len(data)>100000:
        frame=frame[frame['#1_chr']=='chr1']
        data=data[data['d_chro']=='chr1']
       
    print(len(data))
    print(len(frame))
    f=pybedtools.bedtool.BedTool.from_dataframe(frame)
    d=pybedtools.bedtool.BedTool.from_dataframe(data)
    d=d.intersect(f,wb=True)
    data_names=['d_chro','d_start','d_end']
    frame_names=frame.columns.values.tolist()
    data=d.to_dataframe(names=data_names+frame_names)
    data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/tmp.bed',sep="\t",index=False)
    data=data[frame_names]
    if rel_cols=='':
        rel_cols=cluster_order
        rel_cols=[a for a in frame.columns.values.tolist() if 'tor' in a]
    data=data[rel_cols]
    frame=frame[rel_cols]
    lab=input('label your feature ex: snps,peaks')
    data.boxplot(rel_cols)
    plt.legend()
    plt.show()
    frame.boxplot(rel_cols)
    plt.legend()
    plt.show()


def correlations_byfeature(data_file=ToR_functions.isr_data,method='spearman',start=0,exps=['PGC', 'PGC_nov16', 'CD8_nov16', 'small_cell_number','Gil_mm9'],filter_data=gil_rt_switching_labels,data_col='',win_size='100'):
    '''In the original correlations function a feature value is calculated for each tor window
    hear, we will take a dataframe of features and calculate each features tor'''
    frame=get_merged_data(exps,win_size=win_size)
 #   frame=frame.dropna(axis=0,how='any')
    data=pd.read_csv(data_file,sep='\t',na_values='nan',index_col=False)
    print(len(frame))

    #merge on chro name and closest coordinate
    data['4_mid']=(data['2_start']+data['3_end'])/2
    data_modified = data.groupby('#1_chr').apply(find_nearest, frame, '#1_chr')
    data_merge_cols=['#1_chr','4_mid']
    frame=pd.merge(data_modified,frame,how='inner',left_on=data_merge_cols, right_on=merge_cols,sort=False)

    #if filtering perform same merge
    if filter_data:
        print(len(frame))
        gil_data=pd.read_csv(filter_data,sep='\t')
        frame=pd.merge(frame,gil_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False)
    
    print(len(frame))
    print([a for a in frame.columns if 'tor' not in a])
    if data_col=='':
        data_col=input('which feature  column do you want to correlate?\n')
    print([a for a in frame.columns if 'tor' in a])
#    tor_cols=input('which tor columns do you want to correlate?\n')
    tor_cols=cluster_order
    corrs=[]
    print(len(frame))
    
    frame.columns=ToR_functions.rename_frame(frame.columns.values.tolist())
    tor_cols=ToR_functions.rename_frame(tor_cols)

    
    for col in tor_cols:
        print(col)
        corrs.append(frame[col].corr(frame[data_col],method=method))
    print(tor_cols)
    print(corrs)
    plt.bar(range(len(corrs)),corrs)
    plt.xticks([a+.5 for a in range(len(corrs))],tor_cols,rotation='vertical')
#    plt.ylim(start,1)
    plt.show()

def scatter_dif(dif_file,exps=['CD8_nov16','small_cell_number']):
    '''This makes a scatter of two ToR groups based 
    it colors by differential or not with a line connecting sequential dots'''

    data=pd.read_csv(dif_file,sep='\t')
    print(data.columns.values.tolist())
    cols=list(input('choose until i automate thjis\n'))
    data.columns=ToR_functions.rename_frame(data.columns.values.tolist())
    cols=ToR_functions.rename_frame(cols)

    plt.scatter(data[cols[0]][data['dif_final']==False],data[cols[1]][data['dif_final']==False],color='blue',label='same')
    plt.scatter(data[cols[0]][(data['dif_final']==False) & (data['diff']==True) ],data[cols[1]][(data['dif_final']==False) & (data['diff']==True)],color='green',label='Too short to be diff')
#    plt.scatter(data[cols[0]][(data['FDR_outer']==True) & (data['diff']==False) ],data[cols[1]][(data['FDR_outer']==True) & (data['diff']==False)],color='yellow',label='high inner noise')    
    plt.scatter(data[cols[0]][data['dif_final']==True],data[cols[1]][data['dif_final']==True],color='red',label='differential')
    
    ax=plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
##    cut_list=data[data['dif_final']==True].index.tolist()
##    groups=[]
##    for k, g in groupby(enumerate(cut_list), lambda (i,x):i-x):
##            group = map(itemgetter(1), g)
##            groups.append(group)
##    for group in groups:    
##        plt.plot(data[cols[0]].loc[group].values.tolist(),data[cols[1]].loc[group].values.tolist(),color='green')

    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.show()



data_files=[ToR_functions.gc_data_mm9,ToR_functions.geneid_data_mm9,ToR_functions.sine_data_mm9,
            ToR_functions.line_data_mm9,ToR_functions.chen_bed_extended,ToR_functions.PGC_chromatin,
            ToR_functions.hapmap_nogenes,ToR_functions.hapmap_nogenesmask,ToR_functions.snp_nogenes,
            ToR_functions.snp_nogenesmask,ToR_functions.hotspot_itamar]
#ToR_functions.snp_data_freq_mm9
def heatmap(data_files=data_files,exps=['PGC', 'PGC_nov16', 'CD8_nov16', 'small_cell_number','Gil_mm9'],filter_data=gil_rt_switching_labels,data_cols=[],tors=None):
    """ this makes a heatmap of tor profiles vs data profiles"""
    frame=get_merged_data(exps)
    frame=frame.dropna(axis=0,how='any')
    for data_file in data_files:
        data=pd.read_csv(data_file,sep='\t',na_values='nan')
        print(len(frame))
        frame=pd.merge(frame,data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False)
    if filter_data:
        print(len(frame))
        gil_data=pd.read_csv(filter_data,sep='\t')
        frame=pd.merge(frame,gil_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False) 
    if len(data_cols)==0:
        print([a for a in frame.columns if 'tor' not in a])
        data_cols=list(input('which columns do you want to correlate?\n'))
    frame=frame[[a for a in frame.columns if 'tor' in a]+[a for a in frame.columns if a in data_cols]]
    if tors:
        frame=frame[[a for a in frame.columns if a in tors]+[a for a in frame.columns if a in data_cols]]
    frame.columns=ToR_functions.rename_frame(frame.columns.values.tolist())
        

    method='spearman'
    correlation_matrix=frame.corr(method=method)
    ntors=len(frame.columns)-len(data_cols)
    correlation_matrix=correlation_matrix.ix[:ntors,ntors:]
    print(correlation_matrix)
    correlation_matrix.to_csv('/mnt/lustre/hms-01/fs01/britnyb/temp_heat.txt',sep='\t',na_rep='nan')
    scale=False
    if scale:
        plt.pcolor(correlation_matrix,cmap='bwr',vmin=scale)
    else:
    #    plt.pcolor(correlation_matrix,cmap='bwr')
        plt.pcolor(correlation_matrix,cmap='bwr')
    plt.yticks(np.arange(0.5, len(correlation_matrix.index), 1), correlation_matrix.index)
    plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns,rotation='vertical')        
    plt.colorbar()
#    plt.savefig(DIR+'/heatmap_correlations.jpg')
    plt.show()

    
def correlations_barplot(data_file=ToR_functions.gc_data_mm9,method='spearman',start=0,exps=['PGC', 'PGC_nov16', 'CD8_nov16', 'small_cell_number','Gil_mm9'],
                         filter_data=gil_rt_switching_labels,data_col='',rel_cols=cluster_order,boot=False,fisher=False,remove_outlier=False,chro=None,ylim=None,frame=[],save_figure_data=[]):
    fig_folder='/mnt/lustre/hms-01/fs01/britnyb/lab_files/figures/gc_tor_pics/'
    if len(frame)>1:
        frame=frame
    else:
        frame=get_merged_data(exps,clean=False)
        frame=frame.dropna(axis=0,how='any')
        data=pd.read_csv(data_file,sep='\t',na_values='nan')
        print(len(frame))
        frame=pd.merge(frame,data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False)
        if chro:
            frame=frame[frame["#1_chr"]=='chr'+chro]
        if filter_data:
            gil_data=pd.read_csv(filter_data,sep='\t')
            gil_data=gil_data[gil_data.columns[:4]] ##need to remove tor columns but they are not named correctly so can't use 'tor' to remove them  
            gil_data.columns=frame.columns[:4]
            frame=pd.merge(frame,gil_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False) 
        print(len(frame))
    #    print([a for a in frame.columns if 'tor' not in a])
        if data_col=='':
            data_col=input('which column do you want to correlate?\n')

    if remove_outlier:
        print(len(frame))
        frame=frame[abs(frame[data_col]-frame[data_col].mean())<2*frame[data_col].std()]
        print(len(frame))
        print('outliers_out')

    corrs=[]
    print(len(frame))
#    print([a for a in frame.columns if 'tor' in a])
    frame.columns=ToR_functions.rename_frame(frame.columns.values.tolist())
    rel_cols=ToR_functions.rename_frame(rel_cols)
    data_col=ToR_functions.rename_frame([data_col])[0]

    yerrs=[]
    for col in rel_cols:
        print(col)
#        print (col+": "+str(frame[col].corr(frame[data_col],method='spearman')))
        if boot:
                ## bootstrap each one and then do t test between all their corr coefficients
            cor=frame[col].corr(frame[data_col],method=method)
            corrs.append(cor)
            CI=bootstrap.ci(frame[[data_col,col]],statfunction=spearmanr,n_samples=1000,method='pi')[:,0]
            yerrs.append(CI)
            print(CI,cor)

        elif fisher:
            ###stopped working on this because i think it is irrelevant
            cor=frame[col].corr(frame[data_col],method=method)
            corrs.append(cor)
            difs=[]
            arr=frame[[col,'SSC combined']].as_matrix()
            data=frame[data_col].values.tolist()
            for i in range(10):
                ##this method took too long
    #            frame[rel_cols]=frame[rel_cols].apply(np.random.permutation,axis=1)
    #            a=frame[pair[0]].corr(frame[data_col],method=method)
    #            b=frame[pair[1]].corr(frame[data_col],method=method)
                shuf=[np.random.permutation(x) for x in arr]
                a=spearmanr(data,[a for a,b in shuf])[0]
                b=spearmanr(data,[b for a,b in shuf])[0]
                difs.append(abs(a-b))
            conf_int=norm.interval(.95,loc=np.mean(difs),scale=np.std(difs)/np.sqrt(len(difs)))
            print(conf_int,cor,col,difs)


        else:
            corrs.append(frame[col].corr(frame[data_col],method=method))
#        plt.scatter(frame[col],frame[data_col])
#        plt.title('spearman: '+str(frame[col].corr(frame[data_col],method='spearman')))
#        plt.savefig(fig_folder+col+'.jpg')
#        plt.clf()
    print(rel_cols)
    print(corrs)
    print(filter_data)
    if boot:
        yerrs=[[abs(number-cor) for number in group] for group,cor in zip(yerrs,corrs)]
        plt.bar(range(len(corrs)),corrs,yerr=np.transpose(yerrs).tolist(),color='steelblue')
    else:
        plt.bar(range(len(corrs)),corrs,color='steelblue')
    plt.xticks([a for a in range(len(corrs))],rel_cols,rotation='vertical')
#    pv=round(mannwhitneyu(corrs[-2:],corrs[:-2])[1],2)
#    plt.ylim(start,1)
    plt.ylabel('Spearman Correlation')
    if ylim:
        plt.ylim(0,ylim)
    plt.title(data_col)
    ylim=plt.ylim()[1]
    if len(save_figure_data)>0:
        print('saving')
        fig_name,fig_fold,inches=save_figure_data
        ToR_functions.save_figure(fig_name,fig_fold,inches)
    plt.show()

    return(ylim)


        
def bin_feature(data_file,filter_data,tor_col,exps,data_col,tor_cols=None,nbins=5,colors=[],frame=[],save_figure_data=[]):
    plt.rc('axes', color_cycle=ToR_functions.color_cycler(colors))
    if len(frame)>1:
        frame=frame.dropna()
    else:
        frame=get_merged_data(exps,[data_file])
        frame=frame.dropna(axis=0,how='any')

        if filter_data:
            print(len(frame))
            gil_data=pd.read_csv(filter_data,sep='\t')
            frame=pd.merge(frame,gil_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False) 
        print(len(frame))
        print([a for a in frame.columns if 'tor' not in a])
        if data_col=='':
            data_col=input('which column do you want to correlate?\n')

    data_col=ToR_functions.rename_frame([data_col])[0]
    if tor_cols:
        tor_cols=ToR_functions.rename_frame(tor_cols)
        frame.columns=ToR_functions.rename_frame(frame.columns.values.tolist())
        bins=nbins
        frame_empty=pd.DataFrame(data=None, columns=frame.columns.values.tolist()+['RT bins','tor_type'])
        new_lab=[]
        for tor_col in tor_cols:
            temp=frame
            temp['RT bins']=pd.qcut(frame[tor_col].values,bins,labels=list(range(1,bins+1)))
            temp['tor_type']=tor_col
            gradient, intercept, r_value, p_value, std_err = linregress(temp[data_col].values.tolist(),temp['RT bins'].values.tolist())
            print(spearmanr(temp[data_col].values.tolist(),temp['RT bins'].values.tolist()))
            print(spearmanr(temp[data_col].values.tolist(),temp[tor_col].values.tolist()))
            print(tor_col,str(p_value),str(r_value**2),str(r_value))
            frame_empty=pd.concat([frame_empty,temp])
            #new_lab.append(tor_col+"- "+r"${R}^2$"+"="+str(round(r_value**2,3))+"  p.v.="+'%.2E' % Decimal(p_value))
            if p_value==0:
                new_lab.append(tor_col+"- "+"R"+"="+str(round(r_value,3)))
            else:
                new_lab.append(tor_col+"- "+"R"+"="+str(round(r_value,3))+"  p.v.="+'%.2E' % Decimal(p_value))
        print('this function is not ready') #split each dataframe to be the rank and matching tor and then melt each one and concat at the end
        frame_empty.to_csv('/mnt/lustre/hms-01/fs01/britnyb/temp.bed')
        frame_empty=pd.read_csv('/mnt/lustre/hms-01/fs01/britnyb/temp.bed')
        sns.boxplot(x="RT bins", y=data_col, hue='tor_type', data=frame_empty,showfliers=False)
        ax=plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Set legend #
        handles, labels = ax.get_legend_handles_labels()
        lgd=ax.legend(handles=handles, labels=new_lab, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(ToR_functions.rt_bins_label)
        if len(save_figure_data)>0:
            print('saving')
            fig_name,fig_fold,inches=save_figure_data
            ToR_functions.save_figure(fig_name,fig_fold,inches)
        plt.show()
##        sns.violinplot(x="bins", y=data_col, hue='tor_type', data=frame_empty,split=True)
##        ax=plt.gca()
##        box = ax.get_position()
##        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
##        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
##        # Set legend #
##        handles, labels = ax.get_legend_handles_labels()
##        ax.legend(handles=handles, labels=new_lab, loc='center left', bbox_to_anchor=(1, 0.5))
##        plt.show()
    else:
        tor_col=ToR_functions.rename_frame([tor_col])[0]
        frame.columns=ToR_functions.rename_frame(frame.columns.values.tolist())

        bins=5
        frame['RT bins']=pd.qcut(frame[tor_col].values,bins,labels=list(range(1,bins+1)))
        gradient, intercept, r_value, p_value, std_err = linregress(frame[data_col].values.tolist(),frame['bins'].values.tolist())
        print(tor_col,str(p_value),str(r_value**2),str(r_value))
        frame.boxplot(data_col,by='RT bins',sym='')
        plt.xlabel('replication timing bins')
        plt.ylabel(data_col)
        plt.title(data_col+' binned by replication timing')
        plt.suptitle("")
        if len(save_figure_data)>0:
            print('saving')
            fig_name,fig_fold,inches=save_figure_data
            ToR_functions.save_figure(fig_name,fig_fold,inches)
        plt.show()

def correlations_scatter(data_file=ToR_functions.gc_data_mm9,method='spearman',start=0,
                         exps=['PGC', 'CD8', 'sperm_L1210', 'smc1_sep16', 'smc1_dec15', 'small_cell_number', 'pre-b_clone3'],
                         filter_data=gil_rt_switching_labels,data_col='',rel_columns=[]):
    frame=get_merged_data(exps)
    frame=frame.dropna(axis=0,how='any')

    data=pd.read_csv(data_file,sep='\t',na_values='nan')
    print(len(frame))
    frame=pd.merge(frame,data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False)
    if filter_data:
        print(len(frame))
        gil_data=pd.read_csv(filter_data,sep='\t')
        frame=pd.merge(frame,gil_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False) 
    print(len(frame))
    print([a for a in frame.columns if 'tor' not in a])
    if data_col=='':
        data_col=input('which column do you want to correlate?\n')
    corrs=[]
    print(len(frame))

    if len(rel_columns)==0:
        rel_columns=[a for a in frame.columns if '_tor_' in a]
    for col in rel_columns:
        print(col)
        corrs.append(frame[col].corr(frame[data_col],method=method))
        plt.scatter(frame[col],frame[data_col])
        plt.xlabel(col)
        plt.ylabel(data_col)
        plt.title('spearman: '+str(frame[col].corr(frame[data_col],method='spearman')))
        plt.show()

def multiple_likelihoods(chro,samples,colors,project_names,difs,win_size='100',SIZE=10,save_figure_data=[],zoom=[]):
    plt.rc('axes', color_cycle=ToR_functions.color_cycler(colors))
    plt.rc('legend', fontsize=12)  # legend fontsize
    plt.rc('ytick', labelsize=12)  # legend fontsize
    plt.rc('ytick', labelsize=12)  # legend fontsize
    plt.rc('axes', labelsize=12)  # legend fontsize
    plt.rc('axes', titlesize=12)  # legend fontsize
    rel_file=ToR_functions.multi_exp_file%{'name':project_names[0],'win_size':win_size}
    our_data=pd.read_csv(rel_file,sep='\t',na_values='nan')
    counter=1
    our_data.columns=[a+ToR_functions.sep+project_names[0] if a not in ToR_functions.joining_cols else a for a in our_data.columns]

    for other_project_name in project_names[1:]:
        other_file=ToR_functions.multi_exp_file%{'name':other_project_name,'win_size':win_size}
        other_data=pd.read_csv(other_file,sep='\t',na_values='nan')
        other_data.columns=[a+ToR_functions.sep+other_project_name if a not in ToR_functions.joining_cols else a for a in other_data.columns]
        our_data=pd.merge(our_data,other_data,how='inner',left_on=["#1_chr","4_mid"], right_on=["#1_chr","4_mid"],sort=False)
    our_data=our_data[our_data['#1_chr']=='chr'+chro]
    our_data.columns=ToR_functions.rename_frame(our_data.columns.values.tolist())
    samples=ToR_functions.rename_frame(samples)

    #plot dif region sizes
    bars_a=[]
    bars_b=[]
    names=[]
    color_dict = {'SSC':'blue', 'PGC':'lightblue','MEF':'red'}
    a_early=[]
    b_early=[]
    for dif in difs:
        all_data=(ToR_functions.diff_regions_file_delt%{'sample1':dif[0],'sample2':dif[1]}).replace(' ','_')
        frame=pd.read_csv(all_data,sep='\t',na_values='nan')
        bars_a.append(len(frame)/10)
        a_early.append(dif[0].split(' ')[0])
        all_data=(ToR_functions.diff_regions_file_delt%{'sample1':dif[1],'sample2':dif[0]}).replace(' ','_')
        frame=pd.read_csv(all_data,sep='\t',na_values='nan')
        bars_b.append(len(frame)/10)
        b_early.append(dif[1].split(' ')[0])
        names.append('%s vs %s'%(dif[0].split(' ')[0],dif[1].split(' ')[0]))
    bars_b=[sum(x) for x in zip(bars_a, bars_b)]
    plt.bar(range(len(difs)),bars_b,color=[color_dict[n] for n in b_early])
    plt.bar(range(len(difs)),bars_a,color=[color_dict[n] for n in a_early])
    plt.xticks(range(len(difs)),names)
    plt.legend()
    plt.ylabel('Megabase differential RT',size=12)
    print(bars_a,bars_b,b_early,a_early)
    plt.show()

    ##plot dif regions
    fig,axes=plt.subplots(len(difs)+1,1,sharex=True,gridspec_kw = {'height_ratios':[3, 1,1,1]})
    print(samples,our_data.columns)
    for sample in samples:
        print(sample)
        axes[0].plot(our_data['4_mid'],our_data[sample],label=sample)
        axes[0].set_ylabel(ToR_functions.RT_label.replace('Replication Timing','RT'),size=12)
        lgd=axes[0].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    counter=1
    for dif in difs:
        all_data=(ToR_functions.diff_regions_file_delt%{'sample1':dif[0],'sample2':dif[1]+'_everything'}).replace(' ','_')
        frame=pd.read_csv(all_data,sep='\t',na_values='nan')
        frame=frame[frame['#1_chr']=='chr'+chro]
        axes[counter].plot(frame['4_mid'],frame['likelihood_test_FDR'],label='Adjusted p.v.',color='green')
#        axes[counter].fill_between(frame['4_mid'],frame['likelihood_test_FDR'],where=frame['dif'],alpha=.5,color='red')
        title='D.R. - %s vs. %s'%(dif[0].split(' ')[0],dif[1].split(' ')[0])
#        axes[counter].fill_between(frame['4_mid'],frame['likelihood_test_FDR'],where=frame['dif'],alpha=.5,color='green',label=title)
        dif_list=pybedtools.bedtool.BedTool.from_dataframe(frame[frame['dif']==True]).merge()
        i=0
        for pair in dif_list:
            axes[counter].axvspan(int(pair[1]),int(pair[2]),color='green',alpha=.3,label="_"*i+title)
            i+=1
        axes[counter].set_ylabel('-log',size=12)
        axes[counter].set_ylim(0,45)
 #       axes[counter].legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        axes[counter].legend(loc='center left')
        counter+=1
    plt.xlabel('Location on Chromosome %s'%chro,labelpad=20)
    plt.subplots_adjust(hspace=0.001)
    if zoom:
        plt.xlim(zoom)
    mkfunc=lambda x,pos: '%dM' % (x * 1e-6) 
    mkformatter=mpl.ticker.FuncFormatter(mkfunc)
    axes[counter-1].xaxis.set_major_formatter(mkformatter)
    if len(save_figure_data)>0:
        print('saving')
        fig_name,fig_fold,inches=save_figure_data
        ToR_functions.save_figure(fig_name,fig_fold,inches)

    plt.show()
    plt.rc('ytick', labelsize=12)  # legend fontsize

    
def boxplot(data_file=ToR_functions.gc_data_mm9,
            exps=['PGC', 'CD8', 'sperm_L1210', 'smc1_sep16', 'smc1_sep14', 'smc1_dec15', 'small_cell_number', 'pre-b_clone3'],
            testing=False,filter_data='/mnt/lustre/hms-01/fs01/britnyb/lab_files/ToR_data/diff_regions/9_tor_Sperm.all_16_tor_1X10_5_all_diff.bed',
            data_col='',rel_columns=[],colors=[],distrib=False,frame=[],save_figure_data=[],violin=False,error_start_height=[]):
    plt.rc('axes', color_cycle=ToR_functions.color_cycler(colors))
    if len(frame)>1:
        frame=frame.dropna(axis=0,how='any')
    else:
        frame=get_merged_data(exps,[data_file])
        frame=frame.dropna(axis=0,how='any')
        if len(rel_columns)==0:
            print([a for a in frame.columns.tolist() if '_tor_' in a])
            rel_columns=list(input('how many of these do you actually want?\n'))
        frame=frame[[data_col]+rel_columns+['#1_chr','4_mid']]
        if data_col=='':
            print(data.columns)
            data_col=input('which column do you want to correlate?\n')

        if filter_data:
            print(len(frame))
            gil_data=pd.read_csv(filter_data,sep='\t')
            gil_data=gil_data[merge_cols]
            frame=pd.merge(frame,gil_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False) 
            print(len(frame))
    if testing:
        return(frame)

    lowest=frame[data_col].values.mean()
    highest=frame[data_col].values.mean()
    mini=lowest-lowest*2
    maxi=highest+highest*2
    
    frame.columns=ToR_functions.rename_frame(frame.columns.values.tolist())
    rel_columns=ToR_functions.rename_frame(rel_columns)
    data_col=ToR_functions.rename_frame([data_col])[0]

    frame['is_early']=np.where(frame[rel_columns[0]]>frame[rel_columns[1]],rel_columns[0],rel_columns[1])
    print(str(frame[data_col][frame['is_early']==rel_columns[0]].mean())+' mean at '+str(rel_columns[0]))
    print(str(frame[data_col][frame['is_early']==rel_columns[1]].mean())+' mean at '+str(rel_columns[1]))
    print(str(len(frame[data_col][frame['is_early']==rel_columns[0]]))+' len at '+str(rel_columns[0]))
    print(str(len(frame[data_col][frame['is_early']==rel_columns[1]]))+' len at '+str(rel_columns[1]))
    sig=ttest_ind(frame[data_col][frame['is_early']==rel_columns[0]].values.tolist(),frame[data_col][frame['is_early']==rel_columns[1]].values.tolist())
    print(sig)
    if not distrib:
        if len(error_start_height)==0:
            sns.boxplot(x='is_early',y=data_col,hue=None,orient='v',data=frame)
            plt.ylabel(data_col)
            plt.show()
            y,h=input('choose where to start the line (y) and how high it should go')
        else:
            y,h=error_start_height
        frame['is_early']=frame['is_early']+' early'
        order=[a+' early' for a in rel_columns]
        print(order)
        sns.boxplot(x='is_early',y=data_col,hue=None,orient='v',order=order,data=frame,showfliers=False)
        plt.ylabel(data_col)
        plt.xlabel('')
        plt.plot([0,0,1,1],[y,y+h,y+h,y],lw=1.5,c='k')
        text="p.v.="+'%.2E' % Decimal(sig[1])
        plt.text(.5,y+h,text,ha='center',va='bottom',color='k')
        plt.ylim(top=y+3*h)
        if len(save_figure_data)>0:
            print('saving')
            fig_name,fig_fold,inches=save_figure_data
            ToR_functions.save_figure(fig_name,fig_fold,inches)

            plt.show()
        if violin:
            sns.violinplot(x='is_early',y=data_col,hue=None,orient='v',order=order,data=frame,showfliers=False)
            plt.ylabel(data_col)
            plt.xlabel('')
            plt.ylim(top=y+3*h)
            plt.show()
        ##    frame.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/hotspots_dif_%s_%s.bed'%(rel_columns[0],rel_columns[1]),sep='\t',index=False,na_rep='na')
    else:
        a=frame[data_col][frame['is_early']==rel_columns[0]]
        b=frame[data_col][frame['is_early']==rel_columns[1]]
        sns.distplot(a,label=rel_columns[0],norm_hist=True)
        sns.distplot(b,label=rel_columns[1],norm_hist=True)
        plt.ylabel(data_col)
        plt.xlabel('')
        plt.legend()
        plt.show()
        data=np.array(a)
        y,binEdges=np.histogram(data,bins=range(6),normed=True)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(binEdges[1:],y,'-',label=rel_columns[0])
        data=np.array(b)
        y,binEdges=np.histogram(data,bins=range(6),normed=True)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.plot(binEdges[1:],y,'-',label=rel_columns[1])
        plt.legend()
        plt.xlabel(data_col)
        plt.ylabel('percentage')
        plt.show()
def expression_correlation2(data_file=ToR_functions.snp_list,method='spearman',start=0,exps=['PGC', 'PGC_nov16', 'CD8_nov16', 'small_cell_number','Gil_mm9'],filter_data=gil_rt_switching_labels,data_col='',win_size='100'):
    """this is a trial to make this more accessible for other things like expression
    """
    expr='/mnt/lustre/hms-01/fs01/britnyb/lab_files/genome_browser_data/mouse_expression/mouse_expression.sorted.bed'
    def myround(x, base=100000):
        return int(base * round(float(x)/base))
    data=pd.read_csv(expr,sep='\t',na_values='nan',index_col=False)
    frame=get_merged_data(exps,win_size=win_size)
    frame=frame.dropna(axis=0,how='any')
    data['start']=data['start'].apply(myround)
    frame=pd.merge(data,frame,left_index=[0,1],right_index=[0,1])
    rel_columns=['CD8+ 48h rep 1','MEF 10^5 rep 1']
    rel_columns=['SSC rep 1','MEF 10^5 rep 1']
    filter_data='/mnt/lustre/hms-01/fs01/britnyb/lab_files/ToR_data/diff_regions/'+rel_columns[0]+'_'+rel_columns[1]+'_diff.bed'
    print(len(frame))
    gil_data=pd.read_csv(filter_data,sep='\t')
    frame=pd.merge(frame,gil_data,how='inner',left_index=[0,1], right_index=[0,1],sort=False)
    col='T-cells_CD8+'
    col="testis"
    print(len(frame))
    frame=frame[frame[col]<=1000]
    frame=frame[frame[col]<=(np.mean(frame[col])+np.std(frame[col])*3)]
    print(len(frame))
    frame['early']=np.where(frame[rel_columns[0]]>frame[rel_columns[1]],rel_columns[0],rel_columns[1])
    ttest=ttest_ind(frame[col][frame['early']==rel_columns[0]].values.tolist(),frame[col][frame['early']==rel_columns[1]].values.tolist())
    print(ttest[1])
    sns.boxplot(y=col,orient='v',x='early',data=frame)
    plt.show()
    sig=ttest_ind(frame[col][frame['early']==rel_columns[0]].values.tolist(),frame[col][frame['early']==rel_columns[1]].values.tolist())
    print(sig)


def expression_correlation(method='single_box',data_file=ToR_functions.expression_data,data_col='',rel_columns=[],exps=['CD8_nov16', 'small_cell_number'],testing=False,
                           filter_data='/mnt/lustre/hms-01/fs01/britnyb/lab_files/ToR_data/diff_regions/16_tor_1X10_5_all_12_tor_Act-48h.all_diff.bed',mini=5,maxi=5000):
    '''taking early regions of each type from differential data compares each early region to its expression '''
    frame=get_merged_data(exps)
    frame=frame.dropna(axis=0,how='any')
    print([a for a in frame.columns.tolist() if '_tor_' in a])
    if len(rel_columns)==0:
        rel_columns=list(input('how many of these do you actually want?\n'))
    frame=frame[rel_columns+['#1_chr','4_mid']]
    data=pd.read_csv(data_file,sep='\t',na_values='nan')
    print(data.columns)
    if data_col=='':
        data_col=list(input('which expressions do you want to correlate?\n'))
    frame=pd.merge(frame,data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False)

    if filter_data:
        print(len(frame))
        print(len(data))
        gil_data=pd.read_csv(filter_data,sep='\t')
        frame=pd.merge(frame,gil_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False) 
        print(len(frame))
    if testing:
        return(frame)
    frame=frame[rel_columns+data_col]
    frame.columns=[a.rsplit('tor_')[-1] for a in frame.columns]
    rel_columns=[a.rsplit('tor_')[-1] for a in rel_columns]

    frame.columns=ToR_functions.rename_frame(frame.columns.values.tolist())
    rel_columns=ToR_functions.rename_frame(rel_columns)



    for col in data_col:
        frame=frame[(frame[col]<maxi) & (frame[col]>mini)]
        ##frame[col]=frame[col].rank()  ##considered ranking expression because what if one is consistently lower, this didn't help
        ##print('ranked expression')

    if method=='percent_over':
        percents=[]
        bins=5
        for col in rel_columns:
            frame[col]=pd.qcut(frame[col].values,bins,labels=list(range(1,bins+1)))
        fig,ax=plt.subplots(2,2)
        countera=0
        counterb=0
        for col in rel_columns:
            for d in data_col:
                cur=ax[countera,counterb]
                sns.boxplot(y=d>np.mean(d),orient='v',x=col,data=frame,ax=cur)
                cur.set_xlabel(col)
                cur.set_ylabel(d)
                counterb+=1
            countera+=1
            counterb=0
        print('no idea how to do this.....')
        plt.show()

    for d in data_col:
        frame[d]=np.log(frame[d])
    
    if method=='corr':
        correlation_matrix=frame.corr()
        correlation_matrix=correlation_matrix[rel_columns]
        correlation_matrix=correlation_matrix.loc[data_col]
        plt.pcolor(correlation_matrix,cmap='YlOrBr')
        plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns,rotation='vertical')
        plt.yticks(np.arange(0.5, len(correlation_matrix.index), 1), correlation_matrix.index)
        plt.colorbar()
        plt.show()
    elif method=='double_box':
        fig,ax=plt.subplots(1,2)
        counter=0
        for col in data_col:
            print(col)
            frame['early']=np.where(frame[rel_columns[0]]>frame[rel_columns[1]],rel_columns[0],rel_columns[1])
            ttest=ttest_ind(frame[col][frame['early']==rel_columns[0]].values.tolist(),frame[col][frame['early']==rel_columns[1]].values.tolist())
            print(ttest[1])
            sns.boxplot(y=col,orient='v',x='early',data=frame,ax=ax[counter])
       #     ax.set_ylim(mini,maxi)
            plt.ylabel(col)
            counter+=1
        plt.show()
    elif method=='single_box':
        col='ratio'
        frame[col]=np.log(frame[data_col[0]]/frame[data_col[1]])
        frame['early']=np.where(frame[rel_columns[0]]>frame[rel_columns[1]],rel_columns[0],rel_columns[1])
        ttest=ttest_ind(frame[col][frame['early']==rel_columns[0]].values.tolist(),frame[col][frame['early']==rel_columns[1]].values.tolist())
        print(ttest[1])

        print(frame.head())
        sns.boxplot(y=col,orient='v',x='early',data=frame)
       #     ax.set_ylim(mini,maxi)
        plt.ylabel(data_col[0]+' exp/'+data_col[1]+' exp')
        plt.show()
        sig=ttest_ind(frame[col][frame['early']==rel_columns[0]].values.tolist(),frame[col][frame['early']==rel_columns[1]].values.tolist())
        print(sig)

    elif method=='scatter':
        fig,ax=plt.subplots(2,2)
        countera=0
        counterb=0
        for col in rel_columns:
            for d in data_col:
                cur=ax[countera,counterb]
                cur.scatter(frame[col],frame[d])
                cur.set_xlabel(col)
                cur.set_ylabel(d)
                cur.set_title('spearman r='+str(spearmanr(frame[col],frame[d])[0]))
                counterb+=1
            countera+=1
            counterb=0
        plt.show()
    elif method=='scatter_bin':
        bins=4
        for col in rel_columns:
            frame[col]=pd.qcut(frame[col].values,bins,labels=list(range(1,bins+1)))
        fig,ax=plt.subplots(2,2)
        countera=0
        counterb=0
        for col in rel_columns:
            for d in data_col:
                cur=ax[countera,counterb]
                sns.boxplot(y=d,orient='v',x=col,data=frame,ax=cur)
                cur.set_xlabel(col)
                cur.set_ylabel(d)
                counterb+=1
            countera+=1
            counterb=0
        plt.show()
    elif method=='top percent':
        p=.2
        percent=int(len(frame)*p)
        for d in data_col:
            print(d)
            frame.sort(d,ascending=False,inplace=True)
            print(frame[d].head())
            top=frame[:percent]
            plt.boxplot([top[rel_columns[0]].values.tolist(),top[rel_columns[1]].values.tolist()])
            plt.xticks([1,2],rel_columns)
            plt.ylabel('ToR')
            plt.title('top %s genes expressed in %s'%(str(p),d))
            plt.show()


        


def temp_exp_analysis(method=1,n_groups=5,data_file=ToR_functions.expression_data,exps=['CD8', 'small_cell_number'],filter_data=None,data_col='',rel_columns=[]):
    '''this will look at expression by grouping the data instead of broad correlations '''
    frame=get_merged_data(exps)
    frame=frame.dropna(axis=0,how='any')
    print([a for a in frame.columns.tolist() if '_tor_' in a])
    if len(rel_columns)==0:
        rel_columns=list(input('how many of these do you actually want?\n'))
    frame=frame[rel_columns+['#1_chr','4_mid']]
    data=pd.read_csv(data_file,sep='\t',na_values='nan',index_col=False)
    print(data.columns)
    if data_col=='':
        data_col=input('which column do you want to correlate?\n')
    if method==2:
        #get only dif from frame
        print(data[data.columns[4:]].std(axis=1))
        data=data[data[data.columns[4:]].max(axis=1)-data[data.columns[4:]].min(axis=1)>1000]

    frame=pd.merge(frame,data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False)

    if filter_data:
        print(len(frame))
        gil_data=pd.read_csv(filter_data,sep='\t')
        frame=pd.merge(frame,gil_data,how='inner',left_on=merge_cols, right_on=merge_cols,sort=False)

        
    frame.columns=ToR_functions.rename_frame(frame.columns.values.tolist())
    rel_columns=ToR_functions.rename_frame(rel_columns)

    
    print(frame.columns)
    tor_cols=rel_columns
    mini=min(frame[tor_cols].min(axis=0).values.tolist())
    maxi=max(frame[tor_cols].max(axis=0).values.tolist())
    print(len(frame))
    fig,axs=plt.subplots(1,len(tor_cols))

    #method 1 or 2
    if method==1 or method==2:
        labels=range(1,n_groups+1)
        xlabel='bins of %s binned from lowest to highest'%data_col
        frame[xlabel] = pd.cut(frame[data_col], n_groups, labels=labels)
        counter=0
        x=xlabel

    #this is the simplest method to just look at the most highly expressed genes...
    if method==4:
        labels=range(1,n_groups+1)
        xlabel='bins of %s binned from lowest to highest'%data_col
        frame[xlabel] = pd.cut(frame[data_col], n_groups, labels=labels)
        high_frame=frame[frame[xlabel]==n_groups]
        data=[high_frame[rel_columns[0]].values.tolist(),high_frame[rel_columns[1]].values.tolist()]
        plt.boxplot(data,labels=rel_columns)
        plt.legend()
        plt.show()
        return()

    
    #method 3
    if method==3:
        data_col=input('which column do you want to use to match %s?\n'%tor_cols[0])
        data_col2=input('which column do you want to use to match %s?\n'%tor_cols[1])
        frame['logr']=np.log(frame[data_col]/frame[data_col2])
        avg_log=abs(frame['logr'].mean())*2
        print([-np.inf,-avg_log,avg_log,np.inf])
        xlabel='log ratio of %s expression over %s expression'%(data_col,data_col2)
        frame[xlabel] = pd.cut(frame['logr'], [-np.inf,-.2,.2,np.inf], labels=['neg','zero','pos'])
        counter=0
        x=xlabel
        
    for col in tor_cols:
        sns.boxplot(y=col,orient='v',x=x,data=frame,ax=axs[counter])
        axs[counter].set_ylim([mini,maxi])
        counter+=1

    plt.show()
            
############### micha analysis #################################

def time_course(bins=5,cutoff=2):
    fig_folder='/mnt/lustre/hms-01/fs01/britnyb/lab_files/'
    a=get_merged_data(exps=['cancer_sep16','cancer_dec15','cancer_jun16'])
    print(a.columns.values.tolist())
    
    a=a[list(input('which samples do you want\n'))]
#    cancer_vectors=a[['#1_chr', '2_start_x', '3_end_x', '4_mid','5_tor_BJE1', '6_tor_BJELB1', '5_tor_BJ3-1-7',
#                      '6_tor_BJ3-1-12', '7_tor_BJ3-1-15', '8_tor_BJ-ELR-1-10', '9_tor_BJ-ELR-1-17', '10_tor_BJ-ELR-1-25',
  #                    '5_tor_FFp15', '6_tor_FFp20', '7_tor_4OHT-10D-1', '8_tor_4OHT-1D-1', '10_tor_BJE-1', '11_tor_ETOH-1','15_tor_BJE-2', '16_tor_ETOH-2']]
    cancer_vectors_no_locs=a[[b for b in a.columns.values.tolist() if 'tor' in b]]
    cancer_vectors_no_locs=cancer_vectors_no_locs.dropna(axis=0,how='any')
#    binned_frame=bin_frame(cancer_vectors_no_locs,bins)
#    binned_frame=binned_frame[(binned_frame.max(axis=1)-binned_frame.min(axis=1))>cutoff]
#    print(len(binned_frame))
#    return(binned_frame)
    cmap=sns.diverging_palette(220, 20, n=7, as_cmap=True)
    a=sns.clustermap(cancer_vectors_no_locs,metric='correlation', method='average',row_cluster=False,cmap=cmap)
    plt.setp(a.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(a.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    plt.savefig(fig_folder+'clustermap_micha'+'.jpg')
    plt.show()
    

def bin_frame(frame,bins=5):
    new=pd.DataFrame()
    for col in frame.columns:
        new[col]=pd.qcut(frame[col].values,bins,labels=list(range(1,bins+1)))
    return(new)

def cnv_prelim(g1_data=[],exps=['cancer_jun16','cancer_sep16','cancer_dec15'],chro='all',plot=False,specific=False):
    if len(g1_data)==0:
        g1_data=one_phase_cor_matrix(phase='G1',plot=False,exps=exps,method='spearman')
    g1_data=g1_data.dropna()
    count_cols=[a for a in g1_data.columns if 'count' in a]
    fix_header=[a for a in g1_data.columns if 'count' not in a]+ToR_functions.rename_frame(count_cols)
    fix_header[0]='#'+fix_header[0]
    #remove outlier bins
##    cutoff=outlier_cutoff
##    for col in [a for a in g1_data.columns if 'count' in a]:
##        g1_data[col]=g1_data[col].clip(upper=g1_data[col].mean()*cutoff)
##    for col in [a for a in s_data.columns if 'count' in a]:
##        s_data[col]=s_data[col].clip(upper=s_data[col].mean()*cutoff)
    g1_data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/cancer_tor/cnv/g1_cancer_coverage.bed',sep='\t',index=False,na_rep='na',header=fix_header)
    #normalize
    for col in [a for a in g1_data.columns if 'count' in a]:
        g1_data[col]=g1_data[col]/(g1_data[col].mean())
    if plot:
        print(g1_data.columns.values.tolist())
        g1_samples=list(input("copy the names of the g1 samples you want to compare separated by a comma\n"))
        if chro=='all':
            for sample in g1_samples:
                plt.plot(range(len(g1_data)),g1_data[sample],label=sample+'_G1')
        else:
            for sample in g1_samples:
                plt.plot(g1_data['start'][g1_data['chro']=='chr'+chro],g1_data[sample][g1_data['chro']=='chr'+chro],label=sample+'_G1')
        plt.legend()
        plt.showow()
    else:
        g1_data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/cancer_tor/cnv/g1_norm_cancer_coverage.bed',sep='\t',index=False,na_rep='na',header=fix_header)
        cutoff=1.5
        for col in count_cols:
            g1_data[col]=np.where(g1_data[col]>cutoff,True,False)
            g1_data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/cancer_tor/cnv/g1_norm_TF_cancer_coverage.bed',sep='\t',index=False,na_rep='na',header=fix_header)
            g1_data=g1_data[g1_data[count_cols].any(axis=1)]
            g1_data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/cancer_tor/cnv/g1_norm_T_only_cancer_coverage.bed',sep='\t',index=False,na_rep='na',header=fix_header)

    if specific:
        g1_data.columns=fix_header
        print(g1_data.columns.values.tolist())
        group1=list(input('first group of columns that you want to compare?  make sure you use a comma.  Type "n" to finish \n'))
        group2=list(input('first group of columns that you want to compare?  make sure you use a comma.  Type "n" to finish \n'))
        #check that within a group values are the same
        g1_data=g1_data[(g1_data[group1].all(axis=1)) | ((~g1_data[group1]).all(axis=1))]
        g1_data=g1_data[(g1_data[group2].all(axis=1)) | ((~g1_data[group2]).all(axis=1))]
        #check for change
        g1_data=g1_data[g1_data[group1[0]]!=g1_data[group2[0]]]

        g1_data.to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/cancer_tor/cnv/%s_%s.bed'%(group1[0],group2[0]),sep='\t',index=False,na_rep='na',header=fix_header)

    
###################################################

def skew_pipe(split=4):
    for exp in ['small_cell_number','smc1_dec15','PGC_nov16']:
        d=my_experiment_data(True,exp)
        for samp in d.combined_names:
            raw_data=pd.read_csv(ToR_functions.ToR_file_patt%{'name':exp,'exp':samp},sep='\t',na_values='nan')
            raw_data=raw_data.dropna()
            print(exp,samp)
            print(skew(raw_data['ratio']))
            quart=len(raw_data)/split
            d=raw_data['ratio'].sort(inplace=False)
            a=d[:quart].mean()
            b=d[-quart:].mean()
            m=d.mean()
            print(m-a,m-b,abs(m-a)-abs(m-b))


    
