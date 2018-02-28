import sys
import os
from time import strftime
import heapq
import matplotlib.pyplot as plt
from matplotlib import colors, cm,rcParams
import numpy as np
from math import log, floor, ceil, sqrt,erf
from scipy.stats.stats import pearsonr, spearmanr, ttest_ind
from scipy.stats import ranksums, mannwhitneyu, ks_2samp, norm, shapiro,combine_pvalues,chi2_contingency,chi2, probplot
from itertools import combinations, groupby,chain,product, permutations
from operator import itemgetter
import re 
from re import split, sub
import pandas as pd
##print("loading matlab")
##import matlab.engine
##eng=matlab.engine.start_matlab()
import seaborn as sns
import pickle
from enum import Enum
from statsmodels.sandbox.stats.multicomp import multipletests
from pyliftover import LiftOver
from random import randint, seed
import operator
import seaborn as sns
from scipy.interpolate import interp1d
#import statsmodels.api as sm
import pybedtools
from matplotlib import ticker

rt_bins_label='Late  '+r'$\leftarrow$'+' RT bins  '+r'$\rightarrow$'+'  Early'
RT_label='Late  '+r'$\leftarrow$'+' Replication Timing  '+r'$\rightarrow$'+'  Early'
RT_label_short='Late  '+r'$\leftarrow$'+' RT  '+r'$\rightarrow$'+'  Early'
dictionary='/cs/icore/britnyb/lab/scripts/tor/tor_dictionary.txt'
output_dir='/mnt/lustre/hms-01/fs01/britnyb/lab_files/ToR_data/'
multi_exp_file=output_dir+"%(name)s/multi_samples_interp%(win_size)skb.bed"
smooth_ToR_file_patt=output_dir+"%(name)s/smooth_tor/%(exp)s.txt"
counts_file=output_dir+"%(name)s/filter/all_counts_%(exp)s.txt"
counts_file_clean=output_dir+"%(name)s/filter/filtered_counts_%(exp)s.txt"
ToR_file_patt=output_dir+"%(name)s/raw_tor/%(exp)s.txt"
gap_file_patt='/mnt/lustre/hms-01/fs01/britnyb/lab_files/gaps/%(genome)s_gaps/chr%(chro)s.%(genome)s.gaps.txt'#%(chro)
gap_file_all='/mnt/lustre/hms-01/fs01/britnyb/lab_files/gaps/%(genome)s_gaps/ucsc_all_%(genome)s'
stat_file=output_dir+"%(name)s/tor_stats.txt"
diff_regions_file=output_dir+"diff_regions/cutoff_%(cutoff)s_%(cutoff2)s_%(sample1)s_%(sample2)s.bed"
diff_regions_fdr_file=output_dir+"%(name)s/diff_regions/n=%(n)s_%(delta)s.bed"

diff_regions_file_delt=output_dir+"diff_regions/%(sample1)s_%(sample2)s.bed"

pic_file=output_dir+"%(name)s/pics/chr%(chro)s_num%(num)s.jpg"
pooled_MEF=output_dir+"MEF_G1/all_MEF_chr%(chro)s.sorted"
features=output_dir+'features/'
gc_data_mm9=features+"GC_mm9.bed"
snp_data_mm9=features+"SNPs_mm9.bed"
recombination=features+'recombination_hotspots.bed'
recombination_raw=features+'recombination_hotspots.raw.bed'
dsbs=features+'dsb.bed'
skew=features+'skew.bed'
snp_list="/mnt/lustre/hms-01/fs01/britnyb/lab_files/genome_browser_data/mm9/single_snp128.bed"
gquadra_list="/mnt/lustre/hms-01/fs01/britnyb/lab_files/genome_browser_data/mm9/gquadraduplex.bed"
gquad_mm9=features+"g_quadra_density.bed"
srna=features+"sRNA.bed"
lncrna=features+"lncRNA.bed"
srna_d=features+"sRNA_dens.bed"
lncrna_d=features+"lncRNA_dens.bed"
MEF_chromatin=features+'GSM2442671_MEF-rep1.100k_windows_c.bed'
PGC_chromatin=features+'GSM2098124_CGH_c.bed'
CH12_chromatin=features+'GSM1014153_ch12_dnase_c.bed'
CD4_chromatin=features+'GSM1014149_dnase_c.bed'
dsb_mean=features+"b6_smag_dsb.mean.bed"
dsb_sum=features+"b6_smag_dsb.sum.bed"
cnv_deletion=features+"cnv_deletions.bed"
cnv_insertion=features+"cnv_insertion.bed"
cnv_mobile_element_insertion=features+"cnv_mobile_element_insertion.bed"
cnv_copy_number_variation=features+"cnv_cnv.bed"
cnv_deletion_no_count=features+"cnv_deletions_no_count.bed"
cnv_insertion_no_count=features+"cnv_insertion_no_count.bed"
cnv_insertion_deletion_no_count=features+"cnv_insertion_deletions_no_count.bed"
cnv_copy_number_variation_no_count=features+"cnv_cnv_no_count.bed"


packaging_cgh=features+'/pgc_expression/GSM2098124_NOMe-seq_Mouse_E13.5_male_PGC_rep1.GCH.bw.mean.bed'
packaging_wcg=features+'/pgc_expression/GSM2098124_NOMe-seq_Mouse_E13.5_male_PGC_rep1.WCG.bw.mean.bed'

#snps 2018
snp_nogenes=features+'snp128_nogenes.bed'
snp_nogenesmask=features+'snp128_nogenes_nomask.bed'
hapmap_nogenes=features+'hapmap_nogenes.bed'
hapmap_nogenesmask=features+'hapmap_nogenes_nomask.bed'
#alternative lists
hap_map="/mnt/lustre/hms-01/fs01/britnyb/lab_files/genome_browser_data/mm9/mousehapmap_perlegen_imputed_full_HC.bed"
hap_map_density=features+'hap_map.bed'
f1_snp_e5=features+"snps.e5.bed"
f1_snp_c2=features+"snps.clone2.bed"
f1_snp_e5_filter=features+"snps.e5.filter.bed"
f1_snp_c2_filter=features+"snps.clone2.filter.bed"
f1_full_e5=features+'e5.full_snp_list.bed'
f1_full_c2=features+'clone2.full_snp_list.bed'
f1_full_e5_filter=features+'e5.full_snp_list.filter.bed'
f1_full_c2_filter=features+'clone2.full_snp_list.filter.bed'
chen_real_snps=features+'chen_snps_real_100kb.bed'
hotspot_itamar=features+"hotspots_count_michael_mm9.bed"
hotspot_itamar_mega=features+"hotspots_count_michael_mm9_mega.bed"

chen_bed_extended="/mnt/lustre/hms-01/fs01/britnyb/lab_files/ToR_data/features/chen_data/100kb_snp_data_mut_rate.bed"
snp_data_freq_mm9=features+"SNPs_mm9_freq.bed"
gene_data_mm9=features+"refgene_mm9.bed"
geneid_data_mm9=features+"geneid_mm9.bed"
expression_data=features+'expression_mm9.bed'
expression_data2=features+'expression_mm9_rbased.bed'
expression_cd8_data=features+'expression_cd8_mm9.bed'
sine_data_mm9=features+"SINE_mm9.bed"
line_data_mm9=features+"LINE_mm9.bed"
isr_data='/mnt/lustre/hms-01/fs01/britnyb/lab_files/ToR_data/features/isr.bed'
joining_cols=["#1_chr","4_mid"]
sep='.'

SIZE=8
plt.rc('font', size=SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title

############################## calculating RT ############################################
def gilbert_windows_genome(exp,name,file_name,step,smoothing,window_size,chroms,genome,sorted_path):
    """ convert gil data to tor dataset """
    file_full=sorted_path+file_name
    with open(file_full,'r') as f1:
        counter=0
        while True:
            row=f1.readline()
            if row.strip()=='':
                continue
            if row[:2]=='ID':
                break
            counter+=1
    skipto=counter
    print(file_name, skipto)
    data=pd.read_csv(file_full,sep='\t',header=skipto)

    smooth_ToR_file=smooth_ToR_file_patt%{'exp':exp,'name':name}
    #CREATE ITERATED DATA
    smooth_frame=pd.DataFrame(columns=['chromosome','loc','norm_tor'])
    
    #remove 100 kb windows with stdev over 1.1    
    min_len=15

    print(strftime("%H:%M:%S")+' starting to interpolate '+exp)
    #create smoothed interpolated data
    for chro in chroms:
        rel_data=data[data['Chromosome']=='chr'+chro]
        if len(rel_data)==0:
            continue
        rel_data['mid']=(rel_data['Start_Position']+rel_data['End_Position'])/2
        #make list of gaps
        gap_file=gap_file_patt %{'genome':genome,'chro':chro}
        gaps_list=[]
        with open(gap_file, 'r') as gaps:
            for line in gaps:
                words=line.split('\t')
                if words[2]=='chromStart':
                    continue
                start=int(words[1])
                end=int(words[2])
                gaps_list.append([start,end])

        #determine total start and end
        start=int(ceil((rel_data['Start_Position']).iloc[1])/float(step))*step
        end=int(ceil((rel_data['End_Position']).iloc[-1])/float(step))*step
        smooth_data=[]
        for gap in gaps_list[1:]:            
            #determine intergap area
            gap_start=int(floor(gap[0]/float(step)))*step
            locs=rel_data['mid'][(rel_data['mid']>=start) & (rel_data['mid']<gap_start)].tolist()
            tors=rel_data['Data_Value'][(rel_data['mid']>=start) & (rel_data['mid']<gap_start)].tolist()
            
            zipped=zip(locs,tors)
            locs=[a for a,b in zipped if not np.isnan(b)]
            tors=[b for a,b in zipped if not np.isnan(b)]

            if len(locs)==0: continue
            interp_range=range(int(ceil(locs[0]/float(step)))*step,int(floor(locs[-1]/float(step)))*step,step)
            #if long enough calculate interpolation
            if len(locs)>min_len and len(interp_range)>1:
                mat_interp_range=matlab.double(interp_range)
                mat_locs=matlab.double(locs)
                mat_tors=matlab.double(tors)
                mat_tors=eng.fnval(mat_interp_range,eng.csaps(mat_locs,mat_tors,smoothing))
                new_tors=[mat_tors[0][i] for i in range(len(mat_tors[0]))]

                #if too short after removal
                if len([a for a in new_tors if not np.isnan(a)])<min_len:
                    new_tors=len(interp_range)*[np.nan]
                smooth_data.extend(zip(interp_range,new_tors))

            #if too short to begin with ignore
            else:
                new_tors=len(interp_range)*[np.nan]
                smooth_data.extend(zip(interp_range,new_tors))
            start=int(ceil(gap[1]/float(step)))*step
            
            #insert gap area as nan
            interp_range=range(gap_start,start,step)
            if interp_range[0] not in [l for l,t in smooth_data]: #make sure gap isnt too small that it repeats itself
                new_tors=len(interp_range)*[np.nan]
                smooth_data.extend(zip(interp_range,new_tors))


        chro_data=pd.DataFrame(smooth_data,columns=['loc','norm_tor'])
        chro_data['chromosome']='chr'+chro
        chro_data=chro_data[['chromosome','loc','norm_tor']]
        smooth_frame=smooth_frame.append(chro_data)

    #add column of log ratios aftr subtracting avg to norm around zero
#    smooth_frame['norm_tor']=(smooth_frame['tor']-smooth_frame['tor'][smooth_frame['chromosome']!='chrX'].mean())/smooth_frame['tor'][smooth_frame['chromosome']!='chrX'].std()
    if not os.path.exists(smooth_ToR_file.rsplit("/",1)[0]):
        os.makedirs(smooth_ToR_file.rsplit("/",1)[0])

    smooth_frame.to_csv(smooth_ToR_file,sep='\t',index=None,na_rep='nan')

def loess_test(x,y,xnew,frac=.3):
    lowess = sm.nonparametric.lowess(y, x, frac=frac)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # run scipy's interpolation. There is also extrapolation I believe
    f = interp1d(lowess_x, lowess_y, bounds_error=False)

    # this this generate y values for our xvalues by our interpolator
    # it will MISS values outsite of the x window (less than 3, greater than 33)
    # There might be a better approach, but you can run a for loop
    #and if the value is out of the range, use f(min(lowess_x)) or f(max(lowess_x))
    ynew = f(xnew)

    plt.plot(x, y, 'o')
    plt.plot(lowess_x, lowess_y, '*')
    plt.plot(xnew, ynew, '-')
    plt.show()
    

#name=experiment name
#exp=sample name
def windows_genome(exp,rep, smoothing,window_size,step,name,chroms,genome,sorted_path,pooled_file=None,sep=None,allele=None,min_len=15,loess=False):
    """ Creates windows to compare readings
    name is the experiment name and exp is the sample"""
    parameters=locals()
    print(exp,rep,allele)
    #INITIALIZE
    read_file_patt=sorted_path
    if rep=='':
        exp_name=exp
    else:
        exp_name=exp+sep+rep
    smooth_ToR_file=smooth_ToR_file_patt%{'exp':exp_name,'name':name}
    data_file=ToR_file_patt%{'exp':exp_name+".data",'name':name}
    ToR_file=ToR_file_patt%{'exp':exp_name,'name':name}
    if allele:
        smooth_ToR_file=smooth_ToR_file_patt%{'exp':exp_name+sep+allele,'name':name}
        data_file=ToR_file_patt%{'exp':exp_name+sep+allele+".data",'name':name}
        ToR_file=ToR_file_patt%{'exp':exp_name+sep+allele,'name':name}

    data_list=[]
    all_win_sizes=[]
    #CREATE RAW DATA
    sample=[0,0,0,0,0]
    all_raw_data=[]
    
    for chro in chroms:
        if pooled_file:
            g_reads_file=pooled_file%{'exp':exp,'chro':chro}
            if chro=='1':
                print(g_reads_file)
        elif allele:
            g_reads_file=read_file_patt%{'rep':rep,'exp':exp,'phase':'G1','chro':chro,'allele':allele}
        else:
            g_reads_file=read_file_patt%{'rep':rep,'exp':exp,'phase':'G1','chro':chro}

        if allele:
            s_reads_file=read_file_patt%{'rep':rep,'exp':exp,'phase':'S','chro':chro,'allele':allele}
        else:
            s_reads_file=read_file_patt%{'rep':rep,'exp':exp,'phase':'S','chro':chro}
        gap_file=gap_file_patt%{'genome':genome,'chro':chro}       
        results=[]

        g_count=0
        s_count=0
        window_start=0
        window_end=0
        old_g_coord=0

        #iterate through sites to create list of lists
        with open(g_reads_file, 'r') as g_file:
            with open(s_reads_file, 'r') as s_file:
                for line in g_file:
                    new_g_coord=int(line.split('\t')[1])
                    g_count+=1
                    
                    #if we reached max count find s_reads also and reset
                    if g_count>=window_size and (new_g_coord!=old_g_coord):
                        #fill in s reads
                        for row in s_file:
                            s_coord=int(row.split('\t')[1])
                            if s_coord<=old_g_coord:
                                s_count+=1
     
                            #if out of window then track and reset
                            else:
                                results.append(['chr'+str(chro), window_start, window_end, g_count, s_count])
                                s_count=1
                                break
                        #reset g count/windows
                        g_count=0
                        window_start=window_end+1
                        
                    window_end=old_g_coord    
                    old_g_coord=new_g_coord
                #find final s count  
                for row in s_file:
                    s_coord=int(row.split('\t')[1])
                    if s_coord<=old_g_coord:
                        s_count+=1
                  
        if g_count != 0:
            results.append(['chr'+str(chro), window_start, window_end, g_count, s_count])    

        #make list of gaps
        gaps_list=[]
        with open(gap_file, 'r') as gaps:
            for line in gaps:
                words=line.split('\t')
                start=int(words[1])
                end=int(words[2])
                gaps_list.append([start,end])

        #remove windows containing gaps
        for index,row in enumerate(results):
            for gap in gaps_list:
                if (gap[0]<=row[2] and gap[0]>=row[1]) or (row[1]<=gap[1] and row[1]>=gap[0]):
                    new_row=row[:3]+[np.nan,np.nan]
                    results[index]=new_row
                    break
        all_raw_data.extend(results)

    raw_data=pd.DataFrame(all_raw_data,columns=['chromosome','start','end','g','s'])

    #NORMALIZE BY FULL GENOME+ ADD COLUMNS
    total_g=raw_data['g'].sum()
    total_s=raw_data['s'].sum()
    factor=float(total_g)/float(total_s)
    raw_data['normal_s']=raw_data['s']*factor
    raw_data['mid']=(raw_data['start']+raw_data['end'])/2
    raw_data['mid']=raw_data['mid'].astype(int)
    raw_data['ratio']=raw_data['normal_s']/raw_data['g']
    raw_data['log_ratio']=np.log10(raw_data['ratio'])
    #CURRENTLY Z SCORE SKIPS LOG
    ###added clause because in BJ-FTB2 std was huge --> need to remove crazy outliers
    std_cur=raw_data['ratio'].std()
    raw_data=raw_data[abs(raw_data['ratio'])<(std_cur*10)]
    raw_data=raw_data[raw_data['s']>1]
    raw_data=raw_data[raw_data['s']<1000]
    raw_data['z_score']=(raw_data['ratio']-raw_data['ratio'].mean())/raw_data['ratio'].std()
    if not os.path.exists(ToR_file.rsplit("/",1)[0]):
        os.makedirs(ToR_file.rsplit("/",1)[0])

    raw_data.to_csv(ToR_file,sep='\t',index=None,na_rep='nan')

    #CREATE ITERATED DATA
    smooth_frame=pd.DataFrame(columns=['chromosome','loc','tor'])
    
    #remove 100 kb windows with stdev over 1.1    

    print(strftime("%H:%M:%S")+' starting to interpolate '+exp+' '+rep)
    #create smoothed interpolated data
    for chro in chroms:
        #make list of gaps
        gap_file=gap_file_patt %{'genome':genome,'chro':chro}
        gaps_list=[]
        with open(gap_file, 'r') as gaps:
            for line in gaps:
                words=line.split('\t')
                if words[2]=='chromStart':
                    continue
                start=int(words[1])
                end=int(words[2])
                gaps_list.append([start,end])

        #determine total start and end
        start=int(ceil((raw_data['start'][raw_data['chromosome']=='chr'+chro]).iloc[1])/float(step))*step
        end=int(ceil((raw_data['end'][raw_data['chromosome']=='chr'+chro]).iloc[-1])/float(step))*step
        smooth_data=[]
        for gap in gaps_list[1:]:            
            #determine intergap area
            gap_start=int(floor(gap[0]/float(step)))*step
            locs=raw_data['mid'][(raw_data['chromosome']=='chr'+chro) & (raw_data['mid']>=start) & (raw_data['mid']<gap_start)].tolist()
            tors=raw_data['z_score'][(raw_data['chromosome']=='chr'+chro) & (raw_data['mid']>=start) & (raw_data['mid']<gap_start)].tolist()
            nonorm_tors=raw_data['ratio'][(raw_data['chromosome']=='chr'+chro) & (raw_data['mid']>=start) & (raw_data['mid']<gap_start)].tolist()
            nonorm_locs=locs
            
            zipped=zip(locs,tors,nonorm_tors)
            locs=[a for a,b,c in zipped if not np.isnan(b)]
            tors=[b for a,b,c in zipped if not np.isnan(b)]
            nonorm_tors=[c for a,b,c in zipped if not np.isnan(b)]

            if len(locs)==0: continue
            interp_range=range(int(ceil(locs[0]/float(step)))*step,int(floor(locs[-1]/float(step)))*step,step)
            #if long enough calculate interpolation
            if len(locs)>min_len and len(interp_range)>1:
                if loess==True:
                    frac=15.0/len(locs) #consider making this dependent on length? o maybe just .01 which worked well in the first region of scn 10x5_all?
                    lowess = sm.nonparametric.lowess(tors, locs, frac=frac)
                    lowess_x = list(zip(*lowess))[0]
                    lowess_y = list(zip(*lowess))[1]
                    f = interp1d(lowess_x, lowess_y, bounds_error=False)
                    new_tors = f(interp_range)

                    lowess = sm.nonparametric.lowess(nonorm_tors, locs, frac=.3)
                    lowess_x = list(zip(*lowess))[0]
                    lowess_y = list(zip(*lowess))[1]
                    f = interp1d(lowess_x, lowess_y, bounds_error=False)
                    new_nonorm_tors=f(interp_range)
                else:
                    mat_interp_range=matlab.double(interp_range)
                    mat_locs=matlab.double(locs)
                    mat_tors=matlab.double(tors)
                    mat_tors=eng.fnval(mat_interp_range,eng.csaps(mat_locs,mat_tors,smoothing))
                    new_tors=[mat_tors[0][i] for i in range(len(mat_tors[0]))]

                    mat_nonorm_tors=matlab.double(nonorm_tors)
                    mat_nonorm_tors=eng.fnval(mat_interp_range,eng.csaps(mat_locs,mat_nonorm_tors,smoothing))
                    new_nonorm_tors=[mat_nonorm_tors[0][i] for i in range(len(mat_nonorm_tors[0]))]

                #if too short after removal
                if len([a for a in new_tors if not np.isnan(a)])<min_len:
                    new_tors=len(interp_range)*[np.nan]
                    new_nonorm_tors=len(interp_range)*[np.nan]
                smooth_data.extend(zip(interp_range,new_tors,new_nonorm_tors))

            #if too short to begin with ignore
            else:
                new_tors=len(interp_range)*[np.nan]
                new_nonorm_tors=len(interp_range)*[np.nan]
                smooth_data.extend(zip(interp_range,new_tors,new_nonorm_tors))
            start=int(ceil(gap[1]/float(step)))*step
            
            #insert gap area as nan
            interp_range=range(gap_start,start,step)
            if interp_range[0] not in [l for l,t,n in smooth_data]: #make sure gap isnt too small that it repeats itself
                new_tors=len(interp_range)*[np.nan]
                new_nonorm_tors=len(interp_range)*[np.nan]
                smooth_data.extend(zip(interp_range,new_tors,new_nonorm_tors))


        chro_data=pd.DataFrame(smooth_data,columns=['loc','norm_tor','tor'])
        chro_data['chromosome']='chr'+chro
        chro_data=chro_data[['chromosome','loc','norm_tor','tor']]
        smooth_frame=smooth_frame.append(chro_data)

    #add column of log ratios aftr subtracting avg to norm around zero
#    smooth_frame['norm_tor']=(smooth_frame['tor']-smooth_frame['tor'][smooth_frame['chromosome']!='chrX'].mean())/smooth_frame['tor'][smooth_frame['chromosome']!='chrX'].std()
    if not os.path.exists(smooth_ToR_file.rsplit("/",1)[0]):
        os.makedirs(smooth_ToR_file.rsplit("/",1)[0])

    smooth_frame.to_csv(smooth_ToR_file,sep='\t',index=None,na_rep='nan')

    #PRINT STATS        
    with open(data_file, 'w') as out_file:
        out_file.write('chro\tnumber_windows\tav_win_size\tstdev_win_size\ttotal_g_read\ttotal_s_reads\n')
        for chro in chroms:
            out_file.write(chro+'\t')
            total_len=len(raw_data[(raw_data['chromosome']=='chr'+chro) & (np.isfinite(raw_data['z_score']))])
            used_s=(raw_data['s'][(raw_data['chromosome']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]).sum()
            used_g=(raw_data['g'][(raw_data['chromosome']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]).sum()
            total_mean=(raw_data['end'][(raw_data['chromosome']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]-raw_data['start'][(raw_data['chromosome']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]).mean()
            total_sd=(raw_data['end'][(raw_data['chromosome']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]-raw_data['start'][(raw_data['chromosome']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]).std()
            data_list=[total_len,total_mean,total_sd,used_g,used_s]
            for item in data_list:
                out_file.write(str(item)+'\t')
            out_file.write('\n')
        total_len=len(raw_data)
        total_s=(raw_data['s']).sum()
        total_g=(raw_data['g']).sum()
        total_len=len(raw_data)
        total_mean=(raw_data['end']-raw_data['start']).mean()
        total_sd=(raw_data['end']-raw_data['start']).std()
        data_list=['total', total_len,total_mean,total_sd,total_g,total_s]
        for item in data_list:
            out_file.write(str(item)+'\t')
        out_file.write('\n'+str(parameters)+'\n')

def windows_genome_sliding_window (exp,rep, smoothing,window_size,step,name,genome,sorted_path,chroms,pooled_file=None,sep=None,allele=None,min_win=3):
    """ Creates windows to compare readings
    name is the experiment name and exp is the sample"""
    parameters=locals()
    print(exp,rep,allele)
    #INITIALIZE
    if rep=='':
        exp_name=exp
    else:
        exp_name=exp+sep+rep

    if allele:
        smooth_ToR_file=smooth_ToR_file_patt%{'exp':'sliding.'+exp_name+sep+allele,'name':name}
        data_file=ToR_file_patt%{'exp':'sliding.'+exp_name+sep+allele+".data",'name':name}
        ToR_file=ToR_file_patt%{'exp':'sliding.'+exp_name+sep+allele,'name':name}
        read_file_patt=sorted_path.rsplit("/",1)[0]+'/%(allele)s2.bed_sort.sorted'
        s=read_file_patt%{'rep':rep,'exp':exp,'phase':'S','allele':allele.lower()}
        g1=read_file_patt%{'rep':rep,'exp':exp,'phase':'G1','allele':allele.lower()}
    else:
        smooth_ToR_file=smooth_ToR_file_patt%{'exp':'sliding.'+exp_name,'name':name}
        data_file=ToR_file_patt%{'exp':'sliding.'+exp_name+".data",'name':name}
        ToR_file=ToR_file_patt%{'exp':'sliding.'+exp_name,'name':name}
        read_file_patt=sorted_path.rsplit("chr%",1)[0]+'all.bed'
        s=read_file_patt%{'rep':rep,'exp':exp,'phase':'S'}
        g1=read_file_patt%{'rep':rep,'exp':exp,'phase':'G1'}

    g1_bed=pybedtools.bedtool.BedTool(g1)
    s_bed=pybedtools.bedtool.BedTool(s)
    overlapping_windows_template=pybedtools.bedtool.BedTool()
    overlapping_windows_template=overlapping_windows_template.window_maker(genome=genome,w=window_size,s=step)
    overlapping_windows_template=overlapping_windows_template.intersect(g1_bed,c=True)
    overlapping_windows_template=overlapping_windows_template.intersect(s_bed,c=True)
    gaps=pybedtools.bedtool.BedTool(gap_file_all%{'genome':genome})
    overlapping_windows_template=overlapping_windows_template.intersect(gaps,v=True) #remove gaps

    raw_data=overlapping_windows_template.to_dataframe(names=['chr','start','end','g','s'])
    min_g1=400
    raw_data=raw_data[raw_data['g']>=min_g1]
    
    #NORMALIZE BY FULL GENOME+ ADD COLUMNS
    total_g=raw_data['g'].sum()
    total_s=raw_data['s'].sum()
    factor=float(total_g)/float(total_s)
    raw_data['normal_s']=raw_data['s']*factor
    raw_data['mid']=(raw_data['start']+raw_data['end'])/2
    raw_data['mid']=raw_data['mid'].astype(int)
    raw_data['ratio']=raw_data['normal_s']/raw_data['g']
    raw_data['log_ratio']=np.log10(raw_data['ratio'])
    #CURRENTLY Z SCORE SKIPS LOG
    ###added clause because in BJ-FTB2 std was huge --> need to remove crazy outliers
    std_cur=raw_data['ratio'].std()
    raw_data=raw_data[abs(raw_data['ratio'])<(std_cur*10)]
    raw_data=raw_data[raw_data['s']>1]
    raw_data=raw_data[raw_data['s']<1000]
    raw_data['z_score']=(raw_data['ratio']-raw_data['ratio'].mean())/raw_data['ratio'].std()
    if not os.path.exists(ToR_file.rsplit("/",1)[0]):
        os.makedirs(ToR_file.rsplit("/",1)[0])

    raw_data.to_csv(ToR_file,sep='\t',index=None,na_rep='nan')
    tors=pybedtools.bedtool.BedTool.from_dataframe(raw_data[['chr','start','end','z_score']])
    #CREATE ITERATED DATA
    smooth_frame=pybedtools.bedtool.BedTool()
    smooth_frame=smooth_frame.window_maker(genome="mm9",w=window_size)
    smooth_frame=smooth_frame.intersect(gaps,v=True)
    print(len(smooth_frame))
    smooth_frame=smooth_frame.intersect(tors,wa=True,wb=True)
    print(len(smooth_frame))
    smooth_frame=smooth_frame.merge(c=[7,7],o=['count','mean'],d=-1)
    print(len(smooth_frame))
    #for some reason from raw to smooth I get huge gaps???

    
    smooth_frame=smooth_frame.to_dataframe(names=['chromosome','start','end','count','tor'])
    smooth_frame=smooth_frame[smooth_frame['count']>=min_win]
    print(len(smooth_frame))
    smooth_frame=smooth_frame[['chromosome','start','end','tor']]
    if not os.path.exists(smooth_ToR_file.rsplit("/",1)[0]):
        os.makedirs(smooth_ToR_file.rsplit("/",1)[0])
    smooth_frame.to_csv(smooth_ToR_file,sep='\t',index=None,na_rep='nan')

    #PRINT STATS        
    with open(data_file, 'w') as out_file:
        out_file.write('chro\tnumber_windows\tav_win_size\tstdev_win_size\ttotal_g_read\ttotal_s_reads\n')
        for chro in chroms:
            out_file.write(chro+'\t')
            total_len=len(raw_data[(raw_data['chr']=='chr'+chro) & (np.isfinite(raw_data['z_score']))])
            used_s=(raw_data['s'][(raw_data['chr']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]).sum()
            used_g=(raw_data['g'][(raw_data['chr']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]).sum()
            total_mean=(raw_data['end'][(raw_data['chr']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]-raw_data['start'][(raw_data['chr']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]).mean()
            total_sd=(raw_data['end'][(raw_data['chr']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]-raw_data['start'][(raw_data['chr']=='chr'+chro) & (np.isfinite(raw_data['z_score']))]).std()
            data_list=[total_len,total_mean,total_sd,used_g,used_s]
            for item in data_list:
                out_file.write(str(item)+'\t')
            out_file.write('\n')
        total_len=len(raw_data)
        total_s=(raw_data['s']).sum()
        total_g=(raw_data['g']).sum()
        total_len=len(raw_data)
        total_mean=(raw_data['end']-raw_data['start']).mean()
        total_sd=(raw_data['end']-raw_data['start']).std()
        data_list=['total', total_len,total_mean,total_sd,total_g,total_s]
        for item in data_list:
            out_file.write(str(item)+'\t')
        out_file.write('\n'+str(parameters)+'\n')


def multi_interp_sliding(name,exp_names):
    """the equivalent of interpolation - sliding windows don't need interpolation but for the sake of keeping the pipeline standard and comparable in the downstream """
    tor_type='tor'
    exp=exp_names[0]
    tor_col_names=[]
    first_smooth=smooth_ToR_file_patt%{'exp':'sliding.'+exp,'name':name}
    data=pd.read_csv(first_smooth,sep='\t', index_col=None,na_values='nan')
    col_dig=5  #this is the first of the tor columns
    col_name=str(col_dig)+'_tor_'+exp
    data.columns=['#1_chr', '2_start', '3_end']+[col_name]
    data['4_mid']=(data['2_start']+data['3_end'])/2
    tor_col_names=[col_name]

    for exp in exp_names[1:]:
        col_dig+=1
        col_name=str(col_dig)+'_tor_'+exp
        cur_file=smooth_ToR_file_patt%{'exp':'sliding.'+exp,'name':name}
        new_data=pd.read_csv(cur_file,sep='\t', index_col=None)
        new_data[col_name]=new_data[tor_type]
        new_data=new_data[['chromosome','start',col_name]]
        data=pd.merge(data, new_data, right_on=['chromosome','start'], left_on=['#1_chr', '2_start'], how='outer')
        tor_col_names.append(col_name)
    print(len(data))
    data=data.dropna(subset=['2_start', '3_end', '4_mid'])
    print(len(data))
    data['2_start']=data['2_start'].astype(int)
    data['3_end']=data['3_end'].astype(int)
    data['4_mid']=data['4_mid'].astype(int)
    step=int(min([abs(a) for a in data['4_mid'][data['chromosome']=='chr1']-data['4_mid'][data['chromosome']=='chr1'].shift(1) if not np.isnan(a)])/2)
    win_size=str(step*2/1000)
    data=data[['#1_chr', '2_start', '3_end', '4_mid']+tor_col_names]
    data.sort(['#1_chr', '2_start'],ascending=[1,1],inplace=True)
    data.to_csv(multi_exp_file%{"win_size":win_size,'name':name},sep='\t',index=False,na_rep='nan')
    
def multi_interp(name,exp_names,normalized=False):
    """takes all locations  and combines into one file holding all data.  Inserts nan for missing data"""
    #### other tor type for non normalized tor is just 'tor'
    print('multi-interp') 
    if normalized:
        tor_type='norm_tor'
    else:
        tor_type='tor'

    exp=exp_names[0]
    tor_col_names=[]
    first_smooth=smooth_ToR_file_patt%{'exp':exp,'name':name}
    data=pd.read_csv(first_smooth,sep='\t', index_col=None,na_values='nan')
    col_dig=5  #this is the first of the tor columns
    col_name=str(col_dig)+'_tor_'+exp
    data[col_name]=data[tor_type]

    data=data[['chromosome','loc',col_name]]
    tor_col_names.append(col_name)
    for exp in exp_names[1:]:
        col_dig+=1
        col_name=str(col_dig)+'_tor_'+exp
        cur_file=smooth_ToR_file_patt%{'exp':exp,'name':name}
        new_data=pd.read_csv(cur_file,sep='\t', index_col=None)
        new_data[col_name]=new_data[tor_type]
        new_data=new_data[['chromosome','loc',col_name]]
        data=pd.merge(data, new_data, on=['chromosome','loc'], how='outer')
        tor_col_names.append(col_name)
    data['4_mid']=data['loc'].astype(int)
    step=int(min([abs(a) for a in data['4_mid'][data['chromosome']=='chr1']-data['4_mid'][data['chromosome']=='chr1'].shift(1) if not np.isnan(a)])/2)
    win_size=str(step*2/1000)
    data['2_start']=data['loc'].astype(int)-step
    data['3_end']=data['loc'].astype(int)+step
    data['#1_chr']=data['chromosome']
    data=data[['#1_chr', '2_start', '3_end', '4_mid']+tor_col_names]
    data.sort_values(['#1_chr', '2_start'],ascending=[1,1],inplace=True)
    if normalized:
        data.to_csv(multi_exp_file%{"win_size":win_size,'name':name},sep='\t',index=False,na_rep='nan')
    else:
        data.to_csv(multi_exp_file%{"win_size":'_not_normalized'+win_size,'name':name},sep='\t',index=False,na_rep='nan')

############################## calculating Differential Regions ############################################

def likelihood(group1,group2,sds,df=1):
    def probability(sample,mean, sd):
        return(1/(np.sqrt(2*np.pi)*sd)*np.e**-((sample-mean)**2/(2*sd**2)))
    group1=group1.values.tolist()
    group2=group2.values.tolist()
    two_group_likeli=[probability(a,np.mean(group1),sds[0]) for a in group1]+[probability(a,np.mean(group2),sds[1]) for a in group2]
    all_group=group1+group2
#    one_group_likeli=[probability(a,np.mean(all_group),sds[2]) for a in all_group]
    one_group_likeli=[probability(a,np.mean(all_group),sds[0]) for a in group1]+[probability(a,np.mean(all_group),sds[1]) for a in group2]
    two=reduce(lambda x, y: x*y, two_group_likeli)
    one=reduce(lambda x, y: x*y, one_group_likeli)
    return(chi2.sf(-2*np.log(one/two),df))

def likelihood_ratio_delta(name,other_project_names=None,colors=None,graph=True,cutoff2=-np.log10(.05),cutoff1=-np.log10(.01),group1=[],group2=[],chro='1',const_sd=None,silent=False,win_size='100',remove_X=True,save_figure_data=[]):
    rel_file=multi_exp_file%{'name':name,'win_size':win_size}
    our_data=pd.read_csv(rel_file,sep='\t',na_values='nan')
    our_data.columns=[a+sep+name if a not in joining_cols else a for a in our_data.columns]

    if other_project_names!=None:
        for other_project_name in other_project_names:
            other_file=multi_exp_file%{'name':other_project_name,'win_size':win_size}
            other_data=pd.read_csv(other_file,sep='\t',na_values='nan')
            other_data.columns=[a+sep+other_project_name if a not in joining_cols else a for a in other_data.columns]

            our_data=pd.merge(our_data,other_data,how='inner',left_on=["#1_chr","4_mid"], right_on=["#1_chr","4_mid"],sort=False,)
    our_data=our_data.dropna(axis=0,how='any')
    if remove_X:
        if not silent:
            print('ignoring x chromosome')
        our_data=our_data[our_data["#1_chr"]!="chrX"]
    our_data.columns=rename_frame(our_data.columns.values.tolist())
    total_length=len(our_data)
    #create basic inner delta data
    if len(group1)==0:
        print(our_data.columns.values.tolist())
        cols1=list(input('first group of columns that you want to compare?  make sure you use a comma.  Type "n" to finish \n'))
        cols2=list(input('second group of columns that you want to compare?  make sure you use a comma.  Type "n" to finish \n'))
        group1=cols1
        group2=cols2
    else:
        cols1=group1
        cols2=group2
    frame=our_data
    start_col=[a for a in frame.columns if 'start' in a][0]
    #make and check distribution
    distribs=[]
    if graph==True:
        f,axarr=plt.subplots(3,2)
    names=['A','B','A and B']
    plot=0
    
    for group in [group1,group2,group1+group2]:
        distrib=[]
        for i in range(len(group)):
            d=frame[group[i]]-frame[group].mean(axis=1).values.tolist()
            distrib.extend(d)
        if not silent:
            print(shapiro(distrib))
        if graph==True:
            sns.distplot(distrib,ax=axarr[plot,0])
            probplot(distrib, dist="norm", plot=axarr[plot,1])
        distribs.append(distrib)
        plot+=1
    if graph==True:
        plt.tight_layout()
        plt.show()
    sds=[np.std(distrib) for distrib in distribs]
    if not silent:
        print(sds)
    df=1
    if type(const_sd)==float:
        sds=[const_sd, const_sd]
        df=2
        if not silent:
            print(sds)
    elif type(const_sd)==str:
        sds=[min(sds),min(sds)]
        df=2
        if not silent:
            print(sds)
    frame['likelihood_test_p value']=frame[group1+group2].apply(lambda x: -np.log10(likelihood(x[:len(group1)],x[len(group1):],sds,df)),axis=1)
    frame['likelihood_pv']=frame[group1+group2].apply(lambda x: likelihood(x[:len(group1)],x[len(group1):],sds,df),axis=1)
    frame['likelihood_test_FDR']=-np.log10(multipletests(frame['likelihood_pv'],method='bonferroni')[1].tolist())
#    frame['likelihood_test_FDR']=-np.log10(multipletests(frame['likelihood_pv'],method='fdr_by')[1].tolist())

    first_pass_list=frame[frame['likelihood_test_FDR']>=cutoff1].index.tolist()
    pass_list=frame[frame['likelihood_test_FDR']>=cutoff2].index.tolist()
    print('number of windows added in extension: '+str(len(pass_list)-len(first_pass_list)))
    #take consecutive groups passing lenient cutoff and check they contain at least one place passing stringent cutoff
    lrt_groups=[]
    for k, g in groupby(enumerate(pass_list), lambda (i,x):i-x):
        group = map(itemgetter(1), g) 
        lrt_groups.append(group)

    lrt_groups=[group for group in lrt_groups if bool(set(group) & set(first_pass_list))]
    groups_flat=[item for sublist in lrt_groups for item in sublist]
    frame['dif']=False
    frame.loc[groups_flat,'dif']=True
    frame[frame['dif']==True].to_csv('/mnt/lustre/hms-01/fs01/britnyb/lab_files/temp.bed',index=None,sep='\t')

    if graph==True:
        gframe=frame[frame['#1_chr']=='chr'+chro]
        chro_indeces=gframe.index.tolist()
        ax1=plt.subplot(211)
        ax2=plt.subplot(212, sharex=ax1)
        for tor in group1:
            ax1.plot(gframe[start_col],gframe[tor],color='red',label=tor)
        ax1.set_ylabel(RT_label) 

        for tor in group2:
            ax1.plot(gframe[start_col],gframe[tor],color='blue',label=tor)
            
        for group in lrt_groups:
            if group[0]<min(chro_indeces) or group[-1]>max(chro_indeces):
                continue
            ax1.axvspan(gframe['4_mid'].loc[group[0]],gframe['4_mid'].loc[group[-1]],color='red',alpha=.3)

        for test in [a for a in gframe.columns if 'test' in a]:
            print(test)
            ax2.plot(gframe[start_col],gframe[test],label=test.split('test_')[1])
        ax2.set_ylabel('-log')
        
                    
    #    h1, l1 = ax1.get_legend_handles_labels()
    #    h2, l2 = ax2.get_legend_handles_labels()
    #    ax1.legend(h1+h2, l1+l2)
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
    #    ax1.legend(h1+h2, l1+l2, loc='center left', bbox_to_anchor=(1, 0.5))
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        lgd=ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        mkfunc=lambda x,pos: '%1.1fM' % (x * 1e-6) 
        mkformatter=ticker.FuncFormatter(mkfunc)
        ax1.xaxis.set_major_formatter(mkformatter)
        ax2.xaxis.set_major_formatter(mkformatter)
        ax1.xaxis.set_visible(False)

        plt.xlabel('location on chro '+chro)
        plt.subplots_adjust(hspace=0.001)
        if len(save_figure_data)>0:
			print('saving')
			fig_name,fig_fold,inches=save_figure_data
			save_figure(fig_name,fig_fold,inches,lgd)

        plt.show()
    main_data=(group1[0],group2[0])
    if not silent:
        print(len(lrt_groups))
    our_data=frame
    our_data.to_csv((diff_regions_file_delt%{'sample1':main_data[0],'sample2':main_data[1]+'_everything'}).replace(' ','_'),sep='\t',na_rep='nan',index=False)
    if not silent:
        print((diff_regions_file_delt%{'sample1':main_data[0],'sample2':main_data[1]+'_everything'}).replace(' ','_'))
    our_data=our_data[our_data['dif']==True]
    our_data.to_csv((diff_regions_file_delt%{'sample1':main_data[0],'sample2':main_data[1]+'_diff'}).replace(' ','_'),sep='\t',na_rep='nan',index=False)
    if not silent:
        print((diff_regions_file_delt%{'sample1':main_data[0],'sample2':main_data[1]+'_diff'}).replace(' ','_'))
        print('%genome= '+str(float(len(our_data))/float(total_length)*100))
        print('num dif regions '+str(len(lrt_groups)))
        print('mean size: '+str(np.mean([len(g) for g in lrt_groups])*100)+' kb')
    our_data1=our_data[our_data[main_data[0]]>our_data[main_data[1]]]
    our_data2=our_data[our_data[main_data[1]]>our_data[main_data[0]]]
    our_data1.to_csv((diff_regions_file_delt%{'sample1':main_data[0],'sample2':main_data[1]}).replace(' ','_'),sep='\t',na_rep='nan',index=False)
    our_data2.to_csv((diff_regions_file_delt%{'sample1':main_data[1],'sample2':main_data[0]}).replace(' ','_'),sep='\t',na_rep='nan',index=False)
    if not silent:
        print(main_data[0]+' earlier: '+str(float(len(our_data1))/float(len(our_data))))
        print(main_data[1]+' earlier: '+str(float(len(our_data2))/float(len(our_data))))


def graph(name,chro,samples,colors=[],win_size='100',saving_only=False,SIZE=10,norm=True,save_figure_data=[],zoom=[]):
    """Graphs ToR """

    if not norm:
        rel_file=multi_exp_file%{"win_size":'_not_normalized'+win_size,'name':name}
    else:
        rel_file=multi_exp_file%{'name':name,'win_size':win_size}
        
    print(rel_file)
    plt.rc('axes', color_cycle=color_cycler(colors))

    plt.rc('font', size=SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title


    data=pd.read_csv(rel_file,sep='\t',na_values='nan')
    data=data[data['#1_chr']=='chr'+chro]
    data.columns=[a+sep+name if a not in joining_cols else a for a in data.columns]

    data.columns=rename_frame(data.columns.values.tolist())
    samples=rename_frame(samples)
    
    if saving_only:
        resolution=5000000/(int(win_size)*1000)
        length=len(data)
        pic_counter=1
        for i in range(0,length,resolution):
            start=i
            end=i+resolution
            subdata=data[start:end]
            fig, ax = plt.subplots()
            for sample in samples:
                subdata.plot('4_mid',sample,label=sample)
            plt.xlabel('Location on Chromosome %s'%chro)
            plt.ylabel(RT_label)
            plt.legend()
            fig.savefig(pic_file%{'name':name,'chro':chro,'num':str(pic_counter)})
            pic_counter+=1

    else:
        for sample in samples:
            print(sample)
            plt.plot(data['4_mid'],data[sample],label=sample)
        plt.xlabel('Location on Chromosome %s'%chro, labelpad=20)
        plt.ylabel(RT_label)
        if len(zoom)>0:
            plt.xlim(zoom)
        ax=plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        mkfunc=lambda x,pos: '%fM' % (x * 1e-6) 
        mkformatter=ticker.FuncFormatter(mkfunc)
        ax.xaxis.set_major_formatter(mkformatter)
        if len(save_figure_data)>0:
		    print('saving')
		    fig_name,fig_fold,inches=save_figure_data
		    save_figure(fig_name,fig_fold,inches,lgd)

        plt.show()
        print('hey')
        
def graph_with_other(name,chro,samples,colors,other_project_names,genomes,win_size='100',SIZE=10,include_gc=False,star_gc=False,norm=True,include_mut=False,frame=[],save_figure_data=[],zoom=[]):
    """Graphs ToR with samples from another experiment """
    plt.rc('axes', color_cycle=color_cycler(colors))
    plt.rc('font', size=SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SIZE)  # fontsize of the figure title

    if not norm:
        rel_file=multi_exp_file%{"win_size":'_not_normalized'+win_size,'name':name}
    else:
        rel_file=multi_exp_file%{'name':name,'win_size':win_size}

    if len(frame)>1:
        our_data=frame.dropna()
    else:
        our_data=pd.read_csv(rel_file,sep='\t',na_values='nan')
        counter=1
        our_data.columns=[a+sep+name if a not in joining_cols else a for a in our_data.columns]

        for other_project_name in other_project_names:
            if not norm:
                other_file=multi_exp_file%{"win_size":'_not_normalized'+win_size,'name':other_project_name}
            else:
                other_file=multi_exp_file%{'name':other_project_name,'win_size':win_size}
            other_data=pd.read_csv(other_file,sep='\t',na_values='nan')
            if genomes[0]!=genomes[counter]:
                print('lifting over')
                print(genomes[0],genomes[counter])
                lo=LiftOver(genomes[counter],genomes[0])
                other_data['4_mid']=other_data.apply(lambda row: convert_help(row['#1_chr'],int(row['4_mid']),lo,win_size),axis=1)
            counter+=1
            other_data.columns=[a+sep+other_project_name if a not in joining_cols else a for a in other_data.columns]

            our_data=pd.merge(our_data,other_data,how='inner',left_on=["#1_chr","4_mid"], right_on=["#1_chr","4_mid"],sort=False)
    our_data=our_data[our_data['#1_chr']=='chr'+chro]
    our_data.columns=rename_frame(our_data.columns.values.tolist())
    samples=rename_frame(samples)
    print(samples,our_data.columns)

    plt.rc('axes', color_cycle=color_cycler(colors))    
    for sample in samples:
        print(sample)
        plt.plot(our_data['4_mid'],our_data[sample],label=sample)
    plt.xlabel('Location on Chromosome %s'%chro,labelpad=20)
    plt.ylabel(RT_label)
    ax=plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if include_mut:
        if include_mut=='divergence':
            ax2 = ax.twinx()
            gc=pd.read_csv(chen_bed_extended,sep='\t')
            gc=gc[gc['#1_chr']=='chr'+chro]
            print(gc.columns.values.tolist())
            col='mutation_rate'
            gc[col]=gc[col]*(-1)
            gc[col]=gc[col]
            gc[col]=gc[col].rolling(window=7).mean()  #this may have been 3
    ##        gc[col]=(gc[col]-gc[col].mean())/gc[col].std()
    ##        ax.plot(gc['4_mid'],gc[col],label='Inverse Divergence',color='grey',linestyle='dashed')
            ax2.plot(gc['4_mid'],gc[col],label='Inverse Divergence',color='grey',linestyle='dashed')
    ##        plt.ylabel('Replication Time / Inverse Divergence')
            ax2.set_ylim(-.16,-.1)
            ax2.set_ylabel('Inter-mammalian Divergence')
            ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax2.grid(None)
            ax2.legend(loc='center left', bbox_to_anchor=(1.45, 1))

        else:
            ax3 = ax.twinx()
    ##        ax3.spines['right'].set_position(('axes', 1.25))
    #        gc=pd.read_csv(chen_real_snps,sep='\t')
            gc=pd.read_csv(hotspot_itamar_mega,sep='\t')
            
            gc=gc[gc['#1_chr']=='chr'+chro]
            print(gc.columns.values.tolist())
    #        col='5_SNPs'
            col='5_hotspots'
            if col in our_data.columns:
                gc=our_data
            gc=gc[gc[col]!='.']
            gc[col]=gc[col].astype(int)
    #        gc[col]=gc[col]*(-1)
            gc[col]=gc[col].rolling(window=2).mean()
    ##        gc[col]=(gc[col]-gc[col].mean())/gc[col].std()
    ##        ax.plot(gc['4_mid'],gc[col],label='Inverse Divergence',color='grey',linestyle='dashed')
    ##        ax3.plot(gc['4_mid'],gc[col],label='SNP density',color='black',linestyle='dashed')
            ax3.plot(gc['4_mid'],gc[col],label='Hotspots Density',color='black',linestyle='dashed')
    ##        plt.ylabel('Replication Time / Inverse Divergence')
    ##        ax3.set_ylim(30,125)
    ##        ax3.set_ylabel('SNP Density')
    ##        ax3.set_ylim(0,1.5)
            ax3.set_ylabel('Number of Hotspots')
            ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax3.grid(None)
            ax3.legend(loc='center left', bbox_to_anchor=(1.45, .9))

    if include_gc:
        ax2 = ax.twinx()
        gc=pd.read_csv(gc_data_mm9,sep='\t')
        gc=gc[gc['#1_chr']=='chr'+chro]
        print(gc.columns.values.tolist())
        col='6_pct_gc'
        ax2.plot(gc['4_mid'],gc[col],label='GC content',color='black',linestyle='dashed')
#        ax2.plot(gc['4_mid'],gc[col],label='gc',color='black',linewidth=2)
        ax2.set_ylabel('GC content')
        ax2.set_ylim(.3,.55)
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.grid(None)
##        if include_mut:
##            ax2.legend(loc='center left', bbox_to_anchor=(1.4, 1))
##            ax3 = ax.twinx()
##            gc=pd.read_csv(chen_bed_extended,sep='\t')
##            gc=gc[gc['#1_chr']=='chr'+chro]
##            print(gc.columns.values.tolist())
##            col='mutation_rate'
##            gc[col]=gc[col]*(-1)
##            ax3.plot(gc['4_mid'],gc[col],label='mutation rate',color='gray',linestyle='dashed')
##    #        ax2.plot(gc['4_mid'],gc[col],label='gc',color='black',linewidth=2)
##            ax3.set_ylabel('mutation rate')
##    #        ax2.set_ylim(.3,.55)
##            ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
##            ax3.spines['right'].set_position(('outward', 60))      
##
##            ax3.grid(None)
##            ax3.legend(loc='center left', bbox_to_anchor=(1.4, .8))
##        else:
        ax2.legend(loc='center left', bbox_to_anchor=(1.4, 1))

    
    if star_gc:
        gc=pd.read_csv(gc_data_mm9,sep='\t')
        gc=gc[gc['#1_chr']=='chr'+chro]
        gc=pd.merge(gc,our_data,how='inner',left_on=["#1_chr","4_mid"], right_on=["#1_chr","4_mid"],sort=False)
        print(gc.columns.values.tolist())
        col='6_pct_gc'
#        ax2.plot(gc['4_mid'],gc[col],label='gc',color='black',linestyle='dashed')
        gc=gc[gc[col]>(gc[col].mean()+gc[col].std())]
        plt.scatter(gc['4_mid'],gc[samples[0]],marker='*',label='GC 1 sd above mean')

##
##    if input('include isr? y/n \n')=='y':
##        ax2 = ax.twinx()
##        gc=pd.read_csv(isr_data,sep='\t')
##        gc['4_mid']=(gc['2_start']+gc['3_end'])/2
##        gc=gc[gc['#1_chr']=='chr'+chro]
##        print(gc.columns.values.tolist())
##        col=input('which column?\n')
##        ax2.scatter(gc['4_mid'],gc[col],label='isr',color='black')
##        ax2.set_ylabel('isr')
##       #ax2.set_ylim(.3,.55)
##        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    
    # Put a legend to the right of the current axis
    if include_mut or include_gc:
        lgd=ax.legend(loc='center left', bbox_to_anchor=(1.4, 0.5))

    else:
        lgd=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if len(zoom)>0:
        plt.xlim(zoom)
    mkfunc=lambda x,pos: '%dM' % (x * 1e-6) 
    mkformatter=ticker.FuncFormatter(mkfunc)
    ax.xaxis.set_major_formatter(mkformatter)
    if len(save_figure_data)>0:
        print('saving')
        fig_name,fig_fold,inches=save_figure_data
        save_figure(fig_name,fig_fold,inches,lgd)

##	save_figure(save_figure_data)
    plt.tight_layout()
    plt.show()


def is_normal(name,win_size='100'):
    ''' checks if the data is a normal distribution '''
    rel_file=multi_exp_file%{'name':name,'win_size':win_size}
    our_data=pd.read_csv(rel_file,sep='\t',na_values='nan')

    rel_cols=[a for a in our_data.columns if 'tor' in a]
    for a in rel_cols:
        data=our_data[a].values.tolist()
        print(a)
        sns.kdeplot(our_data[a],label=a)
        plt.show()
        print(shapiro(data))

        
def heatmap(name,samples,other_project_names=None,genomes=None,win_size='100',abc=False,method='spearman',filt=None,manual_order=False,scale=None,remove_x=False,save_figure_data=[]):
    """Heatmap of correlations of ToR profiles """
    print(name,samples,other_project_names,genomes,win_size,abc,method,filt)
    sep='.'
    rel_file=multi_exp_file%{'name':name,'win_size':win_size}
    joining_cols=["#1_chr","4_mid"]
    our_data=pd.read_csv(rel_file,sep='\t',na_values='nan')
    our_data.columns=[a+sep+name if a not in joining_cols else a for a in our_data.columns]
    our_data=our_data[[a for a in our_data.columns if a in samples]+[a for a in joining_cols]]

    if other_project_names!=None:
        for other_project_name in other_project_names:
            #print(other_project_name)
            other_file=multi_exp_file%{'name':other_project_name,'win_size':win_size}
            other_data=pd.read_csv(other_file,sep='\t',na_values='nan')
            if genomes[0]!=genomes[1]:
                lo=LiftOver(genomes[1],genomes[0])
                other_data['4_mid']=other_data.apply(lambda row: convert_help(row['#1_chr'],int(row['4_mid']),lo,win_size),axis=1)

                
            other_data.columns=[a+sep+other_project_name if a not in joining_cols else a for a in other_data.columns]
            other_data=other_data[[a for a in other_data.columns if a in samples]+[a for a in joining_cols]]

            our_data=pd.merge(our_data,other_data,how='inner',left_on=joining_cols, right_on=joining_cols,sort=False,)


    if filt:
        print(len(our_data))
        filt_data=pd.read_csv(filt,sep='\t')
        our_data=pd.merge(our_data,filt_data,how='inner',left_on=joining_cols, right_on=joining_cols,sort=False) 
        print(len(our_data))
    if remove_x:
        print('removed X')
        our_data=our_data[our_data['#1_chr']!='chrX']
    print(len(our_data))
    our_data=our_data[samples]
    our_data.columns=rename_frame(our_data.columns.values.tolist())
    if abc:
        our_data=our_data[sorted(our_data.columns)]
    if manual_order:
        print(our_data.columns.values.tolist())
        our_data=our_data[list(input('reorder the columns as you"d like\n'))]

    correlation_matrix=our_data.corr(method=method)
    print(correlation_matrix)
    correlation_matrix.to_csv('/mnt/lustre/hms-01/fs01/britnyb/temp_heat.txt',sep='\t',na_rep='nan')
    if scale:
        cm1 = colors.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
        plt.pcolor(correlation_matrix,cmap=cm1,vmin=scale)
        plt.pcolor(correlation_matrix,cmap='YlOrBr')
    else:
        cm1 = colors.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])

        plt.pcolor(correlation_matrix,cmap=cm1)
        plt.pcolor(correlation_matrix,cmap='YlOrBr')
    plt.yticks(np.arange(0.5, len(correlation_matrix.index), 1), correlation_matrix.index)
    plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns,rotation='vertical')        
    cbar=plt.colorbar()
    plt.title('Heatmap of RT profile correlations\n')
#    plt.savefig(DIR+'/heatmap_correlations.jpg')
    if len(save_figure_data)>0:
		print('saving')
		fig_name,fig_fold,inches=save_figure_data
		save_figure(fig_name,fig_fold,inches)
    plt.show()


def create_mask(sample,name,sorted_file,chroms,allele=None,rep=None,win_size='100'):
    skel=pybedtools.bedtool.BedTool(multi_exp_file%{'name':name,'win_size':win_size})
    for phase in ['S','G1']:
        reads=pd.DataFrame(columns=['chro','start','end'])
        for chro in chroms:
            if allele:
                if rep:
                    exp_out=sample+'.'+rep+'.'+allele
                    read_counts_f=sorted_file%{'phase':phase,'exp':sample,'chro':chro,'allele':allele,'rep':rep}
                else:  
                    exp_out=sample+'.'+allele
                    read_counts_f=sorted_file%{'phase':phase,'exp':sample,'chro':chro,'allele':allele}
            else:
                if rep:
                    exp_out=sample+'.'+rep
                    read_counts_f=sorted_file%{'phase':phase,'exp':sample,'chro':chro,'rep':rep}
                else:
                    exp_out=sample
                    read_counts_f=sorted_file%{'phase':phase,'exp':sample,'chro':chro}
                    
            raw_data=pd.read_csv(read_counts_f,sep="\t",names=['chro','start'])
            raw_data['end']=raw_data['start']+1
            reads=reads.append(raw_data)
        reads['start']=reads['start'].astype(int)
        reads['end']=reads['end'].astype(int)
        reads_bed=pybedtools.bedtool.BedTool.from_dataframe(reads)
        skel=skel.intersect(reads_bed,c=True)
    frame=skel.to_dataframe()
    length=len(frame.columns)
    frame=frame[[0,1,2,3,length-2,length-1]]
    frame.columns=["1_chr","2_start","3_end","4_mid","S_reads","G1_reads"]
    path=counts_file%{'name':name,'exp':exp_out}
    path=path.rsplit("/",1)[0]
    if not os.path.exists(path):
        os.makedirs(path)
    frame.to_csv(counts_file%{'name':name,'exp':exp_out},sep="\t",index=None)
    frame_clean=frame[(frame['G1_reads']>np.mean(frame['G1_reads'])-np.std(frame['G1_reads'])*2) & (frame['G1_reads']<np.mean(frame['G1_reads'])+np.std(frame['G1_reads'])*2) &
                      (frame['S_reads']>np.mean(frame['S_reads'])-np.std(frame['S_reads'])*2) & (frame['S_reads']<np.mean(frame['S_reads'])+np.std(frame['S_reads'])*2)]
    frame_clean.to_csv(counts_file_clean%{'name':name,'exp':exp_out},sep="\t")
        

def plot_raw(name,samples,chro,win_size='100'):
    """plots raw and smoothed data """
    colors=['green','blue','red','orange','purple','gray']
    light_colors=['lightgreen','skyblue','pink']
    color_counter=0
#    smooth_no_norm_data=pd.read_csv(multi_exp_file%{'name':name,'win_size':'100','appendix':appendices.not_normalized.value},sep='\t')
    smooth_norm_data=pd.read_csv(multi_exp_file%{'name':name,'win_size':win_size},sep='\t')
#    smooth_no_norm_data=smooth_no_norm_data[smooth_no_norm_data['#1_chr']=='chr'+chro]
    smooth_norm_data=smooth_norm_data[smooth_norm_data['#1_chr']=='chr'+chro]
    tor_cols=[b for b in smooth_norm_data.columns.values.tolist() if 'tor' in b]
    fig,ax1=plt.subplots()
#    ax2=ax1.twinx()
#    print(smooth_no_norm_data.columns)
    print(smooth_norm_data.columns)
    for sample in samples:
        color=colors[color_counter]
        light_color=light_colors[color_counter]
        raw_data=pd.read_csv(ToR_file_patt%{'name':name,'exp':sample},sep='\t',na_values='nan')
        raw_data=raw_data[raw_data['chromosome']=='chr'+chro]
        col=[a for a in tor_cols if a.split('tor_')[1]==sample][0]
        print(col)
    #    ax2.plot(smooth_no_norm_data['4_mid'],smooth_no_norm_data[col],label=sample+' smooth unnormalized data',color=light_color)
        ax1.plot(smooth_norm_data['4_mid'],smooth_norm_data[col],label=sample+' MEF smooth ToR',color=color)
        ax1.scatter(raw_data['mid'],raw_data['z_score'],label=sample+' MEF normalized ToR',color=light_color,s=7,alpha=.6)
        print(len(raw_data['z_score']))
        print(len(smooth_norm_data[col]))
        color_counter+=1
    plt.xlabel('location')
    ax1.set_ylabel('Replication Time normalized')
#    ax1.set_ylim([smooth_no_norm_data[col].min(),smooth_no_norm_data[col].max()])
####    ax1.set_ylim([smooth_no_norm_data[col].min(),(smooth_no_norm_data[col].max()+1)])
    ax1.set_ylim([smooth_norm_data[col].min()-1,smooth_norm_data[col].max()+1])
####    ax2.set_ylim([smooth_norm_data[col].min(),(smooth_norm_data[col].max()+1)])
#    ax2.set_ylabel('Replication Time not normalized')
  #  ax2.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    
    plt.title('chromosome %s'%chro)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    
    plt.show()

def convert_help(chro,loc,lift,win_size):
    new=lift.convert_coordinate(chro,loc)
    if len(new)==0:
        return np.nan
    else:
        conversion=int(win_size)*1000
        return ceil(new[0][1]/conversion)*conversion



def color_cycler(pattern):
    colors=[]

    blues=['blue','lightblue','royalblue','deepskyblue','cyan']
    greens=['green','lightgreen','lime','darkgreen','palegreen']
    reds=['red','salmon','pink', 'darkred','tomato','orangered']
    browns=['saddlebrown','chocolate','goldenrod','farkgoldenrod']
    purples=['indigo','darkorchid','darkviolet','purple']
    greys=['darkgray','silver','lightgray']
    color_types=[blues,greens,reds,purples,browns,greys]

    if len(pattern)==0:
        colors=['red','orange','yellow','green','blue','purple','black','gray','cyan','pink','lime','brown','olive']
    elif isinstance(pattern[0],int):
        for index, group_len in enumerate(pattern):
            for i in range(group_len):
                colors.append(color_types[index][i])
    else:
        colors=pattern

    return(colors)


def rename_frame(cols):
    a=pd.read_csv(dictionary,sep='\t')
    d=a.set_index('sample').T.to_dict('list')
    cols=[c.rsplit('tor_')[-1] for c in cols]
    fix_columns=[d[b][0] if b in d.keys() else b for b in cols]
    return(fix_columns)
    
    
def save_figure(fig_name,fig_fold,inches,lgd=None):
    """ This will find the current figure and save it at a given size in inches"""
    fig=plt.gcf()
    fig.tight_layout()
    fig.set_size_inches(inches)
    if lgd:
        fig.savefig(fig_fold+fig_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        fig.savefig(fig_fold+fig_name, bbox_inches='tight')
		
	
