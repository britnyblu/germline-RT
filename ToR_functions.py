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
import matlab.engine
eng=matlab.engine.start_matlab()
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



############################## calculating RT ############################################

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



	
