#!/bin/bash

#SBATCH -c32
#SBATCH --mem 100G
#SBATCH -N 1
#SBATCH -o slurm.%N.%j.out
#SBATCH -t 0-1:00		# time (D-HH:MM)


echo $0
echo `date`
data_path=/mnt/lustre/hms-01/fs01/britnyb/lab_files/tor_seq_data
human_chroms=(`seq 22` X Y M)
mouse_chroms=(`seq 19` X Y M)
chicken_chroms=(`seq 28` 32 W Z M)
picard=/mnt/lustre/hms-01/fs01/britnyb/lab_files/bioinformatics/picard-tools-2.1.0/picard.jar




sample_name=$1
genome=$2
exp=$3
paired=$4
ending=$5
organism=$6
echo 'fastq' $sample_name `date` 
mkdir -p $data_path/$exp/mapped/stats

#set -x
if $paired; then
	bowtie2 -q --no-discordant --no-mixed --no-unal -X 1000 -p 32 -x $genome -1 $data_path/$exp/fastq/first_mate/$sample_name.$ending -2 $data_path/$exp/fastq/second_mate/$sample_name.$ending -S $data_path/$exp/mapped/$sample_name.sam >& $data_path/$exp/mapped/$sample_name.bt2.log
else
	bowtie2 -q --no-unal -p 32 -x $genome -U $data_path/$exp/fastq/first_mate/$sample_name.$ending -S $data_path/$exp/mapped/$sample_name.sam >& $data_path/$exp/mapped/$sample_name.bt2.log
fi

	# sam to bam
samtools view -bhu -q 30 $data_path/$exp/mapped/$sample_name.sam | samtools sort - > $data_path/$exp/mapped/$sample_name.bam
samtools index $data_path/$exp/mapped/$1.bam
java -Xmx4g -jar $picard MarkDuplicates INPUT=$data_path/$exp/mapped/$sample_name.bam OUTPUT=$data_path/$exp/mapped/$sample_name.nodups.bam METRICS_FILE=$data_path/$exp/mapped/$sample_name.metrics.txt REMOVE_DUPLICATES=true
samtools index $data_path/$exp/mapped/$sample_name.nodups.bam
echo 'added clause to delete either normal bam and sam to save space on lustre'
rm $data_path/$exp/mapped/$sample_name.bam
rm $data_path/$exp/mapped/$sample_name.sam


if [ $organism == "mouse" ]; then
	chroms=${mouse_chroms[@]}
elif [ $organism == "chicken" ]; then
	chroms=${chicken_chroms[@]}
else
	chroms=${human_chroms[@]}
fi
mkdir -p $data_path/$exp/sorted/$sample_name
echo 'sorting' $sample_name `date`
bam=$data_path/$exp/mapped/$sample_name.nodups.bam
samtools sort -n $bam > $bam.nsorted.bam

	#included sed to deal with the illumina labeling of pairs:
	#in older scripts used to filter using samtools view -bu -F 1804 $bam.nsorted.bam
if $paired; then
	samtools view -h $bam.nsorted.bam | sed 's/\/1;1//g' - | sed 's/\/2;1//g' - | samtools view -bu | bedtools bamtobed -i stdin -bedpe | awk '{print $1"\t"$2"\t"$3"\t"}' | sort -k1,1 -k2,2n > $data_path/$exp/sorted/$sample_name/$sample_name.all.bed
else
	samtools view -bu $bam.nsorted.bam | bedtools bamtobed -i stdin | awk '{print $1"\t"$2"\t"$3"\t"}' | sort -k1,1 -k2,2n > $data_path/$exp/sorted/$sample_name/$sample_name.all.bed 
fi

echo ${chroms[@]}
for i in ${chroms[@]}; do
	awk -v i=$i '{if($1=="chr"i) print $1"\t"$2}' $data_path/$exp/sorted/$sample_name/$sample_name.all.bed | sort -k1,1 -k2,2n > $data_path/$exp/sorted/$sample_name/$sample_name.chr$i.sorted
done
rm $bam.nsorted.bam


sacct -j $SLURM_JOBID -o MaxRSS,AveRSS,ReqMem,Elapsed,AllocCPUS
