for ((ip=6000;ip<=8000;ip=ip+200))
do
	python predict.py ../../../ismir2014 ../../../final_project_result/thre_0.1/${ip}_predict.json /media/yuhuei.tseng/model/e_${ip}
done
