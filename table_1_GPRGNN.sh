for l in 2 4 8 16 32 64
do
	python -u full-supervised.py --data cora 				--layer $l --lr 0.002 --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 0.1 --model GPRGNN --hidden 64 > GPRGNN_cora_$l.txt 2>&1
	python -u full-supervised.py --data citeseer 		--layer $l --lr 0.002 --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 0.1 --model GPRGNN --hidden 64 > GPRGNN_citeseer_$l.txt 2>&1
	python -u full-supervised.py --data pubmed 			--layer $l --lr 0.05  --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 0.2 --model GPRGNN --hidden 64 > GPRGNN_pubmed_$l.txt 2>&1
	python -u full-supervised.py --data chameleon 	--layer $l --lr 0.05  --dprate_GPRGNN 0.7 --weight_decay 0    --alpha_GPRGNN 1   --model GPRGNN --hidden 64 > GPRGNN_chameleon_$l.txt 2>&1
	python -u full-supervised.py --data cornell 		--layer $l --lr 0.05  --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 0.9 --model GPRGNN --hidden 64 > GPRGNN_cornell_$l.txt 2>&1
	python -u full-supervised.py --data texas 			--layer $l --lr 0.05  --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 1   --model GPRGNN --hidden 64 > GPRGNN_texas_$l.txt 2>&1
	python -u full-supervised.py --data wisconsin 	--layer $l --lr 0.05  --dprate_GPRGNN 0.5 --weight_decay 5e-4 --alpha_GPRGNN 1   --model GPRGNN --hidden 64 > GPRGNN_wisconsin_$l.txt 2>&1
	python -u full-supervised.py --data squirrel 		--layer $l --lr 0.05  --dprate_GPRGNN 0.7 --weight_decay 0    --alpha_GPRGNN 0   --model GPRGNN --hidden 64 > GPRGNN_squirrel_$l.txt 2>&1
	python -u full-supervised.py --data film 				--layer $l --lr 0.01  --dprate_GPRGNN 0.5 --weight_decay 0    --alpha_GPRGNN 0.9 --model GPRGNN --hidden 64 > GPRGNN_film_$l.txt 2>&1
done
