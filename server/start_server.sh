home=/home/lirui/codes/SIGMOD2025CardOOD
# python3 -u main.py --schema=imdb --datapath="${home}/CardDATA/imdb_clean2"  --querypath="${home}/CardDATA/job-light-enrich/${qset}/${split}" --model="${model}"  --is_join_query='True' --skew_split_keys="${split}" --model_type="${model_type}" --num_negs="${neg}"  --epochs="${epoch}" --learning_rate="${rate}" --batch_size="${batch}" --model_save_path="${model_path}" --predict_file="${predict_file}" --config="${config_file}"
cd $home/ood-src
python3 setup.py install
cd $home/ceb-src
python3 setup.py install
cd $home/server

CUDA_VISIBLE_DEVICES=0  python3 -u main.py \
        --schema=imdb \
        --datapath=$home/data/imdb \
        --querypath=$home/query/job-light/all \
        --model=ttt \
        --is_join_query='True' \
        --skew_split_keys=template_no  \
        --model_type=MSCN \
        --num_negs=3  \
        --epochs=80  \
        --learning_rate=5e-5 \
        --batch_size=64 \
        --model_save_path=$home/server/models \
        --predict_file=predictions.txt  \
        --train_sql_path=$home/query/job-light/train \
        --test_sql_path=$home/query/job-light/test \
