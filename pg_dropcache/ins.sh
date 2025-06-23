make USE_PGXS=1 clean
make USE_PGXS=1
pg_ctl stop
make install USE_PGXS=1
pg_ctl stop
pg_ctl start 
psql --port 5433 -U lirui -d kf_job -c "create extension pg_dropcache;"
#  psql -U kfzhao -d kf_tpcds -c "create extension pg_dropcache;"
#  psql -U kfzhao -d tpcds2G -c "create extension pg_dropcache;"
