# CardOOD_repo
## Unzip the folders
```shell
cat data.tar.zip.part_* > data.tar.zip
for f in *.zip; do
    unzip $f
done
for f in *.tar; do
    tar -xf $f
done
```


## Step 1: Install conda environment.
```shell
conda env create -f ood.yaml
```

## Step 2: Compile Postgresql & prepare the datasets.
```shell
cd path-to-repo
mkdir postgres
mkdir data
cd postgresql-12.4
./configure --prefix=/path-to-repo/postgres CFLAGS="-O3" CXXFLAGS="-O3"
make
make install
../postgres/bin/initdb -D ../data
echo "shared_preload_libraries=pg_ood" >> ../data/postgresql.conf
```

## Step 3: Install Postgresql extension.
```shell
cd path-to-repo/pg_extension
make 
make USE_PGXS=1 install
../postgres/bin/pg_ctl start
```

## Step 4: Install ood & ceb src.
```shell
cd path-to-repo/ood-src
python3 setup.py install

cd path-to-repo/ceb-src
python3 setup.py install
```
## Step 5: Start python server.
```shell
cd path-to-repo/server
./start_server.sh
```

## Step 6: Run queries on pg.
CardOOD/postgres/psql imdb -f CardOOD/query/job-light/t70.txt.sql


