# pg_dropcache

`pg_dropcache` is a PostgreSQL extension that invalidates `shared_buffers` cache

## Installation

To install `pg_dropcache` clone this repository and run:

```
make install USE_PGXS=1
```

Then in psql (or any other client) execute:

```
create extension pg_dropcache;
```

## Usage

**WARNING**: Dirty pages will be just dropped, therefore they won't be flushed on the disk! It should be used with extreme caution!

To clear whole buffer cache run:

```
select pg_dropcache();
```

To clear cache buffers for just a single relation:

```
select pg_drop_rel_cache(<relation>);
```

If you need to clear a specific buffer cache, you can specify it as second parameter:

```
select pg_drop_rel_cache(<relation>, <fork>);
```

`fork` can have one of the following values:
* 'main'
* 'vm'
* 'fsm'
* 'init'

Have fun!