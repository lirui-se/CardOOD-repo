#include <sys/types.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <arpa/inet.h>
#include <math.h>

#include "ood_configs.h"
#include "ood_util.h"
#include "ood_hmap.h"
#include "postgres.h"
#include "fmgr.h"
#include "nodes/print.h"
#include "parser/parsetree.h"
#include "executor/executor.h"
#include "optimizer/planner.h"
#include "utils/guc.h"
#include "commands/explain.h"
#include "tcop/tcopprot.h"

#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"

/* include ood_tranform.h at last so that it inherents the above .h files. */
#include "ood_transform.h"
#include "ood_plantree.h"


static bool enable_ood;
static bool enable_ood_mix;
static int ood_tables_lb;
static int ood_tables_ub;

PG_MODULE_MAGIC;
void _PG_init(void);
void _PG_fini(void);


static planner_hook_type prev_planner = NULL;

static PlannedStmt *ood_planner(Query *parse, 
                                int cursorOptions, 
                                ParamListInfo boundParams) ;


static void ood_set_joinrel_size_estimates(PlannerInfo *root, 
                                            RelOptInfo *rel,
                                            RelOptInfo *outer_rel,
                                            RelOptInfo *inner_rel,
                                            SpecialJoinInfo *sjinfo,
                                            List *restrictlist);

static double ood_ret_baserel_size_estimates (PlannerInfo *root, 
                                    RelOptInfo *rel,
                                   List *clauses,
                                   int varRelid,
                                   JoinType jointype,
                                   SpecialJoinInfo *sjinfo) ;


void _PG_init(void) {
  // install each ood hook
  // selectivity_hook = ood_selectivity;
  //
  elog(LOG, "pg_ood::_PG_init is called.");

  prev_planner = planner_hook;
  planner_hook = ood_planner;
  joinrel_size_estimates_hook = ood_set_joinrel_size_estimates;
  ret_baserel_size_estimates_hook = ood_ret_baserel_size_estimates;


  // define Bao user-visible variables
  DefineCustomBoolVariable(
    "pg_ood.enable_ood",
    "Enable the ood estimator",
    "Enables the ood estimator. When enabled, the cardinality is estimated by ood extension.",
    &enable_ood,
    true,
    PGC_USERSET,
    0,
    NULL, NULL, NULL);

  DefineCustomBoolVariable(
    "pg_ood.enable_ood_mix",
    "Enable the ood mix estimator",
    "Enables the ood mix estimator. When enabled, table number <= 2 are estimated by pg. ",
    &enable_ood_mix,
    true,
    PGC_USERSET,
    0,
    NULL, NULL, NULL);

  DefineCustomIntVariable(
    "pg_ood.ood_tables_lb",
    "Number of tables over which PG estimation is used.",
    "Number of tables over which PG estimation is used.",
    &ood_tables_lb,
    1, 1, 65536,
    PGC_USERSET,
    0,
    NULL, NULL, NULL);

  DefineCustomIntVariable(
    "pg_ood.ood_tables_ub",
    "Number of tables under which PG estimation is used.",
    "Number of tables under which PG estimation is used.",
    &ood_tables_ub,
    2, 1, 65536,
    PGC_USERSET,
    0,
    NULL, NULL, NULL);


    DefineCustomStringVariable(
    "pg_ood.ood_host",
    "Nngp server host", NULL,
    &ood_host,
    "localhost",
    PGC_USERSET,
    0,
    NULL, NULL, NULL);

  DefineCustomIntVariable(
    "pg_ood.ood_port",
    "Nngp server port", NULL,
    &ood_port,
    8001, 1, 65536, 
    PGC_USERSET,
    0,
    NULL, NULL, NULL);

}

void _PG_fini(void) {
    planner_hook = prev_planner;
  elog(LOG, "finished extension");
}


static PlannedStmt *planner_call_back(Query *parse, int cursorOptions, ParamListInfo boundParams) {
    if (prev_planner) 
        return prev_planner(parse, cursorOptions, boundParams);
    else 
        return standard_planner(parse, cursorOptions,  boundParams);
}

static PlannedStmt *ood_planner(Query *parse, 
                                int cursorOptions, 
                                ParamListInfo boundParams) {
    int conn_fd;
    char *json;
    char *json2;

    int bufsize;
    int bufsizeb;
    double *cardvarbuf;
    int varoffset;
    QueryExt *ext;
    PlannedStmt *stdres ;
    int read_buf_size;

    if(!enable_ood) {
        return planner_call_back(parse, cursorOptions, boundParams);
    }
    if(!parse->jointree) {
        return planner_call_back(parse, cursorOptions, boundParams);
    }
    if(!parse->jointree->fromlist) {
        return planner_call_back(parse, cursorOptions, boundParams);
    }
    if(!parse->jointree->quals) {
        return planner_call_back(parse, cursorOptions, boundParams);
    }

    ext = (QueryExt *)palloc0(sizeof(QueryExt));
    ext->htab = NULL;
    ext->empty = true;
    parse->ext = (void *)ext;
    

    // elog(NOTICE, "Entering ood_planner.");
    /* Make connection with python sever. pass the query string. */
    conn_fd = connect_to_ood(ood_host, ood_port);
    if (conn_fd < 0) {
        elog(WARNING, "Unable to connect to NNGP server.");
        return planner_call_back(parse, cursorOptions, boundParams);
    }
    json = query_to_json(parse->qstr);
    json2 = relid_to_json(parse->rtable);

    write_all_to_socket(conn_fd, START_QUERY_MESSAGE);
    write_all_to_socket(conn_fd, json);
    write_all_to_socket(conn_fd, json2);
    write_all_to_socket(conn_fd, TERMINAL_MESSAGE);
    
    shutdown(conn_fd, SHUT_WR);
    
    /* Now, accept a float array, each element indicates a cardinality estimation for a subquery. */

    bufsize  = (int)pow(2, list_length(parse->rtable)) - 1;
    bufsizeb = 2 * bufsize * sizeof(double);
    cardvarbuf = (double *)palloc0(bufsizeb);
    varoffset = bufsize;

    read_buf_size = read(conn_fd, cardvarbuf, bufsizeb);
    if(read_buf_size != bufsizeb) {
        shutdown(conn_fd, SHUT_RDWR);
        elog(WARNING, "PostgreSQL could not read the response from the sever. Size: %d", read_buf_size);
        memset(cardvarbuf, 0, bufsizeb);
    }
    else {
        Bitmapset *subset, *fullset;
        int *s, *indices;
        int sublen, size;
        int ind;
        shutdown(conn_fd, SHUT_RDWR);
        elog(NOTICE, "PostgreSQL successfully reads the response from the sever. Trying to parse it.");


        /* Create a hash table. */
        ext->htab = HMapCreate();
        ext->empty = false;

        /* Push cardinality info into hash table. */
        subset = NULL;
        fullset = NULL;
        s = NULL;
        indices = NULL;
        sublen = 0;
        size = 0;
        ind = 0;
        for(ind = 1; ind <= list_length(parse->rtable); ind++) 
            fullset = bms_add_member(fullset, ind);
        CreateSubsetEnu(fullset, &s, &indices, &sublen, &size);
        nextLSubset(&subset, s, indices, &sublen, size);
        // elog(NOTICE, "Log card.");
        // for(int i = 0; i < bufsize; i++) {
        //     elog(NOTICE, "%lf", cardvarbuf[i]);
        // }
        // elog(NOTICE, "Log var.");
        // for(int i = 0; i < bufsize; i++) {
        //     elog(NOTICE, "%lf", cardvarbuf[i + varoffset]);
        // }

        for(int i = 0; i < bufsize; i++) {


            /* Add the original value into the map. */
            HMapAddCardVar(ext->htab, subset, cardvarbuf[i], cardvarbuf[i + varoffset] );
            nextLSubset(&subset, s, indices, &sublen, size);
        }

        /* Try to query the hash map. */
        // Bitmapset *temp = NULL;
        // temp = bms_add_member(temp, 1);
        // temp = bms_add_member(temp, 2);
        // double card = HMapGetCard(ext->htab, temp);
        // double var = HMapGetVar(ext->htab, temp);
        // elog(NOTICE, "Query Card = %lf",  card);
        // elog(NOTICE, "Query Var = %lf",  var);
    }

    stdres = planner_call_back(parse, cursorOptions, boundParams);
    /* Store some necessary information. */
    stdres->planner_parse = (Query *)parse;
    return stdres;
}




static void ood_set_joinrel_size_estimates(PlannerInfo *root, 
                                            RelOptInfo *rel,
                                            RelOptInfo *outer_rel,
                                            RelOptInfo *inner_rel,
                                            SpecialJoinInfo *sjinfo,
                                            List *restrictlist) 
{
    QueryExt *ext = (QueryExt *)root->parse->ext;
    HTAB *htable;
    double card;
    // double var;

    if (!enable_ood || ext->empty || (enable_ood_mix && bms_num_members(rel->relids) <= ood_tables_ub && bms_num_members(rel->relids) >= ood_tables_lb )) {
        standard_set_joinrel_size_estimates(root, rel, outer_rel, inner_rel, sjinfo, restrictlist);
        return ;
    }

    htable = ext->htab;
    card = HMapGetCard(htable, rel->relids);
    // var = HMapGetVar(htable, rel->relids);

    if(card <= 0) { 
        card = 0.1;
    }
    card = pow(2, card);
    rel->rows = card;
}

static double ood_ret_baserel_size_estimates (PlannerInfo *root, 
                                    RelOptInfo *rel,
                                   List *clauses,
                                   int varRelid,
                                   JoinType jointype,
                                   SpecialJoinInfo *sjinfo) 

{
    QueryExt *ext = (QueryExt *)root->parse->ext;
    HTAB *htable;
    double card;
    // double var;
    double final_card;
    if(!enable_ood || ext->empty || (enable_ood_mix && 1 <= ood_tables_ub && 1 >= ood_tables_lb)) {
        return  rel->tuples * clauselist_selectivity(root, 
                                                    clauses,
                                                    varRelid,
                                                    jointype,
                                                    sjinfo);

    }
    htable = ext->htab;
    card = HMapGetCard(htable, rel->relids);
    // var = HMapGetVar(htable, rel->relids);
    final_card = -1;
    if(card < 0) {
        card = 0.1;
    }
    card = pow(2, card);
    final_card =  card;
    return final_card;
}



