#ifndef NNGP_PLANTREE_H
#define NNGP_PLANTREE_H


#include "nodes/plannodes.h"



typedef struct NNGPTree {
    Bitmapset *ids;
    double rows;
    double startup_ms, total_ms;
    struct NNGPTree *l, *r;
} NNGPTree;

Bitmapset * construct_tree_recur(PlanState *ps, NNGPTree **node);
NNGPTree *construct_tree_from_planstate(PlanState *ps);
void dumpbms(FILE *fout, Bitmapset *b);
void dumpNNGPTree_recur(FILE *fout, NNGPTree *t, int d);
void dumpNNGPTree(NNGPTree *t);
void register_subqueries(QueryExt *ext, NNGPTree *t);
Bitmapset * construct_tree_recur(PlanState *ps, NNGPTree **node) {
    double nloops;
    Plan *plan; 
    if(!ps) return NULL;
    nloops = ps->instrument->nloops;
    plan = ps->plan;
    (*node) = (NNGPTree *)palloc0(sizeof(NNGPTree));
    (*node)->l = NULL;
    (*node)->r = NULL;
    (*node)->rows = ps->instrument->ntuples / nloops;
    (*node)->startup_ms = 1000.0 * ps->instrument->startup / nloops;
    (*node)->total_ms = 1000.0 * ps->instrument->total / nloops;
    switch(nodeTag(plan)) {
        case T_SeqScan:
        case T_SampleScan:
        case T_BitmapHeapScan:
        case T_TidScan:
        case T_SubqueryScan:
        case T_FunctionScan:
        case T_TableFuncScan:
        case T_ValuesScan:
        case T_CteScan:
        case T_WorkTableScan:
        case T_ForeignScan:
        case T_CustomScan:
        case T_IndexScan:
        case T_IndexOnlyScan:
        case T_BitmapIndexScan: 
                {
                    Scan *splan = (Scan *)plan;
                    (*node)->ids = bms_make_singleton(splan->scanrelid);
                }
                break;
        default:
                (*node)->ids = bms_union(
                    construct_tree_recur(outerPlanState(ps), &((*node)->l)), 
                    construct_tree_recur(innerPlanState(ps), &((*node)->r))
                );
                break;
    }
    
    return (*node)->ids;
}

NNGPTree *construct_tree_from_planstate(PlanState *ps) {
    NNGPTree *t = (NNGPTree *)palloc0(sizeof(NNGPTree));
    t = NULL;
    construct_tree_recur(ps, &t);
    return t;
}

void dumpbms(FILE *fout, Bitmapset *b) {
    int x = -1;
    fprintf(fout, "( ");
    while((x = bms_next_member(b, x)) >= 0) {
        fprintf(fout, "%d ", x);
    }
    fprintf(fout, " )");
    
}

void dumpNNGPTree_recur(FILE *fout, NNGPTree *t, int d) {
    if(!t) return;
    for(int i = 0; i < d * 4; i++) {
        fprintf(fout, " ");
    }
    fprintf(fout, "row = %lf, t = %lf", t->rows, t->total_ms);
    fprintf(fout, ", relids = ");
    dumpbms(fout, t->ids);
    fprintf(fout, "\n");
    dumpNNGPTree_recur(fout, t->l, d + 1);
    dumpNNGPTree_recur(fout, t->r, d + 1);

}

void dumpNNGPTree(NNGPTree *t) {
    FILE *fout  = fopen("/home/kfzhao/debugpsql/dump.log", "w");
    if(!fout)  {
        elog(ERROR, "Can not open /home/kfzhao/debugpsql/dump.log.");
    }
    dumpNNGPTree_recur(fout, t, 0);
    fflush(fout);
}



void register_subqueries(QueryExt *ext, NNGPTree *t) {
#ifdef DUMP_CHOSEN_PLAN
    HTAB *vs_htable;
    int ind;
    if(!t) return;
    vs_htable = ext->vs_htab;
    ind = (int)HMapGetCard(vs_htable, t->ids);
    ext->valid_subqueries[ind] = 1;
    ext->collected_cards[ind] = t->rows;

    register_subqueries(ext, t->l);
    register_subqueries(ext, t->r);
#endif
}

#endif
