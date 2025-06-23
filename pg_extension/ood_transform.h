#ifndef  NNGP_TRANSFORM_H
#define NNGP_TRANSFORM_H
#include "utils/fmgroids.h"
#include "ood_configs.h"



typedef struct PredVar {
    Var *v;
    float low, high;
} PredVar;




/* Declare the prototype to keep compiler quiet. */
void out_table_name_list(PlannerInfo *root, Relids ids, FILE* stream);
void out_preds_list(PlannerInfo *root, Relids ids, FILE* stream); 
void out_joinpreds_list(PlannerInfo *root, List *rinfolist, FILE* stream); 
void emit_base_query_str(PlannerInfo *root, RelOptInfo *rel, FILE* stream);
void emit_join_query_str(PlannerInfo *root, RelOptInfo *rel, RelOptInfo *outer_rel, RelOptInfo *inner_rel, SpecialJoinInfo *sjinfo, List *restrictlist, FILE *stream) ;
char *baserel_to_json_query(PlannerInfo *root, RelOptInfo *rel);
char *joinrel_to_json_query(PlannerInfo *root ,
                            RelOptInfo *rel, 
                            RelOptInfo *outer_rel,
                            RelOptInfo *inner_rel,
                            SpecialJoinInfo *sjinfo,
                            List *restrictlist);

inline static Index getVarno(const Var *var) {
    return (var->varno == OUTER_VAR || var->varno == INNER_VAR) ? var->varnoold : var->varno;
}

inline static AttrNumber getVarattno(const Var *var) {
    return (var->varno == OUTER_VAR || var->varno == INNER_VAR) ? var->varoattno : var->varattno;
}


inline static bool varEqual(const Var *v1, const Var *v2) {
    return getVarno(v1) == getVarno(v2) && getVarattno(v1) == getVarattno(v2);
}


/* TODO: Will this cause memory leak? */
static char *get_op_str(Oid funcid) {
    switch (funcid) {
        case F_FLOAT4EQ : 
            return " = ";
        case F_FLOAT4NE :  
            return " != ";
        case F_FLOAT4LT :  
            return " < ";
        case F_FLOAT4LE :  
            return " <= ";
        case F_FLOAT4GT :  
            return " > ";
        case F_FLOAT4GE :  
            return " >= ";
        case F_FLOAT8EQ :  
            return " = ";
        case F_FLOAT8NE :  
            return " != ";
        case F_FLOAT8LT :  
            return " < ";
        case F_FLOAT8LE :  
            return " <= ";
        case F_FLOAT8GT :  
            return " > ";
        case F_FLOAT8GE :  
            return " >= ";
        case F_FLOAT48EQ:   
            return " = ";
        case F_FLOAT48NE:   
            return " != ";
        case F_FLOAT48LT:   
            return " < ";
        case F_FLOAT48LE:   
            return " <= ";
        case F_FLOAT48GT:   
            return " > ";
        case F_FLOAT48GE:   
            return " >= ";
        case F_FLOAT84EQ:   
            return " = ";
        case F_FLOAT84NE:   
            return " != ";
        case F_FLOAT84LT:   
            return " < ";
        case F_FLOAT84LE:   
            return " <= ";
        case F_FLOAT84GT:   
            return " > ";
        case F_FLOAT84GE:   
            return " >= ";
        default:
            return " NULL ";
    }
}

static bool const_is_larger(Oid opfuncid) {
    switch (opfuncid) {
        case F_FLOAT4EQ : 
            elog(ERROR, "We do not support =.");
            return true;
        case F_FLOAT4NE :  
            elog(ERROR, "We do not support !=.");
            return true;
        case F_FLOAT4LT :  
            return true;
        case F_FLOAT4LE :  
            return true;
        case F_FLOAT4GT :  
            return false;
        case F_FLOAT4GE :  
            return false;
        case F_FLOAT8EQ :  
            elog(ERROR, "We do not support =.");
            return true;
        case F_FLOAT8NE :  
            elog(ERROR, "We do not support !=.");
            return true;
        case F_FLOAT8LT :  
            return true;
        case F_FLOAT8LE :  
            return true;
        case F_FLOAT8GT :  
            return false;
        case F_FLOAT8GE :  
            return false;
        case F_FLOAT48EQ:   
            elog(ERROR, "We do not support =.");
            return true;
        case F_FLOAT48NE:   
            elog(ERROR, "We do not support !=.");
            return true;
        case F_FLOAT48LT:   
            return true;
        case F_FLOAT48LE:   
            return true;
        case F_FLOAT48GT:   
            return false;
        case F_FLOAT48GE:   
            return false;
        case F_FLOAT84EQ:   
            elog(ERROR, "We do not support =.");
            return true;
        case F_FLOAT84NE:   
            elog(ERROR, "We do not support !=.");
            return true;
        case F_FLOAT84LT:   
            return true;
        case F_FLOAT84LE:   
            return true;
        case F_FLOAT84GT:   
            return false;
        case F_FLOAT84GE:   
            return false;
        default:
            elog(ERROR, "We do not support %d", opfuncid);
            return true;
    }
}

static void out_table_name(PlannerInfo *root, Var *v, FILE *stream) {
    Index varno = getVarno(v);

    RangeTblEntry *rte = root->simple_rte_array[varno];

    fprintf(stream, "%s", rte->eref->aliasname);
}


static void out_col_name(PlannerInfo *root, Var *v, FILE *stream) {
    Index varno = getVarno(v);
    AttrNumber varattno = getVarattno(v);

    RangeTblEntry *rte = root->simple_rte_array[varno];
    int i = 1;
    ListCell *lc;
    foreach(lc, rte->eref->colnames) {
        if(i == varattno) {
            Value *value = (Value *)lfirst(lc);
            fprintf(stream,  "%s", value->val.str);
            return;
        }
        i++;
    }

    elog(ERROR, "Var invalid: %d.%d", varno, varattno);
    
}




static float get_const(PlannerInfo *root, Const *c, FILE *stream) {
    switch (c->consttype) {
        case 701: 
            return DatumGetFloat8(c->constvalue);
        default:
            /* We should deal with float4 as well, but we do not know
             * the Oid for float4 yet. At this moment, we simply
             * reject all the data type except for 701(float8).
             */
            elog(ERROR, "We do not support datatype %d", c->consttype);
    }
}


inline static void out_float(float fv, FILE *stream) {
    fprintf(stream, "%f", fv);
}

inline static void out_const(PlannerInfo *root, Const *c, FILE *stream ){
    fprintf(stream, "%f", get_const(root, c, stream));
}


/*
static void out_expr(PlannerInfo *root, Expr *expr, FILE *stream) {
    if(IsA(expr, Var)) {
        out_col_name(root, (Var *)expr, stream);
    }
    else if(IsA(expr, Const)) {
        out_const(root, (Const *)expr, stream);
    }
    else {
        // pprint(expr);
        elog(ERROR, "We do not support type %d", expr->type);
    }
}
*/


inline static void out_op(Oid funcid, FILE *stream) {
    fprintf(stream, "%s", get_op_str(funcid));
}

inline static void out_comma(FILE *stream) {
    fprintf(stream, ",");
}

inline static void out_at(FILE *stream) {
    fprintf(stream, "@");
}

inline static void out_sharp(FILE *stream) {
    fprintf(stream, "#");
}


/* TODO: Transform the parse tree to a self-defined query tree, so that 
 * restrictinfo_list will be convert to string format (e.g., "r1.a > 3")
 * only once. 
 */

void out_table_name_list(PlannerInfo *root, Relids ids, FILE* stream) {
    bool isfirst =true;
    int x = -1;
    while(( x = bms_next_member(ids, x)) >= 0) {
        RangeTblEntry *rte = root->simple_rte_array[x];

        if(!isfirst) {
            out_comma(stream);
        }

        fprintf(stream, "%s", rte->eref->aliasname);
        isfirst = false;
    }
}

/* CAUTION: predicate order must corresponds to table_name_list */
void out_preds_list(PlannerInfo *root, Relids ids, FILE* stream) {

    List *baserlist = NULL;
    List *predlist = NULL;
    ListCell *lc;
    int x;
    bool isfirst = true;

    x = -1;
    while(( x = bms_next_member(ids, x)) >= 0) {
        bool isPredFirst;
        RelOptInfo *rel = root->simple_rel_array[x];
        baserlist = rel->baserestrictinfo;
        if (!isfirst) {
            out_at(stream);
        }
        foreach(lc, baserlist) {
            RestrictInfo *rinfo = (RestrictInfo *)lfirst(lc);
            Node *clause = NULL;
            if(rinfo->orclause) {
                clause = (Node *) rinfo->orclause;
                elog(ERROR, "We do not support OR.");
            }
            else 
                clause = (Node *)rinfo->clause;

            if(clause != NULL && IsA(clause, OpExpr)) {
                OpExpr *opclause = (OpExpr *)clause;
                Oid opfuncid = opclause->opfuncid;

                Expr *e1 = linitial_node(Expr, opclause->args);
                Expr *e2 = lsecond_node(Expr, opclause->args);

                Var *v = (Var *)e1;
                float fv = get_const(root, (Const *)e2, stream);

                /* Locate the position of v. */
                ListCell *lc;
                PredVar *ipv = NULL;

                foreach(lc, predlist) {
                    PredVar *pv = (PredVar *)lfirst(lc);
                    if(varEqual(v, pv->v)) {
                        ipv = pv;
                        break;
                    }
                }

                if(!ipv) {
                    ipv =  (PredVar *)palloc0(sizeof(PredVar));
                    ipv->v = v;
                    predlist = lappend(predlist, ipv);
                }

                if(const_is_larger(opfuncid)) {
                    ipv->high = fv;
                }
                else {
                    ipv->low = fv;
                }
            }
        }
        isPredFirst = true;
        foreach(lc, predlist) {
            PredVar *pv = (PredVar *)lfirst(lc);
            if(!isPredFirst) {
                out_sharp(stream);
            }
            out_col_name(root, pv->v, stream);
            out_comma(stream);
            out_float(pv->high, stream);
            out_comma(stream);
            out_float(pv->low, stream);
            isPredFirst = false;
        }
        list_free_deep(predlist);
        predlist = NULL;
        isfirst = false;
    }
}

void out_joinpreds_list(PlannerInfo *root, List *rinfolist, FILE* stream) {
    ListCell *lc;
    bool isfirst = true;
    foreach(lc, rinfolist) {
        RestrictInfo *rinfo = (RestrictInfo *)lfirst(lc);
        Node *clause = NULL;
        if(!isfirst) {
            out_sharp(stream);
        }
        if(rinfo->orclause) {
            clause = (Node *) rinfo->orclause;
            elog(ERROR, "We do not support OR.");
        }
        else 
            clause = (Node *)rinfo->clause;

        if(clause != NULL && IsA(clause, OpExpr)) {
            OpExpr *opclause = (OpExpr *)clause;
            Expr *e1 = linitial_node(Expr, opclause->args);
            Expr *e2 = lsecond_node(Expr, opclause->args);
            out_table_name(root, (Var *)e1, stream);
            out_comma(stream);
            out_table_name(root, (Var *)e2, stream);
            out_comma(stream);
            out_col_name(root, (Var *)e1, stream);
        }
        isfirst = false;
    }
}

void emit_base_query_str(PlannerInfo *root, RelOptInfo *rel, FILE* stream) {
    fprintf(stream, "\"");


    /* table_name_list */
    out_table_name_list(root, rel->relids, stream);
    out_at(stream);

    /* preds_list */
    out_preds_list(root, rel->relids, stream);
    out_at(stream);

    /* join_conditions */
    out_joinpreds_list(root, NULL, stream);


    fprintf(stream, "\"");
}

void emit_join_query_str(PlannerInfo *root, RelOptInfo *rel, RelOptInfo *outer_rel, RelOptInfo *inner_rel, SpecialJoinInfo *sjinfo, List *restrictlist, FILE *stream) 
{ 
    fprintf(stream, "\"");

    /* table_name_list */   
    out_table_name_list(root, rel->relids, stream);
    out_at(stream);


    /* preds_list */
    out_preds_list(root, rel->relids, stream);
    out_at(stream);

    /* join_conditions */
    out_joinpreds_list(root, restrictlist, stream);

    fprintf(stream, "\"");
}

char *baserel_to_json_query(PlannerInfo *root, RelOptInfo *rel) {
    char *buf;
    size_t json_size;
    FILE* stream;

    stream = open_memstream(&buf, &json_size);
    fprintf(stream, "{\"Plan\": ");
    emit_base_query_str(root, rel, stream);
    fprintf(stream, "}\n");
    fclose(stream);
    return buf;
}

char *joinrel_to_json_query(PlannerInfo *root ,
                            RelOptInfo *rel, 
                            RelOptInfo *outer_rel,
                            RelOptInfo *inner_rel,
                            SpecialJoinInfo *sjinfo,
                            List *restrictlist)
{
    char *buf;
    size_t json_size;
    FILE* stream;

    stream = open_memstream(&buf, &json_size);
    fprintf(stream, "{\"Plan\": ");
    emit_join_query_str(root, rel, outer_rel, inner_rel, sjinfo, restrictlist, stream);
    fprintf(stream, "}\n");
    fclose(stream);
    return buf;
}

#endif
