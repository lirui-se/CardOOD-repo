#ifndef NNGP_HMAP_H
#define NNGP_HMAP_H
#include "utils/hsearch.h"
#include "nodes/bitmapset.h"
#include "ood_configs.h"

typedef struct QueryExt {
    bool empty;
    HTAB *htab;
} QueryExt;

typedef struct CardEntry {
    /* The first field is necessary! */
    Bitmapset *ids;
    double card;
    double var;
} CardEntry;

HTAB *HMapCreate(void);

double HMapGetCard(HTAB *hashtab, Bitmapset *relids);
double HMapGetVar(HTAB *hashtab, Bitmapset *relids);
const CardEntry *HMapGetEntry(HTAB *hashtab, Bitmapset *relids);
void HMapAddCardVar(HTAB *hashtab, Bitmapset *ids, double card, double var);

HTAB *HMapCreate() {
    HASHCTL hash_ctl;
    MemSet(&hash_ctl, 0, sizeof(hash_ctl));
    hash_ctl.keysize = sizeof(Bitmapset *);
    hash_ctl.entrysize = sizeof(CardEntry);
    hash_ctl.hash = bitmap_hash;
    hash_ctl.match = bitmap_match;
    hash_ctl.hcxt = CurrentMemoryContext;

    return hash_create(
                    "CardMap", 
                    256L, 
                    &hash_ctl, 
                    HASH_ELEM | HASH_FUNCTION | HASH_COMPARE | HASH_CONTEXT);
}


double HMapGetCard(HTAB *hashtab, Bitmapset *relids) {
    CardEntry *entry = (CardEntry *)hash_search(hashtab, &relids, HASH_FIND, NULL);
    if(entry) return entry->card;
    else return -1;
}

double HMapGetVar(HTAB *hashtab, Bitmapset *relids) {
    CardEntry *entry = (CardEntry *)hash_search(hashtab, &relids, HASH_FIND, NULL);
    if(entry) return entry->var;
    else return -1;
}

const CardEntry *HMapGetEntry(HTAB *hashtab, Bitmapset *relids) {
    CardEntry *entry = (CardEntry *)hash_search(hashtab, &relids, HASH_FIND, NULL);
    if(entry) return entry;
    else return NULL;
}

/*
void HMapAddCard(HTAB *hashtab, Bitmapset *ids, double card) {
    bool found;
    Bitmapset *tempr = bms_copy(ids);
    CardEntry *entry = (CardEntry *)hash_search(hashtab, &tempr, HASH_ENTER, &found);
    entry->card = card;
}
*/

void HMapAddCardVar(HTAB *hashtab, Bitmapset *ids, double card, double var) {
    bool found;
    Bitmapset *tempr = bms_copy(ids);
    CardEntry *entry = (CardEntry *) hash_search(hashtab, &tempr, HASH_ENTER, &found);
    entry->card = card;
    entry->var = var;
}
                

#endif 
