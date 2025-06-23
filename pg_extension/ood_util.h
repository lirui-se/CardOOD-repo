#ifndef NNGP_UTIL_H
#define NNGP_UTIL_H

#include <arpa/inet.h>
#include <unistd.h>
#include "postgres.h"
#include "fmgr.h"
#include "nodes/parsenodes.h"
#include "ood_configs.h"


static const char* START_QUERY_MESSAGE = "{\"type\": \"query\"}\n";
static const char *START_CALLBACK_MESSAGE = "{\"type\": \"valid\"}\n";
static const char *START_COLLECT_MESSAGE = "{\"type\": \"chosen\"}\n";
static const char* TERMINAL_MESSAGE = "{\"final\": true}\n";


char *query_to_json(const char *q); 
char *relid_to_json(const List *rtable);
char *int_array_to_json(int *a, int num);
char *double_array_to_json(double *a, int num);
void CreateSubsetEnu(const Bitmapset *rids, int **s, int **indices, int *sublen, int *size ); 
void nextLSubset(Bitmapset **subset, int *s, int *indices, int *psublen, int size);

/* Copied from bao_util.h. */
char *query_to_json(const char *q) {
    char *buf;
    size_t json_size;
    FILE* stream;

    stream = open_memstream(&buf, &json_size);
    fprintf(stream, "{\"Plan\": ");
    fprintf(stream, "\"");
    fprintf(stream, "%s", q);
    fprintf(stream, "\"");
    fprintf(stream, "}\n");

    fclose(stream);
    return buf;
}

char *relid_to_json(const List *rtable) {
    char *buf;
    size_t json_size;
    FILE *stream;
    ListCell *lc;
    int rti = 1;

    stream = open_memstream(&buf, &json_size);
    fprintf(stream, "{\"Relidmap\": ");
    fprintf(stream, "{");
    foreach(lc, rtable) {
        RangeTblEntry *rte = (RangeTblEntry *)lfirst(lc);
        if(rti > 1) {
            fprintf(stream, " ,");
        }
        fprintf(stream, "\"%s\": %d", rte->eref->aliasname, rti);
        rti++;
    }
    fprintf(stream, "}");
    fprintf(stream, "}\n");
    fprintf(stream, "{\"Revrelidmap\": ");
    fprintf(stream, "{");

    rti = 1;
    foreach(lc, rtable) {
        RangeTblEntry *rte = (RangeTblEntry *)lfirst(lc);
        if(rti > 1) {
            fprintf(stream, ", ");
        }
        fprintf(stream, "\"%d\": \"%s\"", rti, rte->eref->aliasname);
        rti++;
    }
    fprintf(stream, "}");
    fprintf(stream, "}\n");

    fclose(stream);
    return buf;
}

char *int_array_to_json(int *a, int num) {
    char *buf;
    size_t json_size;
    FILE *stream;

    stream = open_memstream(&buf, &json_size);
    fprintf(stream, "{\"ValidSubqueries\": ");
    fprintf(stream, "[");
    for(int i = 0; i < num ; i++) {
        if(i > 0) {
            fprintf(stream, ", ");
        }
        fprintf(stream, "%d", a[i]);
    }
    
    fprintf(stream, "]");
    fprintf(stream, "}\n");
    fclose(stream);
    return buf;
   
}

char *double_array_to_json(double *a, int num) {
    char *buf;
    size_t json_size;
    FILE *stream;

    stream = open_memstream(&buf, &json_size);
    fprintf(stream, "{\"ChosenSubqueries\": ");
    fprintf(stream, "[");
    for(int i = 0; i < num; i++) {
        if(i > 0) {
            fprintf(stream, ", ");
        }
        fprintf(stream, "%lf", a[i]);
    }

    fprintf(stream, "]");
    fprintf(stream, "}\n");
    fclose(stream);
    return buf;
}

void CreateSubsetEnu(const Bitmapset *rids, int **s, int **indices, int *sublen, int *size ) {
    int x = -1;
    int i = 0;
    int num = bms_num_members(rids);
    *indices = (int *)palloc0(num * sizeof(int));
    *s = (int *)palloc0(num * sizeof(int));
    while((x = bms_next_member(rids, x)) >= 0) {
        (*s)[i] = x;
        i++;
    }
    *sublen = 0;
    *size = num;
}

void nextLSubset(Bitmapset **subset, int *s, int *indices, int *psublen, int size ) {
    for(int k = *psublen  - 1; k > -1; k--) {
        if(indices[k] < size - (*psublen - k - 1) - 1) {
            int g;
            *subset = bms_del_member(*subset, s[indices[k]]);
            indices[k]++;
            *subset = bms_add_member(*subset, s[indices[k]]);
            g = k + 1;
            while(g < *psublen) {
                *subset = bms_del_member(*subset, s[indices[g]]);
                indices[g] = indices[g - 1] + 1;
                *subset = bms_add_member(*subset, s[indices[g]]);
                g++;
            }
            return;
        }
    }

    /* Time for larger set. */
    (*psublen)++;
    if(*psublen > size) *psublen = 0;
    bms_free(*subset);
    *subset = NULL;
    memset(indices, 0, size * sizeof(int));
    for(int i = 0; i < *psublen; i++) {
        indices[i] = i;
        *subset = bms_add_member(*subset, s[i]);
    }
}


static void write_all_to_socket(int conn_fd, const char* json) {
  size_t json_length;
  ssize_t written, written_total;
  json_length = strlen(json);
  written_total = 0;
  
  while (written_total != json_length) {
    written = write(conn_fd,
                    json + written_total,
                    json_length - written_total);
    written_total += written;
  }
}

// Connect to the ood server.
static int connect_to_ood(const char* host, int port) {
  int ret, conn_fd;
  struct sockaddr_in server_addr = { 0 };

  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  inet_pton(AF_INET, host, &server_addr.sin_addr);
  conn_fd = socket(AF_INET, SOCK_STREAM, 0);
  // conn_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (conn_fd < 0) {
    return conn_fd;
  }
  
  ret = connect(conn_fd, (struct sockaddr*)&server_addr, sizeof(server_addr));
  if (ret == -1) {
    return ret;
  }

  return conn_fd;

}



#endif
