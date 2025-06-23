import sys, os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append("/home/lirui/codes/PG_CardOOD/CardOOD/ood")
import matplotlib.pyplot as plt
import random
import math

from encoder.transform import JoinQueryDataset, OnehotJoinQueryEncoderKMeans, DnnJoinQueryEncoder
from db.parser import JoinQueryParser
from db.schema import load_schema
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.special import rel_entr
import torch
import numpy as np
import io

schema_name = "imdb_simple"
schema_data_path = "/home/lirui/codes/PG_CardOOD/CardDATA/imdb_clean2"
# source_path = ""
#  source_file = "/home/lirui/codes/PG_CardOOD/CardDATA/job-light-enrich/Jul-25/job.txt"
#  dest_path = "/home/lirui/codes/PG_CardOOD/CardDATA/job-light-enrich/Jul-25/distri"
#  
#  # sql_source_path = "/home/lirui/codes/PG_CardOOD/CardDATA/job-light-enrich/Jul-25/job-light-ext.sql"
#  sql_source_path = "/home/lirui/codes/PG_CardOOD/CardDATA/job-light-enrich/Jul-25/template_no_sql"
#  sql_dest_path = "/home/lirui/codes/PG_CardOOD/CardDATA/job-light-enrich/Jul-25/distri_sql"
template_no_path = "./template_no"
template_no_sql_path = "./template_no_sql"

distri_path = "./distri"
distri_sql_path = "./distri_sql"

n_pca_component = 4
n_kmeans_clusters = 100
n_distribution_clusters = 10
n_bin = 10
n_buckets = 10

all_queries, all_cards = list(), list()
all_query_infos = list()

LOAD_FROM_FILE = False

template = 70
if not LOAD_FROM_FILE:
    schema = load_schema(schema_name, schema_data_path)
    query_parser = JoinQueryParser(schema=schema)
    encoder = DnnJoinQueryEncoder(schema=schema)
    encoder_type = 'onehot'
    feat_dim = None
    encoding_list = []
    # pred_feat_dim, table_feat_dim, join_feat_dim = encoder.pred_feat_dim, encoder.table_feat_dim, encoder.join_feat_dim
    # pred_num_hid, table_num_hid, join_num_hid = 64, 64, 64
    # pred_num_out, table_num_out, join_num_out = 64, 64, 64
    # max1, max2, max3 = -1, -1, -1
    all_query_lines = []
    all_sql_lines = []

    for i in range(template):
        with open(template_no_sql_path + "/q" + str(i) + ".sql", "r") as in_file:
            for line in in_file:
                all_sql_lines.append(line.strip("\n"))

    for i in range(template):
        with open(template_no_path + "/q" + str(i) + ".txt", "r") as in_file:
            for line in in_file:
                all_query_lines.append( line.strip("\n") )
                table_ids, all_pred_list, join_infos, card = query_parser.parse_line(line)
                x = encoder(table_ids, all_pred_list, join_infos)
                # print(x.shape)
                # print(x[0].shape, x[1].shape, x[2].shape)
                # size1 = x[0].shape[0] * x[0].shape[1] 
                # size2 = x[1].shape[0] * x[1].shape[1]
                # size3 = x[2].shape[0] * x[2].shape[1]
                # if size1 > max1:
                #     max1 = size1
                # if size2 > max2:
                #     max2 = size2
                # if size3 > max3:
                #     max3 = size3
                # encoding_list.append( torch.cat( [ x[0].reshape(1, size1), x[1].reshape(1, size2), x[2].reshape(1, size3) ], dim = -1) )
                # encoding_list.append(  (x[0].reshape(1, size1), x[1].reshape(1, size2), x[2].reshape(1, size3)) )
                encoding_list.append(x)
    # arr = np.ones(shape=(len(encoding_list), size1 + size2 + size3))
    # x_train = torch.from_numpy(np.array(encoding_list))
    # encoding_list = [ torch.cat( [torch.nn.functional.pad( x[0], pad=(0, max1 - x[0].shape[0] * x[0].shape[1], 0, 0),mode='constant', value=0 ),
    #                              torch.nn.functional.pad( x[1], pad=(0, max2 - x[1].shape[0] * x[1].shape[1], 0, 0),mode='constant', value=0 ),
    #                              torch.nn.functional.pad( x[2], pad=(0, max3 - x[2].shape[0] * x[2].shape[1], 0, 0),mode='constant', value=0 ),
    #                             ], dim = -1) for x in encoding_list ]
    torch.save(encoding_list, 'tensor.pt')
    fout = open("all_query_lines", "w")
    fout.write( "\n".join(all_query_lines) + "\n" )
    fout.close()
else:
    all_query_lines = []
    all_sql_lines = []

    fin = open("all_query_lines", "r")
    fin2 = open("all_sql_lines", "r")

    all_query_lines = fin.readlines()
    all_sql_lines = fin2.readlines()

    all_query_lines = [ l.strip("\n") for l in all_query_lines ]
    all_sql_lines = [ l.strip("\n") for l in all_sql_lines ]

    encoding_list = torch.load('tensor.pt')
# buffer = io.BytesIO()
# x_train = torch.stack(encoding_list).reshape( len(encoding_list), encoding_list[0].shape[1] )
x_train = torch.stack(encoding_list)
print("x_train.shape = ", x_train.shape)


# 1. PCA , (13996, 49) => (13996, 10)
pca = PCA(n_components=n_pca_component)
pca.fit(x_train)
x_new = torch.tensor(pca.transform(x_train), dtype=torch.float)
print("x_new.shape = ", x_new.shape)

# 2. Kmeans, (13996, 10) =>  1000 * ( 13.996 , 10 )
kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=0).fit(x_new)
cluster_indices = kmeans.predict( x_new )
cluster = {}

for i in range(0, 70):
    _min = i * 200
    _max = (i + 1) * 200
    if _max > len(cluster_indices):
        _max = len(cluster_indices)
    indicies = cluster_indices[_min : _max]
    ind_dict = {}
    for ind in indicies:
        if ind not in ind_dict:
            ind_dict[ind] = 0
        ind_dict[ind]+=1
    cnt_list = [ ind_dict[k] for k in ind_dict ]
    sorted(cnt_list, reverse=True)
    # print( "for pattern ", i, "histogram = ", cnt_list,  " max ratio = ", cnt_list[0] / (_max - _min))

for i, ind in enumerate(cluster_indices):
    if ind not in cluster.keys():
        cluster[ind] = []
    cluster[ind].append(i)

# for k in sorted(cluster.keys()):
#     print(k, len(cluster[k]))

# 3. Quantize each feature into 10 bins. 
col_min = torch.min(x_new, dim = 0).values
col_max = torch.max(x_new, dim = 0).values
col_scale = (col_max - col_min) / (n_bin - 1)
col_zero_point = - (n_bin - 1) * col_min / (col_max - col_min)
x_new = torch.quantize_per_channel(x_new, col_scale, col_zero_point, 1, dtype=torch.quint8).int_repr()
print(col_min, col_max)
print(x_new[0:10])

# 4. Convert each cluster into histograms. 
def get_hist(values: list, v_min: int, v_max: int, v_split: int, buckets: int):
    # elements in vlaues  must be in [ 0, length )
    # print(values)
    total_num = len(values)
    hist = [ 0 for _ in range(buckets) ]
    # Convert to histogram
    for v in values:
        hist[ math.floor( (v - v_min) / v_split) ] += 1
    # Normalize 
    for i in range(len(hist)):
        assert hist[i] <= total_num
        # Prevent 0 value.
        hist[i] += 1
        hist[i] = hist[i] / (total_num + len(hist))
    return torch.tensor(hist, dtype=float)

x_clusters = [ x_new[cluster[k]] for k in sorted(cluster.keys()) ]
# Content is the histogram.
h_clusters = []
h_clusters_tensor = None
print("len(x_clusters) = ", len(x_clusters))
vvalues = []
v_max = -1
v_min = -1
for i_cluster in x_clusters:
    values = []
    for q in i_cluster:
        v = 0
        for i, vv in enumerate(q):
            v += vv * (10 ** i)
        if v > v_max:
            v_max = v
        if v_min == -1 or v < v_min:
            v_min = v
        values.append(v)
    vvalues.append(values)

v_split = math.ceil((v_max - v_min) / n_buckets)
for values in vvalues:
    h_clusters.append(get_hist(values, v_min, v_max, v_split, n_buckets))

h_clusters_tensor = torch.stack(h_clusters)

for i , h in enumerate(h_clusters):
    print(i, h)
    pass

# central_points = [ h_clusters[ i * int( n_kmeans_clusters / n_distribution_clusters ) ] for i in range(n_distribution_clusters) ]
central_points = [ h_clusters[ i ] for i in range(n_distribution_clusters) ]

def KLdistance(h1, h2):
    # print(c1[0:10], c2[0:10])
    return sum(rel_entr(h1, h2))

new_h_clusters = { i: [] for i in range(len(central_points)) }

thre = 0.1
iteration = 0
max_iter = 100
while True:
    for i, i_cluster in enumerate(h_clusters):
        # compute the KL-divergence with each central point. 
        d = []
        for hc in central_points:
            dd = KLdistance(hc, i_cluster)
            # print("hc = ", hc, "\nic =", i_cluster, dd)
            # print("dd = ", dd)
            assert dd >= 0 
            d.append(dd)
        max_ind = torch.argmax(torch.tensor(d)).item()
        new_h_clusters[max_ind].append(i)
    # compute new central_ind
    new_central_points = []
    for i in range(n_distribution_clusters):
        if len(new_h_clusters[i]) > 0:
            # print(new_h_clusters[i])
            new_hc = torch.mean(h_clusters_tensor[new_h_clusters[i]], dim=0)
        else:
            new_hc = central_points[i]
        new_central_points.append(new_hc)
    distance = 0
    for i in range(n_distribution_clusters):
        # distance += torch.nn.PairwiseDistance(p=2)(central_points[i], new_central_points[i])
        distance += KLdistance(central_points[i], new_central_points[i])
    print("Iter = ", iteration, " distance = ", distance)
    iteration += 1
    if distance < thre or iteration >= max_iter:
        break
    central_points = new_central_points
    new_h_clusters = { i: [] for i in range(n_distribution_clusters) }

distri_cluster_indices = {}
for i in range(n_distribution_clusters):
    inds = new_h_clusters[i]
    distri_cluster_indices[i] = []
    for ind in inds:
        distri_cluster_indices[i] += cluster[ind]

sum_ = 0
final_cluster_indices = {}
for k in distri_cluster_indices.keys():
    c = distri_cluster_indices[k]
    print(len(c))
    if len(c) > 0: 
        final_cluster_indices[k]  = c
    sum_ += len(c)
print(sum_)




for i, k in enumerate(final_cluster_indices.keys()):
    fout = open(distri_path + "/" + "q" + str(i) + ".txt", "w")
    for ind in final_cluster_indices[k]:
        fout.write( all_query_lines[ind] + "\n" )
    fout.close()

for i, k in enumerate(final_cluster_indices.keys()):
    fout = open(distri_sql_path + "/" + "q" + str(i) + ".sql", "w")
    for ind in final_cluster_indices[k]:
        fout.write( all_sql_lines[ind] + "\n" )
    fout.close()



cluster = []
pca2 = PCA(n_components=2)
pca2.fit(x_train)
x_visual = torch.tensor(pca2.transform(x_train))

scatter_cluster = {}
for k in final_cluster_indices.keys():
    scatter_cluster[k] = { "x": x_visual[final_cluster_indices[k], 0], "y" : x_visual[final_cluster_indices[k], 1] }

colors = [ "green", "orange", "pink", "grey", "lime", "blue", "yellow", "red", "cyan", "violet" ]
for k in scatter_cluster.keys():
    rand = random.randint(0, 500)
    pc = plt.scatter(scatter_cluster[k]["x"] + torch.ones(size=scatter_cluster[k]["x"].shape) * rand, scatter_cluster[k]["y"] + torch.ones(size=scatter_cluster[k]["y"].shape) * rand, c = colors[k if k < 10 else k % 10 ], s = 15, edgecolors='none', alpha= 0.5)
    pc = plt.scatter(scatter_cluster[k]["x"] + torch.ones(size=scatter_cluster[k]["x"].shape) * rand, scatter_cluster[k]["y"] + torch.ones(size=scatter_cluster[k]["y"].shape) * rand, c = colors[k if k < 10 else k % 10 ], s = 2, edgecolors='none', alpha= 0.7)
plt.savefig("fig.png", dpi=1200, format="png")
plt.cla()
plt.scatter(x_visual[ :, 0], x_visual[ :, 1], s = 10, c = 'orange')
plt.savefig("fig2.png", dpi=1200, format="png")
