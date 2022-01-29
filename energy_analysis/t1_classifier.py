import numpy as np
import synmorph as sm
import synmorph.tri_functions as trf
import triangle as tr
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import networkx as nx
from scipy.sparse import coo_matrix
from itertools import combinations
import time
from energy_analysis.to_graph6 import generate_digraph6_from_adjacency_mat,generate_graph6_from_adjacency
from numba import jit


"""
Topologies
----------
Determine the topologies of each graph, using the Delanauy triangulation
"""

def get_tris(x0s):
    tris = []
    for x0 in x0s:
        t = tr.triangulate({"vertices":x0})
        tris.append(t["triangles"])
    return tris

def get_edge_lists(tris):
    edge_lists = []
    for tri in tris:
        edges = np.row_stack([np.roll(tri, i,axis=1)[:, :2] for i in range(3)])
        # edges = edges[edges[:, 0] < edges[:, 1]]
        edge_lists.append(edges)
    return edge_lists



"""
1. Consider all possible quartet topologies. 
Each cell in the quartet may have 5, 6, or 7 neighbours. 
This gives a total of 81 topologies. (3^4)

"""

class generate_quartets:
    def __init__(self,del_x=0.5,del_sa = 0.5,min_neigh_x=0,max_neigh_x = 2,min_neigh_y=1,max_neigh_y=3, max_neigh=7,non_isomorphic=True,del_y = 0.0,check_degrees=True):
        self.del_x = del_x
        self.del_sa = del_sa
        self.non_isomorphic = non_isomorphic
        self.check_degrees = check_degrees
        self.min_neigh_x,self.max_neigh_x = min_neigh_x,max_neigh_x
        self.min_neigh_y,self.max_neigh_y = min_neigh_y,max_neigh_y

        self.x0_quartet = np.array(((-1 - del_x, 0),
                               (1 + del_x, 0),
                               (0, np.sqrt(3)),
                               (0, -np.sqrt(3))))
        self.x0_corners = np.array(((-1 - del_x*2, -np.sqrt(3)-del_y),
                               (-1 - del_x*2, np.sqrt(3)+del_y),
                               (1 + del_x*2, -np.sqrt(3)-del_y),
                               (1 + del_x*2, np.sqrt(3)+del_y)))

        self.sa1s, self.sa2s, self.sa3s, self.sa4s = [None]*4
        self.get_single_additions()
        self.x0s = []
        self.get_all_neighbourhoods()
        # return self.x0s

    def get_single_additions(self):
        self.single_addition_x = np.arange(self.min_neigh_x,self.max_neigh_x+1, dtype=np.int64)
        self.single_addition_y = np.arange(self.min_neigh_y,self.max_neigh_y+1, dtype=np.int64)

        self.sa1s, self.sa2s, self.sa3s, self.sa4s = np.meshgrid(self.single_addition_x,
                                                                 self.single_addition_x,
                                                                 self.single_addition_y,
                                                                 self.single_addition_y,
                                                                 indexing="ij")
        # if non_isomorphic:
        #     sa = np.array((self.sa1s.ravel(),self.sa2s.ravel(),self.sa3s.ravel(),self.sa4s.ravel())).T
        #     for i in range(sa.shape[0]):
        #         sa[i] = sa[i][sa[i].argsort()]
        #     sa = np.unique(sa,axis=0)
        #     self.sa1s, self.sa2s, self.sa3s, self.sa4s = sa.T
    def get_even_spacing(self, sai):
        if sai == 1:
            return np.zeros((1))
        else:
            return np.linspace(-self.del_sa / 2, self.del_sa / 2, sai)

    def get_additional_pointsx(self,sai):
        return np.array((np.zeros(sai), self.get_even_spacing(sai))).T

    def get_additional_pointsy(self,sai):
        return np.array((self.get_even_spacing(sai), np.zeros(sai))).T


    def get_x0_single_addition(self,i):
        sa1, sa2, sa3, sa4 = self.sa1s.take(i), self.sa2s.take(i), self.sa3s.take(i), self.sa4s.take(i)

        x0_single_addition = np.zeros((0, 2))
        x0_single_addition = np.row_stack((x0_single_addition, 3 * self.x0_quartet[0] + self.get_additional_pointsx(sa1)))
        x0_single_addition = np.row_stack((x0_single_addition, 3 * self.x0_quartet[1] + self.get_additional_pointsx(sa2)))
        x0_single_addition = np.row_stack((x0_single_addition, 3 * self.x0_quartet[2] + self.get_additional_pointsy(sa3)))
        x0_single_addition = np.row_stack((x0_single_addition, 3 * self.x0_quartet[3] + self.get_additional_pointsy(sa4)))
        return x0_single_addition

    def build_neighbourhood(self,i):
        return np.row_stack((self.x0_quartet, self.x0_corners, self.get_x0_single_addition(i)))

    def get_all_neighbourhoods(self):
        x0s = []
        for i in range(self.sa1s.size):
            x0s.append(self.build_neighbourhood(i))
        self.x0s = x0s
        return x0s

    def check_isomorphism(self):
        tris = get_tris(self.x0s)
        edge_list = get_edge_lists(tris)
        identifiers = []
        ns = [np.max(tri) + 1 for tri in tris]
        for i, edges in enumerate(edge_list):
            n = ns[i]
            A = coo_matrix((np.repeat(True, len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
            A = A + A.T
            A = A.toarray()
            degrees = A[:4].sum(axis=1)
            quartet_degree = A[:4,:4].sum(axis=1)
            identifiers.append(set(zip(degrees,quartet_degree)))

        unique_identifiers = {}
        for i, identifier in enumerate(identifiers):
            if sum([identifier == idf for idf in unique_identifiers.values()]) == 0:
                unique_identifiers[i] = identifier
        self.x0s = list(np.array(self.x0s)[list(unique_identifiers.keys())])

    def check_degree(self,min_degree=5,max_degree=7,change=True):
        tris = get_tris(self.x0s)
        degree_ok = np.zeros(len(self.x0s),dtype=bool)
        edge_list = get_edge_lists(tris)
        ns = [np.max(tri) + 1 for tri in tris]
        for i, edges in enumerate(edge_list):
            n = ns[i]
            A = coo_matrix((np.repeat(True, len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
            A = A + A.T
            A = A.toarray()
            degrees = A[:4].sum(axis=1)
            # print(degrees)
            degree_ok[i] = (degrees>=min_degree).all() * (degrees<=max_degree).all()
        if change:
            self.x0s = list(np.array(self.x0s)[degree_ok])
        return degree_ok

    def run(self):
        self.get_all_neighbourhoods()
        if self.non_isomorphic:
            self.check_isomorphism()
        if self.check_degrees:
            self.check_degree()
        return self.x0s
x0s = generate_quartets(non_isomorphic=False,del_sa=2).run()
tris = get_tris(x0s)

i = -1
x0 = x0s[i]
tri = tris[i]
fig = voronoi_plot_2d(Voronoi(x0))
ax = fig.gca()
for i in range(x0.shape[0]):
    ax.text(x0[i,0],x0[i,1],i)
fig.show()

t = tr.triangulate({"vertices": x0},"n")
tri = t["triangles"]
neigh = t["neighbors"]

class tri_hasher:
    def __init__(self,tri,neigh,quartet=[2,0,3,1],cols=None):
        ##quartet is defined CCW from a cell that shares two but not three interfaces with other members of the quartet
        self.tri = tri
        self.neigh = neigh
        self.quartet = np.array(quartet)

        self.a, self.b, self.c, self.d = [None]*4
        self.ca, self.ab, self.bc, self.cd = [None]*4
        self.assign_cells()
        self.cols = np.ones(tri.max()+2,dtype=int)
        if cols is not None:
            self.cols[:-1] = cols + 1
        self.cols[-1] = 0
        self.assign_tris()

    def assign_cells(self):
        self.a,self.b,self.c,self.d = self.quartet

    def return_tri_id(self,triple):
        return np.nonzero((self.tri == triple[0]).any(axis=1)*
                          (self.tri == triple[1]).any(axis=1)*
                          (self.tri == triple[2]).any(axis=1))[0][0]

    def return_tri_id_with_a_but_not_bcd(self,a,b,c,d):
        return np.nonzero((self.tri == a).any(axis=1)*
                          (self.tri != b).all(axis=1)*
                          (self.tri != c).all(axis=1) *
                          (self.tri != d).all(axis=1))[0]

    def exclude_cells_from_list(self,input_list,exclusion_list):
        return list(set(input_list).difference(set(exclusion_list)))

    def get_sole_neighbours(self,a,cw_corner_tri,ccw_corner_tri):
        """
        Go around counter clockwise from the cw corner tri, saving the neighbours until the ccw corner tri is reached.
        :param a:
        :param cw_corner_tri:
        :param ccw_corner_tri:
        :return:
        """
        cwt = cw_corner_tri.copy()
        ccwt = ccw_corner_tri.copy()
        sole_neighbours = np.ones(3,dtype=int)*-1
        i = 0
        first = True
        while cwt != ccwt:
            triple = list(self.tri[cwt])
            a_in_triple = triple.index(a)
            cwt = self.neigh[cwt, (a_in_triple + 1) % 3]
            if (first is False)and(cwt != ccwt):
                sole_neighbours[i-1] = triple[(a_in_triple+2)%3]
            else:
                first = False
            i +=1
        return sole_neighbours

    def assign_tris(self):
        tri0 = np.array((self.quartet[0],self.quartet[1],self.quartet[3]))
        tri1 = np.array((self.quartet[2],self.quartet[3],self.quartet[1]))

        tri0_id = self.return_tri_id(tri0)
        tri1_id = self.return_tri_id(tri1)
        quartet_tri_neighs = list(set(list(self.neigh[tri0_id])+list(self.neigh[tri1_id])).difference(set((tri0_id,tri1_id))))
        for qtn in quartet_tri_neighs:
            tri_neighbour = self.tri[qtn]
            # print(tri_neighbour)
            ab_mask = (tri_neighbour==self.a) + (tri_neighbour==self.b)
            if ab_mask.sum()==2:
                self.ab = tri_neighbour[~ab_mask][0]
                self.triab = qtn


            bc_mask = (tri_neighbour == self.b) + (tri_neighbour == self.c)
            if bc_mask.sum() == 2:
                self.bc = tri_neighbour[~bc_mask][0]
                self.tribc = qtn

            cd_mask = (tri_neighbour == self.c) + (tri_neighbour == self.d)
            if cd_mask.sum() == 2:
                self.cd = tri_neighbour[~cd_mask][0]
                self.tricd = qtn

            da_mask = (tri_neighbour == self.d) + (tri_neighbour == self.a)
            if da_mask.sum() == 2:
                self.da = tri_neighbour[~da_mask][0]
                self.trida = qtn


        self.quartet_corners = np.array((self.da,self.ab,self.bc,self.cd))
        self.quartet_corners_tri = np.array((self.trida,self.triab,self.tribc,self.tricd))

        self.sole_neighbours = np.array([self.get_sole_neighbours(a,cw_corner_tri,ccw_corner_tri) for (a,cw_corner_tri,ccw_corner_tri) in zip(self.quartet,self.quartet_corners_tri,np.roll(self.quartet_corners_tri,-1))])

        for (a,cw_corner_tri,ccw_corner_tri) in zip(self.quartet,self.quartet_corners_tri,np.roll(self.quartet_corners_tri,-1)):
            self.get_sole_neighbours(a, cw_corner_tri, ccw_corner_tri)


    def get_hash(self):
        hash_list = np.concatenate(((self.cols[self.quartet]),(self.cols[self.quartet_corners]),(self.cols[self.sole_neighbours.T.ravel()])))+1
        hash_string = "".join([str(el) for el in hash_list])
        hex_encoding = int(hash_string,3)
        return hex_encoding

    def swap_ids(self,i,j,init_list):
        list_new = init_list.copy()
        list_new[i::4],list_new[j::4] = init_list[j::4],init_list[i::4]
        return list_new

    def get_ids_for_hash(self):
        return list(self.quartet) + list(self.quartet_corners) + list(self.sole_neighbours.T.ravel())

    def get_all_hashes(self):
        ids_for_hash = list(self.quartet) + list(self.quartet_corners) + list(self.sole_neighbours.T.ravel())
        lr_symm = self.swap_ids(1,3,ids_for_hash)
        ud_symm = self.swap_ids(0,2,ids_for_hash)
        lrud_symm = self.swap_ids(0,2,lr_symm)

        hash_lists = self.cols[ids_for_hash],self.cols[lr_symm],self.cols[ud_symm],self.cols[lrud_symm]
        hash_strings = ["".join([str(el) for el in hash_list]) for hash_list in hash_lists]
        hex_encoding = [int(hash_string,3) for hash_string in hash_strings]
        return hex_encoding

    def get_min_hash(self):
        return min(self.get_all_hashes())

    def decode_hash(self,hash):

        def digit_to_char(digit):
            if digit < 10:
                return chr(ord('0') + digit)
            else:
                return chr(ord('a') + digit - 10)

        def str_base(number, base):
            if number < 0:
                return '-' + str_base(-number, base)
            else:
                (d, m) = divmod(number, base)
                if d:
                    return str_base(d, base) + digit_to_char(m)
                else:
                    return digit_to_char(m)

        return str_base(hash,3)
        # format(hex_encoding,"o") returns the hash_string

@jit(nopython=True)
def swap_ids(i, j, init_list):
    list_new = init_list.copy()
    list_new[i::4], list_new[j::4] = init_list[j::4], init_list[i::4]
    return list_new

def get_hash_from_col_string(col_list):
    return int("".join(map(str, col_list)), 3)



@jit(nopython=True)
def get_col_lists(ids_for_hash,nc,combo):
    cols = np.ones(nc+1,dtype=np.int64)
    cols[-1] = 0
    if len(combo)!=0:
        cols[np.array(combo)] = 2
    lr_symm = swap_ids(1, 3, ids_for_hash)
    ud_symm = swap_ids(0, 2, ids_for_hash)
    lrud_symm = swap_ids(0, 2, lr_symm)
    return [cols[ids_for_hash], cols[lr_symm], cols[ud_symm], cols[lrud_symm]]

def get_hash(ids_for_hash,nc,combo):
    col_lists = get_col_lists(ids_for_hash,nc,combo)
    return min(map(get_hash_from_col_string,col_lists))
x0s = generate_quartets(non_isomorphic=True,del_sa=2).run()
# x0s_post_t1 = generate_quartets(non_isomorphic=True,del_sa=2,del_x=2,del_y=2,min_neigh_x=1,max_neigh_x=3,min_neigh_y=1,max_neigh_y=3).run()
qrt_gen_post_t1 = generate_quartets(non_isomorphic=True,del_sa=2,del_x=1,del_y=2,check_degrees=False)
x0s_post_t1 = qrt_gen_post_t1.run()
degrees_post_t1 = qrt_gen_post_t1.check_degree(change=False)

##Elinimate all where the degree is > 8 or < 4.
x0s = [x0s[i] for i in np.nonzero(degrees_post_t1)[0]]
x0s_post_t1 = [x0s_post_t1[i] for i in np.nonzero(degrees_post_t1)[0]]

print(len(x0s_post_t1))
x0 = x0s_post_t1[3]
x0 = x0s[3]

Vor = Voronoi(x0)
# xlim,ylim = zip(Vor.vertices.min(axis=0)*2,Vor.vertices.max(axis=0)*2)
fig = voronoi_plot_2d(Voronoi(x0))
ax = fig.gca()
for i in range(x0.shape[0]):
    ax.text(x0[i,0],x0[i,1],i)
# ax.set(xlim=xlim,ylim=ylim)
fig.show()
def get_homogeneous_hashes(x0s,quartet = [2, 0, 3, 1]):
    hashes = []
    for j, x0 in enumerate(x0s):
        t = tr.triangulate({"vertices": x0}, "n")
        tri = t["triangles"]
        neigh = t["neighbors"]
        # hashes.append(tri_hasher(tri, neigh, quartet).get_ids_for_hash())
        hashes.append(tri_hasher(tri, neigh, quartet).get_min_hash())
    return hashes

##This gives a mapping between hashes
hashes = get_homogeneous_hashes(x0s,[2, 0, 3, 1])
hashes_post_t1 = get_homogeneous_hashes(x0s_post_t1,[1,2,0,3])
#
# G = nx.Graph()
# G.add_edges_from(zip(hashes,hashes_post_t1))
# nx.draw(G)
# plt.show()
##however it is slow. How fast is the colouring algorithm?

ids_for_hashes = []
for j, x0 in enumerate(x0s):
    t = tr.triangulate({"vertices": x0}, "n")
    tri = t["triangles"]
    neigh = t["neighbors"]
    ids_for_hashes.append(tri_hasher(tri, neigh, [2, 0, 3, 1]).get_ids_for_hash())

ncs = [x0.shape[0] for x0 in x0s]



# t0 = time.time()
# for i in range(int(1e4)):
#     get_hash(ids_for_hash, nc, combo)
# t1= time.time()
# print(t1-t0)

unique_hashes = []
for i in range(len(ids_for_hashes)):
    ids_for_hash = np.array(ids_for_hashes[i])
    nc = ncs[i]

    unique_hashes_by_j = [None]*(int(nc / 2) + 1)
    unique_hashes_i = []
    for j in range(int(nc / 2) + 1):
        combos = list(combinations(range(nc), j))
        hashes_j = [None]*len(combos)
        for k, combo in enumerate(combos):
            hashes_j[k] = get_hash(ids_for_hash, nc, combo)
        unique_hashes_j = list(set(hashes_j))
        unique_hashes_i = list(set(unique_hashes_j + unique_hashes_i))
    unique_hashes = list(set(unique_hashes + unique_hashes_i))
    print(len(unique_hashes_i),len(unique_hashes))

i = 3
t = tr.triangulate({"vertices": x0s[i]}, "n")
tri = t["triangles"]
neigh = t["neighbors"]
ids_for_hash = np.array(tri_hasher(tri, neigh, [2, 0, 3, 1]).get_ids_for_hash())
t = tr.triangulate({"vertices": x0s_post_t1[i]}, "n")
tri = t["triangles"]
neigh = t["neighbors"]
ids_for_hash_post_t1 = np.array(tri_hasher(tri, neigh, [1,2,0,3]).get_ids_for_hash())

nc = x0s[i].shape[0]

coloured_hashes = []
coloured_hashes_post_t1 = []
for j in range(int(nc / 2)+1):
    combos = list(combinations(range(nc), j))
    for k, combo in enumerate(combos):
        coloured_hashes.append(get_hash(ids_for_hash, nc, combo))
        coloured_hashes_post_t1.append(get_hash(ids_for_hash_post_t1, nc, combo))


G = nx.DiGraph()
G.add_edges_from(zip(coloured_hashes,coloured_hashes_post_t1))
pos = nx.spring_layout(G)
nx.draw(G,pos=pos,node_size=1)
nx.draw(G,node_size=0)
plt.show()

G = nx.Graph()
G.add_edges_from(zip(coloured_hashes,coloured_hashes_post_t1))
max(nx.algorithms.components.connected_components(G),key=len)