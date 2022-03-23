import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numba import jit
import synmorph.tri_functions as trf

@jit(nopython=True)
def make_quartet_tri():
    return np.array(((0,1,3),
                     (1,2,3)))


@jit(nopython=True)
def get_tri_i_from_ab(tri,a,b,exclude_j=None):
    _tri = tri.copy()
    if exclude_j is not None:
        _tri[exclude_j] = -1
    return np.nonzero(trf.sum_tri(_tri==a)*trf.sum_tri(_tri==b))[0]

@jit(nopython=True)
def get_neigh(tri):
    neigh = np.ones_like(tri)*-1
    for i, tr in enumerate(tri):
        for j, el in enumerate(tr):
            a,b = np.roll(tr,-j)[1:]
            neighbour = get_tri_i_from_ab(tri,a,b,exclude_j = i)
            if neighbour.size !=0:
                neigh[i,j] = neighbour[0]
    return neigh

@jit(nopython=True)
def get_edge_list(tri):
    """ Includes double counting of edges."""
    edges = np.empty((tri.shape[0]*3,2),dtype=np.int64)
    for i,tr in enumerate(tri):
        for j in range(3):
            edges[3*i+j] = np.roll(tr,j)[:2]
    # edges = np.row_stack((np.roll(tri, i, axis=1)[:, :2] for i in range(3)))
    return edges

@jit(nopython=True)
def sort_2d_array(x):
    n,m=np.shape(x)
    for row in range(n):
        x[row]=np.sort(x[row])
    return x


@jit(nopython=True)
def get_edge_list_unique(tri):
    edges = get_edge_list(tri)
    edges = sort_2d_array(edges)
    edges = edges[np.argsort(edges[:,0]*999 + edges[:,1])] # a hack, given the number of nodes won't exceed 1000 practically.
    edges_new = np.zeros((0,2),dtype=np.int64)
    for em1,ep1 in zip(edges[:-1],edges[1:]):
        if np.any(em1 != ep1):
            edges_new = np.row_stack((edges_new,np.expand_dims(em1,0)))
    edges_new = np.row_stack((edges_new, np.expand_dims(ep1,0)))
    return edges_new

@jit(nopython=True)
def get_degree(edges):
    return np.bincount(edges.ravel())

@jit(nopython=True)
def expand_degree(tri,neigh,final_degree):
    nc_orig = tri.max() + 1
    #Add corner nodes.
    new_tri = np.zeros((0,3),dtype=np.int64)
    nc = nc_orig
    # neigh = get_neigh(tri)
    ntri = tri.shape[0]
    open_edges = np.row_stack(np.nonzero(neigh==-1))
    for (i,j) in open_edges.T:
        a, b = np.roll(tri[i], -j)[1:]
        new_tri = np.row_stack((new_tri,np.expand_dims(np.array((b,a,nc)),0)))
        ntri += 1
        neigh[i,j] = ntri
        neigh = np.row_stack((neigh,np.expand_dims(np.array((-1,-1,i)),0)))
        nc += 1

    tri_tot = np.row_stack((tri,new_tri))

    ##Add sole neighbours
    sole_neigh_tris = np.zeros((0,3),dtype=np.int64)
    edges = get_edge_list_unique(tri_tot)
    degree = get_degree(edges)
    extra_edges = final_degree - degree[:nc_orig]
    for cll in range(len(extra_edges)):
        n_extra_edge = extra_edges[cll]
        cll_in_new_tri = new_tri == cll
        if cll_in_new_tri.sum() != 0:
            bs,cws = np.row_stack(np.nonzero(new_tri == cll)) #given the construction, this gives the "b" in the "ab" edge, where "a" is cll, and "cws" is whether that line is cw or not in the original triangle.
            if cws[1] == 0: #ensure the first line is CW, i.e. generate triangles in a CCW manner
                bs,cws = np.flip(bs),np.flip(cws)
            bs += nc_orig
            bs = np.concatenate((np.array((bs[0],)),np.arange(n_extra_edge)+nc,np.array((bs[1],))))

            for i in range(len(bs)-1):
                new_t = np.array((cll,bs[i],bs[i+1]))
                sole_neigh_tris = np.row_stack((sole_neigh_tris,np.expand_dims(new_t,0)))
            nc += n_extra_edge

    ###NEED ADDITIONAL LINES TO MERGE CORNER NODES IF DEGREE IS EXCEEDED

    tri_tot = np.row_stack((tri_tot,sole_neigh_tris))
    return tri_tot


@jit(nopython=True)
def expand_degree0(tri,neigh,final_degree):
    nc_orig = tri.max() + 1
    #Add corner nodes.
    new_tri = np.zeros((0,3),dtype=np.int64)
    nc = nc_orig
    # neigh = get_neigh(tri)
    ntri = tri.shape[0]
    open_edges = np.array(((0,0,1,1),(1,2,2,0)))
    for (i,j) in open_edges.T:
        a, b = np.roll(tri[i], -j)[1:]
        new_tri = np.row_stack((new_tri,np.expand_dims(np.array((b,a,nc)),0)))
        ntri += 1
        neigh[i,j] = ntri
        neigh = np.row_stack((neigh,np.expand_dims(np.array((-1,-1,i)),0)))
        nc += 1

    tri_tot = np.row_stack((tri,new_tri))

    ##Add sole neighbours
    sole_neigh_tris = np.zeros((0,3),dtype=np.int64)
    edges = get_edge_list_unique(tri_tot)
    degree = get_degree(edges)
    extra_edges = final_degree - degree[:nc_orig]
    for cll in range(len(extra_edges)):
        n_extra_edge = extra_edges[cll]
        cll_in_new_tri = new_tri == cll
        if cll_in_new_tri.sum() != 0:
            bs,cws = np.row_stack(np.nonzero(new_tri == cll)) #given the construction, this gives the "b" in the "ab" edge, where "a" is cll, and "cws" is whether that line is cw or not in the original triangle.
            if cws[1] == 0: #ensure the first line is CW, i.e. generate triangles in a CCW manner
                bs,cws = np.flip(bs),np.flip(cws)
            bs += nc_orig
            bs = np.concatenate((np.array((bs[0],)),np.arange(n_extra_edge)+nc,np.array((bs[1],))))

            for i in range(len(bs)-1):
                new_t = np.array((cll,bs[i],bs[i+1]))
                sole_neigh_tris = np.row_stack((sole_neigh_tris,np.expand_dims(new_t,0)))
            nc += n_extra_edge

    ###NEED ADDITIONAL LINES TO MERGE CORNER NODES IF DEGREE IS EXCEEDED

    tri_tot = np.row_stack((tri_tot,sole_neigh_tris))
    return tri_tot


def get_quartet_x(tri,L=9,l_quartet_opt=3.3):
    nc = tri.max() + 1
    edges = get_edge_list_unique(tri)
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G,iterations=2000)
    mid = np.array((L/2,L/2))
    quartet_x = np.array([pos[i] for i in range(nc)])
    if np.cross(quartet_x[1]-quartet_x[0],quartet_x[2]-quartet_x[0])<0:
        quartet_x[:,0] = -quartet_x[:,0]
    # quartet_x = np.array(list(pos.values()))
    l_quartet = max((quartet_x[:,0].max() - quartet_x[:,0].min(),quartet_x[:,0].max() - quartet_x[:,0].min()))
    quartet_x = quartet_x*l_quartet_opt/l_quartet + mid
    if (quartet_x<0).any() or (quartet_x>L).any():
        raise Exception("change L or l_quartet_opt")
    return quartet_x

def plot_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G,iterations=200)
    nx.draw(G,pos, with_labels = True)
    plt.show()

def plot_tri(tri):
    edges = get_edge_list_unique(tri)
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G,iterations=200)
    nx.draw(G,pos, with_labels = True)
    plt.show()

def generate_quartet(degrees,L=9,l_quartet_opt=3.3):
    tri0 = make_quartet_tri()
    quartet_tri = expand_degree0(tri0, get_neigh(tri0), degrees)
    quartet_neigh = get_neigh(quartet_tri)
    quartet_x = get_quartet_x(quartet_tri,L,l_quartet_opt)
    return quartet_x,quartet_tri,quartet_neigh

def generate_quartet_topology(degrees):
    tri0 = make_quartet_tri()
    quartet_tri = expand_degree0(tri0, get_neigh(tri0), degrees)
    quartet_neigh = get_neigh(quartet_tri)
    return quartet_tri, quartet_neigh
    #
    #
    # quartet_tri = expand_degree(tri, get_neigh(tri), degrees)
    # quartet_neigh = get_neigh(quartet_tri)
    # quartet_x = get_quartet_x(quartet_tri, L, l_quartet_opt)
