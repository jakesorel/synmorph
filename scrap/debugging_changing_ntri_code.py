

for i in range(1000):
    sim.t.mesh.x += np.random.normal(0,0.01,(sim.t.mesh.x.shape))
    sim.t.mesh.x = np.mod(sim.t.mesh.x,1)
    sim.t.mesh._triangulate()
