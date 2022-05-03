betas = rand(6)
sp = SimplePotentials(betas,3)
β(p::Potentials)=p.betas[p.id]
betas[3] = 14.5
β(sp)