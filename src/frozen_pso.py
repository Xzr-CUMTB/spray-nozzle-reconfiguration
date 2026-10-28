# -*- coding: utf-8 -*-
"""Frozen Direct PSO update; independent initialization and IO supplied by caller."""
import numpy as np
from frozen_model import LOWER,UPPER,repair
def optimize(positions,rng,score,iterations,on_generation):
 particles,dimensions=positions.shape;velocity=np.zeros_like(positions);repairs=[]
 values=score(positions,0);pbest=positions.copy();pvalues=values.copy();idx=int(np.argmax(values));gbest=positions[idx].copy();gvalue=float(values[idx]);history=[dict(generation=0,best_coverage=gvalue,mean_coverage=float(values.mean()))];yaw_ids=np.arange(3,dimensions,5);surface_ids=np.arange(0,dimensions,5)
 on_generation(0,positions.copy(),values.copy(),history[-1].copy(),[])
 for iteration in range(1,iterations+1):
  inertia=.90+(.40-.90)*iteration/iterations;r1=rng.random((particles,dimensions));r2=rng.random((particles,dimensions));pd=pbest-positions;gd=gbest-positions;pd[:,yaw_ids]=(pd[:,yaw_ids]+180)%360-180;gd[:,yaw_ids]=(gd[:,yaw_ids]+180)%360-180
  velocity=inertia*velocity+1.55*r1*pd+1.75*r2*gd;velocity=np.clip(velocity,-.2*(UPPER-LOWER),.2*(UPPER-LOWER));proposed=positions+velocity;proposed[:,yaw_ids]=(proposed[:,yaw_ids]+180)%360-180
  draw=rng.random((particles,6));surface=np.floor(positions[:,surface_ids]);surface=np.where(draw<.35,np.floor(pbest[:,surface_ids]),surface);surface=np.where((draw>=.35)&(draw<.70),np.floor(gbest[surface_ids]),surface);surface=np.where(draw>=.95,rng.integers(0,3,surface.shape),surface);proposed[:,surface_ids]=surface+.5;velocity[:,surface_ids]=0
  clipped=np.clip(proposed,LOWER,UPPER);velocity[clipped!=proposed]=0;positions=np.asarray([repair(p,rng) for p in clipped]);velocity[positions!=clipped]=0;repairs.append(dict(generation=iteration,changed_particles=int(np.count_nonzero(np.any(positions!=clipped,axis=1))),changed_coordinates=int(np.count_nonzero(positions!=clipped))));values=score(positions,iteration);better=values>pvalues;pbest[better]=positions[better];pvalues[better]=values[better];idx=int(np.argmax(pvalues))
  if pvalues[idx]>gvalue:gvalue=float(pvalues[idx]);gbest=pbest[idx].copy()
  history.append(dict(generation=iteration,best_coverage=gvalue,mean_coverage=float(values.mean())))
  on_generation(iteration,positions.copy(),values.copy(),history[-1].copy(),list(repairs))
 return dict(best_particle=gbest.tolist(),best_coverage=gvalue,history=history,repairs=repairs)
