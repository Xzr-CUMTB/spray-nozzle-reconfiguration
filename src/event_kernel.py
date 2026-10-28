"""Event termination with inherited drag/gravity/DDA physics. No fastmath.

Scalar loops avoid temporary numpy vectors; a verified same-cell DDA shortcut
avoids redundant traversal. Active parcels retain shared adaptive dt per nozzle.
Collision absorbs and removes the parcel from time-step control in this new
protocol. An optional shadow mode exists solely for frozen-kernel regression.
"""
import numpy as np
import math
from numba import njit
from inherited_kernels import mark_segment

@njit(cache=True)
def integrate_event(position,velocity,diameters,weights,wind,origin,upper,h,max_dt,boxes,covered,horizon,stability=.2,shadow=False,shortcut=True):
 n=len(diameters);pos=np.empty((n,3));pos[:]=position;vel=velocity.copy();active=np.ones(n,np.bool_);state=np.zeros(n,np.int64);times=np.full(n,horizon);rates=np.zeros(n)
 planes=np.array([.3,.6]);cross_speed=np.full((2,n),np.nan);cross_radial=np.full((2,n),np.nan)
 elapsed=0.;steps=0;min_dt=1.e99;max_dt_used=0.;max_jump=0.
 while elapsed<horizon-1e-14:
  maximum_rate=0.;maximum_speed=0.;num_active=0
  for j in range(n):
   if not active[j]:continue
   num_active+=1
   rx=wind[0]-vel[j,0];ry=wind[1]-vel[j,1];rz=wind[2]-vel[j,2]
   relative=math.sqrt(rx*rx+ry*ry+rz*rz);re=1.204*diameters[j]*relative/1.81e-5
   cd=24./(re+1.e-12)*(1.+.15*max(re,1.e-12)**.687) if re<=1000. else .44
   rates[j]=.75*1.204/(998.*diameters[j])*cd*relative
   maximum_rate=max(maximum_rate,rates[j]);maximum_speed=max(maximum_speed,math.sqrt(vel[j,0]**2+vel[j,1]**2+vel[j,2]**2))
  if num_active==0:break
  dt=min(max_dt,stability/max(maximum_rate,1.e-12),stability*h/(maximum_speed+1.),horizon-elapsed)
  min_dt=min(min_dt,dt);max_dt_used=max(max_dt_used,dt);steps+=1
  if steps>1000000:raise RuntimeError('Event integrator safety step ceiling')
  for j in range(n):
   if not active[j]:continue
   ax=pos[j,0];ay=pos[j,1];az=pos[j,2]
   vel[j,0]+=rates[j]*(wind[0]-vel[j,0])*dt;vel[j,1]+=rates[j]*(wind[1]-vel[j,1])*dt;vel[j,2]+=rates[j]*(wind[2]-vel[j,2])*dt;vel[j,2]-=9.81*dt
   dx=vel[j,0]*dt;dy=vel[j,1]*dt;dz=vel[j,2]*dt;bx=ax+dx;by=ay+dy;bz=az+dz
   max_jump=max(max_jump,math.sqrt(dx*dx+dy*dy+dz*dz)/h)
   fraction=1.
   if abs(dx)>1.e-15:fraction=min(fraction,max(0.,min(1.,((upper[0] if dx>0 else origin[0])-ax)/dx)))
   if abs(dy)>1.e-15:fraction=min(fraction,max(0.,min(1.,((upper[1] if dy>0 else origin[1])-ay)/dy)))
   if abs(dz)>1.e-15:fraction=min(fraction,max(0.,min(1.,((upper[2] if dz>0 else origin[2])-az)/dz)))
   ex=ax+fraction*dx;ey=ay+fraction*dy;ez=az+fraction*dz;hit=2.
   if state[j]==0:
    # Exact closed AABB slab test, same tangency convention as inherited kernel.
    for k in range(len(boxes)):
     enter=0.;leave=1.;valid=True
     for dim in range(3):
      a=ax if dim==0 else (ay if dim==1 else az);e=ex if dim==0 else (ey if dim==1 else ez);delta=e-a
      if abs(delta)<1.e-15:
       if a<boxes[k,0,dim] or a>boxes[k,1,dim]:valid=False;break
      else:
       t0=(boxes[k,0,dim]-a)/delta;t1=(boxes[k,1,dim]-a)/delta;enter=max(enter,min(t0,t1));leave=min(leave,max(t0,t1))
       if enter>leave:valid=False;break
     if valid and enter<=leave and enter<hit:hit=enter
    travel=min(1.,hit);tx=ax+travel*(ex-ax);ty=ay+travel*(ey-ay);tz=az+travel*(ez-az)
    # Diagnostic operator on x-normal planes is used only for +x fixtures.
    for k in range(2):
     loc=position[0]+planes[k]
     if ax<loc and tx>=loc and np.isnan(cross_speed[k,j]):
      u=(loc-ax)/(tx-ax);yy=ay+u*(ty-ay)-position[1];zz=az+u*(tz-az)-position[2]
      cross_speed[k,j]=math.sqrt(vel[j,0]**2+vel[j,1]**2+vel[j,2]**2);cross_radial[k,j]=math.sqrt(yy*yy+zz*zz)
    if (tx-ax)**2+(ty-ay)**2+(tz-az)**2>=1.e-30:
     ia=int(math.floor((ax-origin[0])/h));ja=int(math.floor((ay-origin[1])/h));ka=int(math.floor((az-origin[2])/h));ib=int(math.floor((tx-origin[0])/h));jb=int(math.floor((ty-origin[1])/h));kb=int(math.floor((tz-origin[2])/h))
     if shortcut and ia==ib and ja==jb and ka==kb and 0<=ia<covered.shape[0] and 0<=ja<covered.shape[1] and 0<=ka<covered.shape[2]:covered[ia,ja,ka]=True
     else:mark_segment(np.array([ax,ay,az]),np.array([tx,ty,tz]),origin,h,covered)
    if hit<=1.:
     state[j]=2;times[j]=elapsed+dt*fraction*hit
     if not shadow:active[j]=False
   pos[j,0]=bx;pos[j,1]=by;pos[j,2]=bz
   if bx<origin[0] or bx>=upper[0] or by<origin[1] or by>=upper[1] or bz<origin[2] or bz>=upper[2]:
    if state[j]==0:state[j]=1;times[j]=elapsed+dt*fraction
    active[j]=False
  elapsed+=dt
 return state,times,elapsed,steps,min_dt,max_dt_used,max_jump,cross_speed,cross_radial,vel,pos
