"""Seeded obstacle layouts, checked using an inflated occupancy grid."""
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from play_native import build_model, motor_config

# A fixed multi-size catalogue allows identical models across GPU worlds.
SIZES = np.array([[.12,.18],[.13,.13],[.20,.12],[.16,.16],
                  [.14,.22],[.18,.18],[.20,.18],[.11,.11]],dtype=np.float32)
KINDS = np.arange(8)%2  # 0 box, 1 cylinder


def model():
    def add(world):
        for i,(size,kind) in enumerate(zip(SIZES,KINDS)):
            body=ET.SubElement(world,'body',name=f'obstacle_{i}',mocap='true',pos=f'{10+i} 0 .3')
            ET.SubElement(body,'geom',name=f'obstacle_geom_{i}',type='box' if kind==0 else 'cylinder',
                size=f'{size[0]} {size[1]} .3' if kind==0 else f'{size[0]} .3',
                rgba='.85 .35 .12 1' if kind==0 else '.15 .55 .7 1',contype='1',conaffinity='1')
    return build_model(motor_config(), add)


def route_exists(centers, active, radius=.19, half_width=1.5):
    xs=np.arange(-.3,3.21,.1);ys=np.arange(-half_width+radius,half_width-radius+.001,.1)
    xx,yy=np.meshgrid(xs,ys,indexing='ij');points=np.stack((xx,yy),-1)
    free=np.ones(xx.shape,bool)
    for i in np.flatnonzero(active):
        d=np.abs(points-centers[i])
        if KINDS[i]: forbidden=np.linalg.norm(d,axis=-1)<=SIZES[i,0]+radius+.05
        else: forbidden=np.all(d<=SIZES[i]+radius+.05,axis=-1)
        free &= ~forbidden
    start=(int(np.argmin(abs(xs))),int(np.argmin(abs(ys))))
    if not free[start]:return False
    queue=deque([start]);seen={start}
    while queue:
        x,y=queue.popleft()
        if x==len(xs)-1:return True
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<len(xs) and 0<=ny<len(ys) and free[nx,ny] and (nx,ny) not in seen:
                seen.add((nx,ny));queue.append((nx,ny))
    return False


def layout(seed, count, radius=.19, half_width=1.5):
    rng=np.random.default_rng(seed)
    for _ in range(1000):
        active=np.zeros(8,bool);active[rng.choice(8,count,replace=False)]=True
        centers=np.zeros((8,2),dtype=np.float32)
        placed=[]
        for i in rng.permutation(np.flatnonzero(active)):
            for attempt in range(200):
                candidate=np.array([rng.uniform(.85,2.75),rng.uniform(-1.05,1.05)])
                if all(np.linalg.norm(candidate-centers[j]) >= np.linalg.norm(SIZES[i])+np.linalg.norm(SIZES[j])+.08 for j in placed):
                    centers[i]=candidate;placed.append(i);break
            else:break
        if len(placed)!=count:continue
        if route_exists(centers,active,radius,half_width):return centers,active
    raise RuntimeError(f'Unable to sample traversable layout for seed {seed}')
