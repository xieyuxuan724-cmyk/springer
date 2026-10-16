"""Seeded obstacle layouts, checked using an inflated occupancy grid."""
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from play_native import build_model, motor_config
from .common import Config

# A fixed multi-size catalogue allows identical models across GPU worlds.
SIZES = np.array([[.12,.18],[.13,.13],[.20,.12],[.16,.16],
                  [.14,.22],[.18,.18],[.20,.18],[.11,.11]],dtype=np.float32)
KINDS = np.arange(8)%2  # 0 box, 1 cylinder


def catalogue(capacity=8):
    return np.tile(SIZES,(int(np.ceil(capacity/8)),1))[:capacity].copy(), np.arange(capacity)%2


def model(cfg=None):
    cfg=cfg or Config()
    sizes,kinds=catalogue(cfg.max_obstacles)
    def add(world):
        for i,(size,kind) in enumerate(zip(sizes,kinds)):
            body=ET.SubElement(world,'body',name=f'obstacle_{i}',mocap='true',pos=f'{10+i} 0 .3')
            ET.SubElement(body,'geom',name=f'obstacle_geom_{i}',type='box' if kind==0 else 'cylinder',
                size=f'{size[0]} {size[1]} .3' if kind==0 else f'{size[0]} .3',
                rgba='.85 .35 .12 1' if kind==0 else '.15 .55 .7 1',contype='1',conaffinity='1')
    return build_model(motor_config(), add)


def route_exists(centers, active, radius=.19, half_width=1.5, cfg=None):
    cfg=cfg or Config(safety_radius=radius,half_width=half_width)
    cfg.validate();radius=cfg.safety_radius;half_width=cfg.half_width
    sizes,kinds=catalogue(len(active))
    step=cfg.route_resolution
    xs=np.arange(-cfg.back_margin+step,cfg.target_distance+2*step+.01*step,step)
    ys=np.arange(-half_width+radius,half_width-radius+.001,step)
    xx,yy=np.meshgrid(xs,ys,indexing='ij');points=np.stack((xx,yy),-1)
    free=np.ones(xx.shape,bool)
    for i in np.flatnonzero(active):
        d=np.abs(points-centers[i])
        if kinds[i]: forbidden=np.linalg.norm(d,axis=-1)<=sizes[i,0]+radius+cfg.route_padding
        else: forbidden=np.all(d<=sizes[i]+radius+cfg.route_padding,axis=-1)
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


class LayoutSamplingError(ValueError):
    """Requested density could not be placed with a checked route."""


def separated(candidate, index, centers, placed, sizes, kinds, gap=.08):
    """Exact planar primitive separation, with the original 8 cm gap."""
    if not placed:return True
    ids=np.asarray(placed)
    delta=np.abs(centers[ids]-candidate)
    if kinds[index]==0:
        boxes=kinds[ids]==0
        distances=np.linalg.norm(np.maximum(delta-sizes[index]-sizes[ids],0),axis=1)
        circles=np.linalg.norm(np.maximum(delta-sizes[index],0),axis=1)-sizes[ids,0]
        distances=np.where(boxes,distances,circles)
    else:
        circles=kinds[ids]==1
        distances=np.linalg.norm(np.maximum(delta-sizes[ids],0),axis=1)-sizes[index,0]
        circle_distance=np.linalg.norm(delta,axis=1)-sizes[index,0]-sizes[ids,0]
        distances=np.where(circles,circle_distance,distances)
    return bool(np.all(distances>=gap))


def layout(seed, count, radius=.19, half_width=1.5, cfg=None):
    cfg=cfg or Config(safety_radius=radius,half_width=half_width)
    cfg.validate();capacity=cfg.max_obstacles
    if not 0<=count<=capacity:raise ValueError('Obstacle count exceeds scene capacity')
    sizes,kinds=catalogue(capacity)
    rng=np.random.default_rng(seed)
    best_placed=0;complete_without_route=0
    for _ in range(1000 if count<=8 else 100):
        active=np.zeros(capacity,bool);active[rng.choice(capacity,count,replace=False)]=True
        centers=np.zeros((capacity,2),dtype=np.float32)
        placed=[]
        for i in rng.permutation(np.flatnonzero(active)):
            for attempt in range(200):
                candidate=np.array([rng.uniform(cfg.obstacle_start,cfg.target_distance-cfg.obstacle_end_margin),rng.uniform(-cfg.half_width+cfg.obstacle_side_margin,cfg.half_width-cfg.obstacle_side_margin)])
                # Preserve historical seeds for the original <=8-obstacle tasks.
                fits=(all(np.linalg.norm(candidate-centers[j]) >= np.linalg.norm(sizes[i])+np.linalg.norm(sizes[j])+.08 for j in placed)
                      if count<=8 else separated(candidate,i,centers,placed,sizes,kinds))
                if fits:
                    centers[i]=candidate;placed.append(i);break
            else:break
        best_placed=max(best_placed,len(placed))
        if len(placed)!=count:continue
        if route_exists(centers,active,cfg=cfg):return centers,active
        complete_without_route+=1
    width=2*(cfg.half_width-cfg.obstacle_side_margin)
    length=cfg.target_distance-cfg.obstacle_end_margin-cfg.obstacle_start
    raise LayoutSamplingError(
        f'Cannot generate {count} obstacles for seed {seed}: center sampling region '
        f'{length:.2f} m x {width:.2f} m; best placed {best_placed}/{count}, '
        f'{complete_without_route} complete layouts failed the route check. '
        'Increase --target-distance / --field-width, reduce --obstacles, or try another --seed. '
        '--max-obstacles only allocates slots; it does not enlarge the field. '
        'Obstacle count, separation and route requirements were not relaxed.')
