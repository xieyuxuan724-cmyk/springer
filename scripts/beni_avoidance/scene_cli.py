"""Shared scene overrides; load checkpoint first, then apply explicit CLI values."""
from .common import Config

FIELDS={
    'target_distance':float,'field_width':float,'back_margin':float,'max_obstacles':int,
    'obstacle_start':float,'obstacle_end_margin':float,'obstacle_side_margin':float,
    'route_resolution':float,'route_padding':float,'timeout_factor':float,
    'timeout_extra':float,'stop_timeout':float,
}


def add_scene_arguments(parser):
    for name,kind in FIELDS.items():
        parser.add_argument('--'+name.replace('_','-'),type=kind,default=None,
                            help='Scene override; field-width is full width, all distances in meters')


def apply_scene_arguments(cfg,args):
    before=cfg.dict()
    for name in FIELDS:
        value=getattr(args,name,None)
        if value is not None:
            if name=='field_width':cfg.half_width=value/2
            else:setattr(cfg,name,value)
    cfg.validate()
    changes={k:{'checkpoint':v,'effective':cfg.dict()[k]} for k,v in before.items() if v!=cfg.dict()[k]}
    if changes:print('Scene overrides (outside checkpoint conditions):',changes,flush=True)
    return changes


def standard_scene(cfg):
    default=Config()
    names=set(FIELDS)-{'field_width'}|{'half_width','safety_radius'}
    return all(getattr(cfg,k)==getattr(default,k) for k in names)
