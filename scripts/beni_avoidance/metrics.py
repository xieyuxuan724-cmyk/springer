"""Episode-weighted training metrics and held-out evaluation summaries."""
import numpy as np

REASONS={1:'success',2:'timeout',3:'stalled',4:'out_of_bounds',5:'fallen',6:'collision'}


def training_metrics(env, info, ids):
    # Return vectors, not batch means: RSL concatenates then averages them,
    # so every finished episode has equal weight even with asynchronous resets.
    metrics={f'RewardStep/{k}':v.detach().clone() for k,v in info['reward_terms'].items()}
    metrics['Curriculum/scheduled_stage']=float(env.stage)
    if not len(ids):return metrics
    metrics.update({f'RewardEpisode/{k}':v[ids].clone() for k,v in info['reward_sums'].items()})
    for key,source in [('success','success'),('progress_m','progress'),('heading_error_rad','heading_error'),
                       ('speed_error_mps','speed_error'),('min_clearance_m','clearance'),('duration_s','time')]:
        metrics[f'Avoidance/{key}']=info[source][ids].float().clone()
    for name,mask in info['events'].items():metrics[f'Failure/{name}']=mask[ids].float().clone()
    speed=env.speed[ids]
    masks={'zero':speed==0,'low_0.1_0.2':(speed>0)&(speed<.2),
           'medium_0.2_0.4':(speed>=.2)&(speed<.4),'high_0.4_0.5':speed>=.4}
    for name,mask in masks.items():
        if mask.any():
            metrics[f'TrainSpeed/{name}/success']=info['success'][ids][mask].float().clone()
            metrics[f'TrainSpeed/{name}/collision']=info['events']['collision'][ids][mask].float().clone()
    for stage in (0,1,2):
        mask=env.episode_stage[ids]==stage
        if mask.any():metrics[f'TrainStage/{stage}/success']=info['success'][ids][mask].float().clone()
    zero=ids[speed==0]
    if len(zero):
        for name,source in [('final_drift_m','drift'),('max_drift_m','max_drift'),('final_speed_mps','final_speed')]:
            metrics[f'Stop/{name}']=info[source][zero].clone()
    return metrics


def write_evaluation(writer,report,step):
    """Never mix train, validation, final test, zero-speed or backend series."""
    split='Validation' if 2_000_000<=report['seed_base']<3_000_000 else 'Test' if report['formal_protocol'] else 'Diagnostic'
    if report.get('stress_test'):split='Stress'
    prefix=f'{split}/{report["backend"]}/{"zero" if report["zero_command_test"] else "navigation"}'
    values={'success_rate':report['success_rate'],'episodes':report['episodes'],
            'wall_seconds':report['wall_seconds']}
    for reason,count in report['reason_counts'].items():values[f'reason/{reason}']=count/report['episodes']
    for speed,group in report['by_speed'].items():
        values[f'speed_{speed}/success_rate']=group['success_rate']
        values[f'speed_{speed}/episodes']=group['episodes']
    for key in ('progress_m','min_clearance_m','mean_heading_error_rad','mean_speed_error_mps',
                'final_drift_m','max_drift_m','final_speed_mps','time_s'):
        values[key]=float(np.mean([r[key] for r in report['records']]))
    for key,value in values.items():writer.add_scalar(f'{prefix}/{key}',value,step)
    writer.flush()
