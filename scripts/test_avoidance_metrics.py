"""Reward accounting, episode weighting and actual TensorBoard event regression."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from beni_avoidance.env import AvoidanceEnv
from beni_avoidance.common import Config
from beni_avoidance.metrics import training_metrics,write_evaluation
from beni_avoidance.training import Adapter,Runner,runner_config


class MetricsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_reward_preserved_and_partial_reset(self):
        e=AvoidanceEnv(2,cfg=Config(version=1))
        e.reset(seeds=[1,2],speeds=[.3,0.],count=0,directions=[0.,0.])
        for _ in range(7):
            x=e.state()[0][:,0].clone();before=e.command.clone()
            _,r,_,info=e.step(torch.zeros(2,2))
            pos,angle,v,_,_=e.state()
            old=2*(pos[:,0]-x)/(e.speed*.1+.01)-.05*(v-e.speed).abs()/.5-.03*angle.abs()
            # No active obstacles, hence no proximity penalty.
            old-=.05*(e.command-before).square().sum(-1)
            old=torch.where(e.speed==0,-.1*v.abs()-.2*pos.norm(dim=-1),old)
            for name,weight in [('collision',-30),('fall',-20),('bounds',-10),('stall',-5),('timeout',-5)]:
                old+=weight*info['events'][name]
            old+=10*info['success'];old[e.steps<=0]=0
            torch.testing.assert_close(r,old)
            torch.testing.assert_close(sum(info['reward_terms'].values()),r)
            torch.testing.assert_close(sum(info['reward_sums'].values()),info['episode_return'])
        sums={k:v[1].clone() for k,v in e.reward_sums.items()}
        e.reset([0],seeds=[3],speeds=[.3],count=0,directions=[0.])
        for k,v in sums.items():
            self.assertEqual(float(e.reward_sums[k][0]),0.)
            torch.testing.assert_close(e.reward_sums[k][1],v)

    def test_episode_vectors_and_speed_bins(self):
        e=AvoidanceEnv(3);e.reset(seeds=[1,2,3],speeds=[0.,.1,.5],count=0,directions=[0.,0.,0.])
        _,_,_,info=e.step(torch.zeros(3,2))
        info['success']=torch.tensor([True,False,False])
        first=training_metrics(e,info,torch.tensor([0]))
        second=training_metrics(e,info,torch.tensor([1,2]))
        self.assertAlmostEqual(float(torch.cat([first['Avoidance/success'],second['Avoidance/success']]).mean()),1/3,places=6)
        self.assertIn('Stop/max_drift_m',first)
        self.assertNotIn('Stop/max_drift_m',second)
        self.assertIn('TrainSpeed/low_0.1_0.2/success',second)
        self.assertIn('TrainSpeed/high_0.4_0.5/success',second)

    def test_real_runner_writes_episode_events(self):
        with tempfile.TemporaryDirectory() as directory:
            e=AvoidanceEnv(2);e.reset(seeds=[1,2],speeds=[0.,0.],count=0,directions=[0.,0.])
            adapter=Adapter(e,56,log=Path(directory)/'episodes.jsonl')
            runner=Runner(adapter,runner_config(42,56),directory,'cpu')
            runner.learn(1)
            events=EventAccumulator(directory);events.Reload();tags=events.Tags()['scalars']
            for tag in ['RewardStep/progress','RewardEpisode/stop_drift','Avoidance/success',
                        'Failure/timeout','Stop/final_drift_m','TrainSpeed/zero/success','TrainStage/0/success']:
                self.assertIn(tag,tags)
            records=[json.loads(line) for line in (Path(directory)/'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),2)
            self.assertAlmostEqual(events.Scalars('Stop/final_drift_m')[0].value,
                                   sum(r['drift'] for r in records)/len(records),places=6)

    def test_validation_preserves_training_state_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            e=AvoidanceEnv(1);adapter=Adapter(e,8)
            settings=dict(interval=1,episodes=3,zero_episodes=1,seconds=60)
            runner=Runner(adapter,runner_config(42,8),directory,'cpu',validation=settings)
            runner.save(str(Path(directory)/'model_final.pt'))
            q=e.backend.qpos.clone();h=e.bottom.hidden.clone();frames=e.frames.clone()
            rng=torch.get_rng_state().clone();numpy_rng=copy.deepcopy(e.rng.bit_generator.state)
            actor={k:v.clone() for k,v in runner.alg.get_policy().state_dict().items()}
            runner.validate_checkpoint(str(Path(directory)/'model_final.pt'),1)
            torch.testing.assert_close(e.backend.qpos,q,rtol=0,atol=0)
            torch.testing.assert_close(e.bottom.hidden,h,rtol=0,atol=0)
            torch.testing.assert_close(e.frames,frames,rtol=0,atol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            self.assertEqual(e.rng.bit_generator.state,numpy_rng)
            for k,v in actor.items():torch.testing.assert_close(runner.alg.get_policy().state_dict()[k],v,rtol=0,atol=0)
            self.assertTrue((Path(directory)/'validation/1_zero.json').exists())

if __name__=='__main__':unittest.main()
