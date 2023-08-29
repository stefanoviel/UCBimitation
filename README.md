# UCBimitation

To reproduce the results use the following commands:

"""
python train_expert/infinite_imitation.py --env-name DiscreteGaussianGridworld-v0  --expert-trajs trajs16.pkl --num-threads 1 --max-iter-num 200 --save-model-interval 10 --grid-type 1 --beta 8 --noiseE 0.0
"""

"""
python train_expert/infinite_imitation.py --env-name DiscreteGaussianGridworld-v0  --expert-trajs trajs6.pkl --num-threads 1 --max-iter-num 200 --save-model-interval 10 --grid-type 1 --beta 8 --noiseE 0.05
"""


"""
python train_expert/infinite_imitation.py --env-name DiscreteGaussianGridworld-v0  --expert-trajs trajs30.pkl --num-threads 1 --max-iter-num 200 --save-model-interval 10 --grid-type 1 --beta 8 --noiseE 0.1
"""

To reproduce proximal point use

"""
python train_learner/ppil.py --env-name DiscreteGaussianGridworld-v0  --expert-trajs trajs16.pkl --num-threads 1 --max-iter-num 200 --save-model-interval 10 --grid-type 1 --eta 1 --noiseE 0.0
"""

To use GAIL

"""
python train_learner/gail.py --env-name DiscreteGaussianGridworld-v0  --expert-trajs trajs16.pkl --num-threads 1 --max-iter-num 200 --save-model-interval 10 --grid-type 1 --noiseE 0.0
"""

For AIRL

"""
python train_learner/gail.py --env-name DiscreteGaussianGridworld-v0  --expert-trajs trajs16.pkl --num-threads 1 --max-iter-num 200 --save-model-interval 10 --grid-type 1 --noiseE 0.0 --reward-type airl
"""

