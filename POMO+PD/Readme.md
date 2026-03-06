#Work flow:

generate datasets for training and validation -> train for each hardness -> test for each hardness.
To replicate the results, you can run only test, because the datasets and trained models are saved.


#Train:

python train.py \
  --problem TSPTW \
  --problem_size 10 \
  --train_episodes 1000 \
  --val_episodes 100 \
  --train_batch_size 64 \
  --validation_batch_size 10 \
  --epochs 1000 \
  --no_cuda \
  --hardness {easy, medium, or hard} \
  --penalty_mode primal_dual\


#Generate:

Use this for both validation and test datasets.
generate_data.py --problem TSPTW \   
  --problem_size 10 \
  --pomo_size 1 \
  --hardness {hard, medium, or easy}\
  --num_samples 10000


#Test:

HARD:
python test.py \
  --problem TSPTW \
  --hardness hard \
  --problem_size 10 \
  --checkpoint "/Users/hamidnakhaei/Documents/Education/PhD/Projects/TSP/PIP-constraint/POMO+PIP/results/20260306_120926_TSPTW10_hard_LM/epoch-1000.pt" \
  --test_set_path "/Users/hamidnakhaei/Documents/Education/PhD/Projects/TSP/PIP-constraint/data/TSPTW/test_tsptw10_hard.pkl" \
  --test_episodes 10000 \
  --test_batch_size 64 \
  --no_cuda

MEDIUM:
python test.py \
  --problem TSPTW \
  --hardness hard \
  --problem_size 10 \
  --checkpoint "/Users/hamidnakhaei/Documents/Education/PhD/Projects/TSP/PIP-constraint/POMO+PIP/results/20260306_113435_TSPTW10_medium_LM/epoch-1000.pt" \
  --test_set_path "/Users/hamidnakhaei/Documents/Education/PhD/Projects/TSP/PIP-constraint/data/TSPTW/test_tsptw10_medium.pkl" \
  --test_episodes 10000 \
  --test_batch_size 64 \
  --no_cuda

EASY:
python test.py \
  --problem TSPTW \
  --hardness hard \
  --problem_size 10 \
  --checkpoint "/Users/hamidnakhaei/Documents/Education/PhD/Projects/TSP/PIP-constraint/POMO+PIP/results/20260306_150017_TSPTW10_easy_LM/epoch-1000.pt" \
  --test_set_path "/Users/hamidnakhaei/Documents/Education/PhD/Projects/TSP/PIP-constraint/data/TSPTW/test_tsptw10_easy.pkl" \
  --test_episodes 10000 \
  --test_batch_size 64 \
  --no_cuda
