for seed in 0 1 2 3 4
do
  python train.py --config $1 --seed $seed
done

python Utils/performance_assessment.py --save_dir $1 \
    --run_ids 0 1 2 3 4