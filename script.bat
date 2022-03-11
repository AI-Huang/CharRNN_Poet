python  main.py gen  --model-path='checkpoints/tang_199.pth' --pickle-path='data/processed/tang.npz' --start-words='深度学习'  --prefix-words='江流天地外，山色有无中。' --acrostic=True --nouse-gpu

python  main.py gen  --model-path='checkpoints/tang_199.pth' --pickle-path='data/processed/tang.npz' --start-words='深度学习'  --prefix-words='江流天地外，山色有无中。' --acrostic=False --nouse-gpu
