# python3 distill_knowledge.py --device cuda --train_batch_size 128 --test_batch_size 128 --train_num_workers 2 --test_num_workers 2 --epochs 2 --lr 0.001 --loss ce-stl --opt adam --student_mod cnn --teacher_mod cnn --log_wandb 1 --ce_w 0.75 --st_w 0.25 --temp 2

python3 distill_knowledge.py --device cuda --train_batch_size 128 --test_batch_size 128 --train_num_workers 2 --test_num_workers 2 --epochs 1 --lr 0.001 --loss ce-mse --opt sgd --student_mod resnet18_pt --teacher_mod resnet34 --log_wandb 0 --ce_w 0.75 --cs_w 0.25 --profile 1 

#python3 distill_knowledge.py --device cuda --train_batch_size 128 --test_batch_size 128 --train_num_workers 2 --test_num_workers 2 --epochs 2 --lr 0.001 --loss ce-mse --opt adam --student_mod cnn --teacher_mod cnn_pt --teacher_mod_path ./cnn_teacher.pth --log_wandb 1 --ce_w 0.75 --cs_w 0.25 
