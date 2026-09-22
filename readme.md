in order to run the code;
1. Download the data folder from google drive link - using https://drive.google.com/drive/folders/1iyCotUND2MA19tbTXP0XfzZ9scphUKkc?usp=drive_link
2. copy this downloaded data folder into root dir.

commands to run the code :-
1. pip install -r requirements.txt
2. python prepare_splits.py
3. python audit
4. python train.py --model cnn 
5. python train.py --model vit 
6. python evaluate.py --model cnn 
7. python evaluate.py --model vit 
8. python explain.py --model cnn 
9. python explain.py --model vit
