Command to run the code :

python main.py  --train_path=<path to train file> \
                --val_path=<path to validation file> \
                --test_path=<path to test file> \
                --out_path=<path to generated output scores> \
                --section=<1 or 2 or 5>

This should work.


Example Command used on local machine (M1 Mac) :

/opt/homebrew/bin/python3.9 "/Users/eshan/Main/OneDrive - IIT Delhi 2/Eshan/IITD/Sem-6/COL341/A1/Main/main.py" --section=2 --train_path="train.csv" --test_path="test.csv" --val_path="validation.csv" --test_path="test.csv" --out_path="out.csv"


