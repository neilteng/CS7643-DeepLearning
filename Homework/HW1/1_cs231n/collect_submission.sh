rm -f 1_cs231n.zip
zip -r 1_cs231n.zip . -x "*.git*" "*cs231n/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collect_submission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs231n/build/*" "__pycache__/*"
