rm -f 2_pytorch.zip
zip -r 2_pytorch.zip . -x "*.git*" "*data/*" "*.pt" "*.ipynb_checkpoints*" "*README.md" "*collect_submission.sh" "*requirements.txt" ".env/*" "*.pyc" "*__pycache__/*"
