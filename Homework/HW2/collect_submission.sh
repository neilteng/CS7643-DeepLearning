rm -f cs7643_hw2.zip
zip -r cs7643_hw2.zip . -x "*.ipynb_checkpoints*" "*README.md" "*collect_submission.sh" ".env/*" "*.pyc" "__pycache__/*"
