https://github.com/datalab-to/surya
Installation
You'll need python 3.10+ and PyTorch. You may need to install the CPU version of torch first if you're not using a Mac or a GPU machine. See here for more details.

Install with:

pip install surya-ocr
Model weights will automatically download the first time you run surya.

- The goal is to achieve <2% CER against the test files in test_bills which has a mix of pdf, png files. Maybe finetune the surya ocr model on samples we have and then test again on the /synthetic_test_bills under /test_bills
- We need evals and benchmarks setup against the ground truth values of the files. Post that, we need to generate hypotheses, test them if they actually improve accuracy, confidence etc for the engine. 
- Keep the winning hypotheses and discard the rest, delete the supporting files generated in testing hypotheses pls. 
- **Do not create multiple junk files** - work with existing files and clean up any temporary files after use
- We need <2% Character error rate in the entire pipeline for the raw text dump extraction without post processing. Then lets focus on Word error rate and skip the field accuracy for now.
- Dont create a new testing file everytime keep one test file covering all our needs in /test subdirectory
- Make sure we carry out deployment testing for the fastAPI and make sure port 8080 is configured
