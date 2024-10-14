!pip install bitsandbytes
!pip install accelerate
!pip install transformers
#
import os

if not os.path.exists('/content/gdrive'):
    from google.colab import drive
    drive.mount('/content/gdrive')
#
!pip install git+https://github.com/huggingface/accelerate.git
!pip install git+https://github.com/huggingface/transformers.git
!pip install bitsandbytes