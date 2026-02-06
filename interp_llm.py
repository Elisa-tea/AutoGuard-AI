import os
from mistralai import Mistral
from openai import OpenAI
from PIL import Image
import numpy as np
import os

# Paths relative to project root
embeddings = np.load("outputs/embeddings/embs.npy")
true_labels = np.load("outputs/embeddings/true_labels.npy")
image_hashes = np.load("outputs/hashes/image_hashes.npy")
image_filenames = np.load("outputs/hashes/image_filenames.npy")



sample_idx = 0
# Path relative to project root
data_root = "outputs/imagesGIDS/shell_d/DoS"

normal_emb_mean_norm = 2145.6382
normal_emb_std_norm = 36.8221
normal_pix_mean = 16.6465
normal_pix_std = 62.9636

# OpenAI API
openai_api_key = ""  # your key
client = OpenAI(api_key=openai_api_key)

# get sample
sample_embedding = embeddings[sample_idx]
sample_label = true_labels[sample_idx]
sample_filename = image_filenames[sample_idx]
sample_image_path = os.path.join(data_root, sample_filename)
sample_hash = image_hashes[sample_idx]

emb_norm = np.linalg.norm(sample_embedding)
emb_std = np.std(sample_embedding)

img = Image.open(sample_image_path).convert("L")
img_array = np.array(img).astype(np.float32).flatten()
img_mean = np.mean(img_array)
img_std = np.std(img_array)


llm_prompt = f"""
<PERSONA>
You are a cybersecurity expert specializing in CAN bus intrusion detection. 
</PERSONA>

<TASKS>
1. Explain in natural language why this sample may be anomalous or normal. 
2. Suggest next steps for an automotive cybersecurity operator. 
</TASKS>

<CONSTRAINTS>
1. Write 3 bullet point for task 1 and 2 bullet points for task 2.
2. Keep bullet points compact.
</CONSTRAINTS>

<CONTEXT>
A GAN was trained on normal CAN bus traffic, converted to one-hot encoded grayscale images.
At inference, the GAN's discriminator outputs embeddings.

Result is as follows: 

**Sample**
- Image Hash: {sample_hash}
- File: {os.path.basename(sample_filename)}
- Ground Truth: {"Normal" if sample_label == 0 else "Anomalous"}

**Embedding**
- L2 Norm: {emb_norm:.2f} (normal mean: {normal_emb_mean_norm:.2f})
- Std Dev: {emb_std:.2f} (normal std: {normal_emb_std_norm:.2f})

**Image Pixels**
- Mean: {img_mean:.2f} (normal mean: {normal_pix_mean:.2f})
- Std Dev: {img_std:.2f} (normal std: {normal_pix_std:.2f})
</CONTEXT>

<OUTPUT_FORMAT>
The output format must be
1. Sample is likely to be malicious[benign]. Potential reasons for that are:
    - 
    - 
    - 
2. Future actions:
    - 
    - 
</OUTPUT_FORMAT>

<RECAP>
Use cybersecurity knowledge and terminology where needed.
</RECAP>
"""


api_key = "" # API key
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": llm_prompt,
            },
        ],
    )
print(chat_response.choices[0].message.content)


