# TheBloke/Llama-2-7b-Chat-GPTQ Cog model

This is an implementation of the [TheBloke/Llama-2-7b-Chat-GPTQ](https://huggingface.co/SG161222/Realistic_Vision_V3.0_VAE) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="Tell me about AI"
