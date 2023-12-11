# NT3AWNOU _[nɛtɔːnou]_
An open-source platform to centralize rescue data during Morocco's 2023 earthquake. 
Find our webapp hosted in [HuggingFace](https://huggingface.co/spaces/nt3awnou/Nt3awnou-rescue-map) 
and a poster in [NeurIPS poster](https://sites.google.com/view/northafricansinml/accepted-posters?authuser=0#h.tm1b3h823fta).

This repository contains some functions used in the processing of arabic and french user inputs, their analysis and visualization utilities.
For example:
- classifying user needs and NGOs supplies to predetermined categories using [multilingual-e5 model](https://huggingface.co/intfloat/multilingual-e5-large)
- mapping multilingual town names from raw users inputs to a standardized reference list using phonetical representation and text similarity methods
- an attempt to resolve the problem above using the LLM [Mistral 7B](https://huggingface.co/docs/transformers/main/model_doc/mistral)