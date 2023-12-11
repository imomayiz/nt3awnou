# NT3AWNOU _[nɛtɔːnou]_: A data-driven platform for humanitarian aid
An open-source platform to collect, process and visualise rescue data during Morocco's 2023 earthquake. 
Find our webapp hosted in [HuggingFace](https://huggingface.co/spaces/nt3awnou/Nt3awnou-rescue-map) 
and our poster in [NeurIPS](https://sites.google.com/view/northafricansinml/accepted-posters?authuser=0#h.tm1b3h823fta).

This repository contains some functions used in the processing of arabic and french user inputs, their analysis and visualization.

Examples of tasks:
- classifying user needs and NGOs supplies to predetermined categories using [multilingual-e5 model](https://huggingface.co/intfloat/multilingual-e5-large)
- mapping multilingual town names from raw users inputs to a standardized reference list using phonetical representation and text similarity methods
- an attempt to resolve the problem above using the LLM [Mistral 7B](https://huggingface.co/docs/transformers/main/model_doc/mistral)


## In a nutshell
On September 8, 2023, a devastating magnitude 6.8 earthquake hit Morocco’s High Atlas Mountains,
uniting Morocco’s government, Non-Governmental Organizations (NGOs), and citizens in an inspir-
ing display of solidarity. Recognizing the need for improved relief coordination, our data-driven
platform was created as a centralized hub to consolidate vital earthquake data and relief efforts.

## Challenges
Consolidating diverse, heterogeneous data sources is complex, particularly in low-resource
languages like the Moroccan dialect. We employed Natural Language Processing (NLP)
techniques to convert collected data into a refined and usable dataset. Key challenges included
authentic data collection in crises, accurately identifying similar-named rural villages (‘douars’),
and obtaining precise geolocation despite inaccuracies in mapping APIs (e.g. Google Maps and
OpenStreetMap). Overcoming these obstacles was essential for effective humanitarian support.

## Credits
More than 50 moroccan researchers, engineers and PhD students contributed to this work. You can find some of them in the contributors list in Nt3awnou's Hugging Face space.
