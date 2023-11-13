---
layout: post
title: Nano Med GPT Part 1
date: 2023-11-09 15:09:00
description: Part 1 of the Nano Med GPT series
tags: llm gpt medical_notes
featured: true
---
# Nano Med GPT Series
[Nano Med GPT](https://github.com/BenjaminBush/nano_medgpt/tree/main) is a small-scale version of the GPT-2 language model trained on medical note data. Although there are more advanced tools available, such as [DAX Dragon Express](https://www.nuance.com/healthcare/ambient-clinical-intelligence/explore-dax-for-clinicians.html) (which help provide inspiration for this project), this important use case presents a strong opportunity for hands-on learning and development with transformer-based DNN architectures. This blog series will be broken into the following parts:
* Part 1 - Problem Statement, Dataset, and Local Machine Development
* Part 2 - Scaling Up DNN on Azure, Effectiveness

# Problem Statement
Clinicians spend nearly 2 hours per day outside of regular working hours writing notes in a patient's chart. This time consuming task can lead to clinician burnout. Generative AI can assist with documentation by suggesting autocompletion, which may reduce the time required to complete a note and therefore alleviate some burden on the clinician. 

# Dataset
For this project, we make use of the [MIMIC-IV Dataset](https://physionet.org/content/mimiciv/2.2/). MIMIC-IV is the latest version of a database comprising the deidentified health-related data from patients who were admitted to the critical care units of the Beth Israel Deaconess Medical Center between 2008-2019. I am grateful to the producers and owners of the database for providing access. Those that are interested in gaining access to the dataset can follow the instructions detailed [here](https://mimic.mit.edu/docs/gettingstarted/). 

MIMIC-IV provides a wealth of interesting data for myriad research purposes. For this project, we are interested in the ```mimiciv_note``` module, which includes deidentified free text clinical notes for hospital details. We are specifically intersted in the discharge summaries, which can be found in the ```discharge``` table. 
## Exploratory Data Analysis
number of notes
distribution of note length
total raw vocabulary size (words, characters)
what data is de-identified
what is the general format of the note?

# Initial Development

Thanks for joining! Please stay tuned for Part 2 of the blog series where we will cover how to scale our model size and dataset size up using the cloud, as well as evaluate the effectiveness of our model. 