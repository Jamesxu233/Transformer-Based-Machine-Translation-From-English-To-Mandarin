# Transformer-Based-Machine-Translation-From-English-To-Mandarin

It is the final project for MSML 641 Natural Language Processing.

Team Member: 
	Jiayuan Shen
	Ke Xu 
	Qiao Qin

Checklist:
	three code files: datasetPreprocess.ipynb, encoder_decoder.ipynb, train.ipynb
	one presentation file: FinalProject_MSML641.pptx
	one writeup file: writeup.pdf
	one pretrained weights: model_state_dict.pth

dataPreprocess.ipynb is used to preprocess the dataset.
It will do the following:
	1. download the dataset from hugging face;
	2. download the tokenizers from hugging face;
	3. preprocess the dataset, including tokenize both English and Chinese sentences, add special tokens, convert tokens into integers;
	4. save the dataset into a json file.
Note: when saving the json file, make sure to change the path in this code:
	data.to_json('/content/drive/MyDrive/2024Spring/641NaturalLanguageProcessing/NewFolder/dataset.json')

encoder_decode.ipynb is used to define the transformer architecture. 
You do not have to run this file, it will be runned in the train.ipynb file.

train.ipynb file is used to train and evaluate the model.
It will do the following:
	1. load the dataset from the json file saved before;
	2. define the dataset and the collate function used for dataLoader
	3. load the metrics, which are BLEU score and BERT score, from hugging face
	4. initialize the model, optimizer, learning rate scheduler
	5. train the model
	6. evaluate the model before and after training
	7. define the generate function, use the trained model to generate output based on arbitrary input
Note: 
1. change the data_path variable to the path where you saved the json file:
	data_path = r'/content/drive/MyDrive/2024Spring/641NaturalLanguageProcessing/NewFolder/dataset.json'
2. change the model_path variable to the path where you want to save the model weights:
	model_path = r'/content/drive/MyDrive/2024Spring/641NaturalLanguageProcessing/NewFolder/model_state_dict.pth'
3. change the following path to where you saved the encoder_decoder.ipynb file.
	%run "/content/drive/MyDrive/2024Spring/641NaturalLanguageProcessing/NewFolder/code/encoder_decoder.ipynb"
4. the "generating" part of the code is at the end of this file, change the input_sentence to any sentence you want:
	input_sentence = 'can you tell me the truth?'
   then run this cell to generate the output.
5. if you want to load the model we submit, make sure to change the model_path variable to the path where the model weights are saved,
    then in the training chunk (right after the first evaluation), change the load_model variable into True:
	load_model = True
