import glob
import json
import os
from collections import defaultdict
import random
import time
import pickle
import argparse
import logging
import datetime

import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig

import GPUtil as GPU

logger = logging.getLogger(__name__)
logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  level=logging.INFO)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens_with_embeddings = set()
seed_words = set()


def add_seed_word(words):
  seed_words.update(words)


def printm():
  GPUs = GPU.getGPUs()
  gpu = GPUs[0]
  logger.info("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal))


def extract_data(data_dir, filter_file_name):
  json_file_names = glob.glob(data_dir + "biorxiv_medrxiv/pdf_json/*.json")
  json_file_names.extend(glob.glob(data_dir + "comm_use_subset/pdf_json/*.json"))
  json_file_names.extend(glob.glob(data_dir + "comm_use_subset/pmc_json/*.json"))
  json_file_names.extend(glob.glob(data_dir + "custom_license/pdf_json/*.json"))
  json_file_names.extend(glob.glob(data_dir + "custom_license/pmc_json/*.json"))
  json_file_names.extend(glob.glob(data_dir + 'noncomm_use_subset/pdf_json/*.json'))
  json_file_names.extend(glob.glob(data_dir + 'noncomm_use_subset/pmc_json/*.json'))

  if filter_file_name is not None:
    with open(filter_file_name, "r") as f:
      file_names = f.readlines()[0].split(", ")
    filter_set = set([file_name + ".json" for file_name in file_names])
    json_file_names = list(filter(lambda x: os.path.split(x)[1] in filter_set, json_file_names))

  logger.info(" sample json file name: %s", str(json_file_names[0]))
  logger.info(" total json files available in dataset: %d", len(json_file_names))
  return json_file_names


def preprocess_data_to_df(json_files):
  empty_abstract_file_names = []
  empty_body_file_names = []
  paper_data = defaultdict(lambda: defaultdict(list))

  stale_keys = set()

  for json_file_name in json_files:
    with open(json_file_name) as f:
      json_file = json.load(f)

      paper_id = json_file["paper_id"]

      # populate the body_text
      if json_file["body_text"] == []:
        empty_body_file_names.append(paper_id)
      else:
        for point in json_file["body_text"]:
          paper_data[paper_id]["body_text"].append(point["text"])

      # abstract
      if "abstract" not in json_file:
        empty_abstract_file_names.append(paper_id)
        stale_keys.add(tuple(json_file.keys()))
        continue

      # populate the abstract
      if json_file["abstract"] == []:
        empty_abstract_file_names.append(paper_id)
      else:
        for point in json_file["abstract"]:
          paper_data[paper_id]["abstract"].append(point["text"])

  data = []
  for paper_id in paper_data.keys():
    if "abstract" in paper_data[paper_id]:
      paper_data[paper_id]["abstract"] = "".join(paper_data[paper_id]["abstract"])
    else:
      paper_data[paper_id]["abstract"] = ""
    paper_data[paper_id]["body_text"] = "".join(paper_data[paper_id]["body_text"])
    data.append((paper_id, paper_data[paper_id]["abstract"], paper_data[paper_id]["body_text"]))

  logger.info(" empty_abstract_file_names: %d", len(empty_abstract_file_names))
  logger.info(" empty_body_file_names %d", len(empty_body_file_names))
  data = pd.DataFrame(data, columns=["paper_id", "abstract", "body_text"])
  logger.info(" shape of data: %s", str(data.shape))
  return data


def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))
  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))


def create_input_ids__attention_masks_tensor(data, tokenizer, max_seq_length):
  # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []
  attention_masks = []

  for idx, point in enumerate(data):
    if len(point) == 0:
      continue

    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
      point,  # Sentence to encode.
      add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
      max_length=max_seq_length,  # Pad & truncate all sentences.
      pad_to_max_length=True,
      return_attention_mask=True,  # Construct attn. masks.
      return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

  # Convert the lists into tensors.
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  torch.save(input_ids, 'input_ids.pt')
  torch.save(attention_masks, 'attention_masks.pt')
  return input_ids, attention_masks


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--data_dir",
    default="extracted/",
    type=str,
    required=True,
    help="The input data dir. Should contain the .json files (or other data files) for the task.")

  parser.add_argument(
    "--out_dir",
    default="extracted/",
    type=str,
    required=True,
    help="The directory where the output embeddings will be stored as a pickled dictionary")

  parser.add_argument(
    "--filter_file",
    default=None,
    type=str,
    required=False,
    help="The input path to file which contains the names of files which should only be considered out of the entire dataset.")

  parser.add_argument(
    "--model_path",
    default=None,
    type=str,
    required=False,
    help="The path to the .bin transformer model.")

  parser.add_argument("--have_input_data", action="store_true", help="Whether the input data is already stored in the form of Tensors")

  parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
         "than this will be truncated, sequences shorter will be padded.")

  parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="The batch size to feed the model")

  parser.add_argument('--seed_words', nargs='+')

  args = parser.parse_args()

  add_seed_word(args.seed_words)
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  json_files = extract_data(args.data_dir, args.filter_file)
  data = preprocess_data_to_df(json_files)

  abstracts = data["abstract"].to_list()
  del data

  logger.info("total abstracts: %d", len(abstracts))

  if not args.have_input_data:
    input_ids, attention_masks = create_input_ids__attention_masks_tensor(abstracts, tokenizer, args.max_seq_length)
  else:
    input_ids = torch.load("input_ids.pt")
    attention_masks = torch.load("attention_masks.pt")

  logger.info("%s", str(input_ids.shape))
  logger.info('Original: %s', str(abstracts[0]))
  logger.info('Token IDs: %s', str(input_ids[0]))

  if args.model_path is None:
    model = BertModel.from_pretrained("bert-base-cased")
  else:
    configuration = BertConfig.from_json_file(f"{args.model_path}/bert_config.json")
    model = BertModel.from_pretrained(f"{args.model_path}/pytorch_model.bin", config=configuration)
  model.cuda()

  tensor_dataset = TensorDataset(input_ids, attention_masks)

  batch_size = args.batch_size

  dataloader = DataLoader(
    tensor_dataset,
    sampler=SequentialSampler(tensor_dataset),
    batch_size=batch_size)

  device = torch.device("cuda")
  seed_val = 42

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  # Measure the total training time for the whole run.
  total_t0 = time.time()

  logger.info("")
  logger.info('Forward pass...')

  model.eval()

  token_to_embedding_map = defaultdict(list)
  seed_embeddings = defaultdict(list)
  t0 = time.time()

  for step, batch in enumerate(dataloader):


    if step % 100 == 0:
      logger.info('======== Batch {:} / {:} ========'.format(step, len(dataloader)))

    # `batch` contains three pytorch tensors:
    #   [0]: input ids
    #   [1]: attention masks
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)

    embeddings, cls = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

    # move everything to cpu to save GPU space
    b_input_ids_np = b_input_ids.cpu().numpy()
    b_input_mask_np = b_input_mask.cpu().numpy()
    embeddings_np = embeddings.detach().cpu().numpy()
    cls_np = cls.detach().cpu().numpy()

    del b_input_ids
    del b_input_mask
    del embeddings
    del cls
    torch.cuda.empty_cache()

    for batch_number in range(len(b_input_ids_np)):
      tokens = tokenizer.convert_ids_to_tokens(b_input_ids_np[batch_number])
      for token, embedding in zip(tokens, embeddings_np[batch_number]):
        if token in seed_words and token not in seed_embeddings:
          seed_embeddings[token] = embedding
        elif token not in token_to_embedding_map and token not in stop_words and token not in tokens_with_embeddings:
          token_to_embedding_map[token] = embedding
          tokens_with_embeddings.add(token)

    if step % 200 == 0 and step > 0:
      with open(f'word_embeddings/{args.out_dir}/word_embeddings_{step}.pickle', 'wb') as handle:
        pickle.dump(token_to_embedding_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
      del token_to_embedding_map
      token_to_embedding_map = defaultdict(list)
      logger.info("Time to find embeddings for batches {} to {}: {:} (h:mm:ss)".format(max(0, step - 500), step, format_time(time.time() - t0)))
      t0 = time.time()

    del b_input_ids_np
    del b_input_mask_np
    del embeddings_np
    del cls_np

  # save the embeddings of the seed words
  with open(f'word_embeddings/{args.out_dir}/seed_embeddings_.pickle', 'wb') as handle:
    pickle.dump(seed_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
  del seed_embeddings

  logger.info("Total time to complete the entire process: {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

  logger.info("\n")
  logger.info("Embeddings received!")


if __name__ == "__main__":
  main()
