"""
Gives the document scores based on how much of their abstracts is similar to the words similar to seed words.

1. Read the tokens (that are similar to the seed words) and their embeddings from "similar_words/"
2. Read the input_ids generated from covid-embeddings-simple-code.py from "inputs/" folder
3. Read the attention_masks generated from covid-embeddings-simple-code.py from "inputs/" folder
4. Read the paper_ids of paper with abstracts generated from covid-embeddings-simple-code.py from "inputs/" folder
5. Writes a dictionary of the format {paper_id: score} in "document_scores/" folder

similar_words/, inputs/ =====>  document_scores/
"""
import argparse
import pickle
import random
import time
import logging
import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertConfig


def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))
  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))


class PaperAbstractDataset(Dataset):
  """
  returns the paper_id np array, input_ids tensor and attention_mask tensor
  """

  def __init__(self, paper_ids, input_ids, attention_masks):
    self.paper_ids = paper_ids
    self.input_ids = input_ids
    self.attention_masks = attention_masks

  def __getitem__(self, index):
    paper_id = self.paper_ids[index]
    input_id = self.input_ids[index]
    attention_mask = self.attention_masks[index]

    return paper_id, input_id, attention_mask

  def __len__(self):
    assert len(self.paper_ids) == self.input_ids.shape[0] == self.attention_masks.shape[0]
    return self.input_ids.shape[0]


logger = logging.getLogger(__name__)
logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  level=logging.INFO)


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--similar_tokens_to_embeddings",
    default="closest_word_to_embeddings_whole_dataset_biobert",
    type=str,
    required=False,
    help="The .pickle file which stores the map {token_similar_to_seed: embedding}.")

  parser.add_argument(
    "--data_dir",
    default="whole_dataset_biobert",
    type=str,
    required=False,
    help="The directory storing the input_ids.pt, attention_masks.pt and paper_ids.")

  parser.add_argument(
    "--model_path",
    default=None,
    type=str,
    required=False,
    help="The path to the .bin transformer model.")

  parser.add_argument(
    "--model_name",
    default="BertModel",
    type=str,
    help="The path to the .bin transformer model.")

  parser.add_argument(
    "--output_file",
    default="whole_dataset_biobert",
    type=str,
    required=True,
    help="The directory storing the word embeddings of the tokens (as python dictionary {token : embedding}) in pickle format")

  parser.add_argument(
    "--batch_size",
    default=4,
    type=int,
    help="The batch size to feed the model")

  parser.add_argument(
    "--top_k",
    default=20,
    type=int,
    help="Only the <top_k> tokens in every abstract are used for measuring an abstracts similarity value.")

  args = parser.parse_args()

  with open(f"similar_words/{args.similar_tokens_to_embeddings}.pickle", "rb") as f:
    similar_token_to_embedding = np.stack(list(pickle.load(f).values())[:-1])  # np.stack is needed as the contents of the pickle file is funny, print and see

  input_ids = torch.load(f"inputs/{args.data_dir}/input_ids.pt")
  attention_masks = torch.load(f"inputs/{args.data_dir}/attention_masks.pt")
  with open(f"inputs/{args.data_dir}/paper_ids.pickle", "rb") as f:
    paper_ids = pickle.load(f)

  logger.info("%s", str(input_ids.shape))

  if args.model_name == "BertForSequenceClassification":
    model = BertForSequenceClassification
  else:
    model = BertModel

  if args.model_path is None:
    logger.info("no model_path has been provided so using 'bert-base-cased'")
    model = BertModel.from_pretrained("bert-base-cased")
  else:
    logger.info(f"loading model and config from {args.model_path}")
    configuration = BertConfig.from_json_file(f"{args.model_path}/config.json")
    model = model.from_pretrained(f"{args.model_path}/pytorch_model.bin", config=configuration)
  model.cuda()

  dataset = PaperAbstractDataset(paper_ids, input_ids, attention_masks)

  batch_size = args.batch_size

  dataloader = DataLoader(
    dataset,
    sampler=SequentialSampler(dataset),
    batch_size=batch_size)

  device = torch.device("cuda")
  seed_val = 42

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  # Measure the total training time for the whole run.
  total_t0 = time.time()
  t0 = time.time()

  logger.info("")
  logger.info('Forward pass...')

  model.eval()

  paper_ids_to_cosine_score = {}
  for step, batch in enumerate(dataloader):
    if step % 100 == 0:
      logger.info('======== Batch {:} / {:} ========'.format(step, len(dataloader)))
      logger.info("Time to find embeddings for batches {} to {}: {:} (h:mm:ss)".format(max(0, step - 100), step, format_time(time.time() - t0)))
      t0 = time.time()
    # `batch` contains two pytorch tensors and 1 numpy array:
    #   [0]: paper ids
    #   [1]: input ids
    #   [2]: attention masks
    paper_ids_np = np.array(batch[0], dtype=str)
    b_input_ids = batch[1].to(device)
    b_input_mask = batch[2].to(device)

    # in case there is "label" in the batch
    if len(batch) == 4:
      _ = batch[3].to(device)

    # embeddings, cls
    outputs = model(b_input_ids,
                    attention_mask=b_input_mask)

    # if model is BertForSequenceClassification
    if args.model_name == "BertForSequenceClassification":
      cls, hidden_states = outputs
      embeddings, layers = hidden_states[0].detach().cpu(), hidden_states[1].detach().cpu()
      del layers
    else:
      embeddings, cls = outputs

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

    for batch_number in range(len(embeddings_np)):
      abstract_cosine_score = np.average(np.sort(cosine_similarity(embeddings_np[batch_number], similar_token_to_embedding))[:args.top_k])
      paper_id = paper_ids_np[batch_number]
      paper_ids_to_cosine_score[paper_id] = abstract_cosine_score

    del b_input_ids_np
    del b_input_mask_np
    del embeddings_np
    del cls_np

  with open(f"document_scores/{args.output_file}.pickle", "wb") as f:
    pickle.dump(paper_ids_to_cosine_score, f, pickle.HIGHEST_PROTOCOL)

  logger.info("Total time to complete the entire process: {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

  logger.info("\n")
  logger.info("Document similarity found!")


if __name__ == '__main__':
  main()
