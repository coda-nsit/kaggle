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
    "--out_dir",
    default="whole_dataset_biobert",
    type=str,
    required=True,
    help="The directory storing the word embeddings of the tokens (as python dictionary {token : embedding}) in pickle format")

  parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="The batch size to feed the model")

  args = parser.parse_args()

  with open(f"similar_words/{args.similar_tokens_to_embeddings}.pickle", "rb") as f:
    similar_token_to_embedding = np.stack(list(pickle.load(f).values())[:-1])

  input_ids = torch.load(f"inputs/{args.data_dir}/input_ids.pt")
  attention_masks = torch.load(f"inputs/{args.data_dir}/attention_masks.pt")
  with open(f"inputs/{args.data_dir}/paper_ids.pickle", "rb") as f:
    paper_ids = pickle.load(f)

  logger.info("%s", str(input_ids.shape))
  logger.info('Token IDs: %s', str(input_ids[0]))

  if args.model_path is None:
    model = BertModel.from_pretrained("bert-base-cased")
  else:
    configuration = BertConfig.from_json_file(f"{args.model_path}/bert_config.json")
    model = BertModel.from_pretrained(f"{args.model_path}/pytorch_model.bin", config=configuration)
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

  for step, batch in enumerate(dataloader):
    if step % 100 == 0:
      logger.info('======== Batch {:} / {:} ========'.format(step, len(dataloader)))
      logger.info("Time to find embeddings for batches {} to {}: {:} (h:mm:ss)".format(max(0, step - 100), step, format_time(time.time() - t0)))
      t0 = time.time()
    # `batch` contains two pytorch tensors and 1 numpy array:
    #   [0]: input ids
    #   [1]: attention masks
    #   [2]: paper_ids
    paper_ids_np = np.array(batch[0], dtype=str)
    b_input_ids = batch[1].to(device)
    b_input_mask = batch[2].to(device)

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

    paper_ids_to_cosine_score = {}
    for batch_number in range(len(embeddings_np)):
      abstract_cosine_score = np.average(cosine_similarity(embeddings_np[batch_number], similar_token_to_embedding))
      paper_id = paper_ids_np[batch_number]
      paper_ids_to_cosine_score[paper_id] = abstract_cosine_score

    del b_input_ids_np
    del b_input_mask_np
    del embeddings_np
    del cls_np

  with open(f"document_scores/{args.out_dir}.pickle", "wb") as f:
    pickle.dump(paper_ids_to_cosine_score, f, pickle.HIGHEST_PROTOCOL)

  logger.info("Total time to complete the entire process: {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

  logger.info("\n")
  logger.info("Document similarity found!")
if __name__ == '__main__':
  main()



