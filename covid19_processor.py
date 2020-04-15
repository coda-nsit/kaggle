import glob
import json
import logging
import os
from collections import defaultdict
import pickle


from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  level=logging.INFO)


class PaperAbstractDataset(Dataset):
  """
  returns the paper_id np array, input_ids tensor and attention_mask tensor
  """
  def __init__(self, paper_ids, input_ids, attention_masks, labels):
    self.input_ids = input_ids
    self.attention_masks = attention_masks
    self.labels = labels
    self.paper_ids = paper_ids

  def __getitem__(self, index):
    return self.paper_ids[index], self.input_ids[index], self.attention_masks[index], self.labels[index]

  def __len__(self):
    assert len(self.paper_ids) == self.input_ids.shape[0] == self.attention_masks.shape[0] == len(self.labels)
    return len(self.paper_ids)


class Covid19Processor(object):
  def __init__(self):
    pass

  def get_vaccine_and_therapeutic_paper_ids(self, filter_files_dir, only_vaccine_file, only_therapeutic_file):
    """
    only_vaccine_file: a file containing a list of paper ids that only contain vaccine keywords
    only_therapeutic_file: a file containing a list of paper ids that only contain therapeutic keywords
    return set of vaccine only paper_ids and therapeutic only paper_ids
    """
    only_vaccine_file_names = set()
    only_therapeutic_file_names = set()

    with open(os.path.join(filter_files_dir, only_vaccine_file), "r") as f:
      only_vaccine_file_names.update(set([paper_id.rstrip("\n") for paper_id in f.readlines()]))
    with open(os.path.join(filter_files_dir, only_therapeutic_file), "r") as f:
      only_therapeutic_file_names.update(set([paper_id.rstrip("\n") for paper_id in f.readlines()]))
    return only_vaccine_file_names, only_therapeutic_file_names

  def extract_json_file_names(self, data_dir, filter_dir):
    json_file_names = glob.glob(f"{data_dir}/biorxiv_medrxiv/pdf_json/*.json")
    json_file_names.extend(glob.glob(f"{data_dir}/comm_use_subset/pdf_json/*.json"))
    json_file_names.extend(glob.glob(f"{data_dir}/comm_use_subset/pmc_json/*.json"))
    json_file_names.extend(glob.glob(f"{data_dir}/custom_license/pdf_json/*.json"))
    json_file_names.extend(glob.glob(f"{data_dir}/custom_license/pmc_json/*.json"))
    json_file_names.extend(glob.glob(f"{data_dir}/noncomm_use_subset/pdf_json/*.json"))
    json_file_names.extend(glob.glob(f"{data_dir}/noncomm_use_subset/pmc_json/*.json"))

    if filter_dir is not None:
      filter_file_names = glob.glob(f"{filter_dir}/*.out")
      filter_set = set()
      for filter_file_name in filter_file_names:
        with open(filter_file_name, "r") as f:
          filter_set.update(set([paper_id.rstrip("\n") + ".json" for paper_id in f.readlines()]))

      json_file_names = list(filter(lambda x: os.path.split(x)[1] in filter_set, json_file_names))

    logger.info(" sample json file name: %s", str(json_file_names[0]))
    logger.info(" total json files available in dataset: %d", len(json_file_names))
    return json_file_names

  def preprocess_data_to_df(self, json_files, only_vaccine_paper_ids=None, only_therapeutic_paper_ids=None):
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
        title = json_file["metadata"]["title"]
        if "abstract" not in json_file and len(title) < 10:
          empty_abstract_file_names.append(paper_id)
          stale_keys.add(tuple(json_file.keys()))
          continue

        # populate the abstract
        if "abstract" in json_file and json_file["abstract"] == [] and len(title) < 10:
          empty_abstract_file_names.append(paper_id)
        else:
          if len(title) > 10:
            paper_data[paper_id]["abstract"] = [title]
          if "abstract" in json_file:
            for point in json_file["abstract"]:
              paper_data[paper_id]["abstract"].append(point["text"])

    data = []
    labels = []
    valid_abstracts = 0
    print("total papers:", len(paper_data.keys()))
    for idx, paper_id in enumerate(paper_data.keys()):
      paper_data[paper_id]["body_text"] = "".join(paper_data[paper_id]["body_text"])
      if "abstract" in paper_data[paper_id]:
        paper_data[paper_id]["abstract"] = "".join(paper_data[paper_id]["abstract"])
      else:
        paper_data[paper_id]["abstract"] = ""

      # if "abstract" in paper_data[paper_id] and detect(paper_data[paper_id]["abstract"]) is "en":
      if len(paper_data[paper_id]["abstract"]) >= 50 and \
              ("cov" in paper_data[paper_id]["body_text"] or
               "COV" in paper_data[paper_id]["body_text"] or
               "Cov" in paper_data[paper_id]["body_text"] or
               "CoV" in paper_data[paper_id]["body_text"]):
        valid_abstracts += 1
      data.append((paper_id,
                   paper_data[paper_id]["abstract"],
                   paper_data[paper_id]["body_text"]))
      if only_vaccine_paper_ids is not None:
        labels.append(0 if paper_id in only_vaccine_paper_ids else 1)
      else:
        # append a garbage value, cause label is not needed just for consistency
        labels.append(-1)

    logger.info(" total valid abstracts: %s", valid_abstracts)
    logger.info(" empty_abstract_file_names: %d", len(empty_abstract_file_names))
    logger.info(" empty_body_file_names %d", len(empty_body_file_names))
    data = pd.DataFrame(data, columns=["paper_id", "abstract", "body_text"])
    labels = pd.DataFrame(labels, columns=["label"])
    logger.info(" shape of data: %s", str(data.shape))
    return train_test_split(data, labels, test_size=0.33)

  def create_input_ids__attention_masks_tensor(self, args, X_train, y_train, tokenizer, max_seq_length):
    # Tokenize all of the sentences and map the tokens to their word IDs.
    abstracts = X_train["abstract"].values
    paper_ids = X_train["paper_id"].values
    labels = y_train["label"].values

    paper_ids_with_abstracts = []
    labels_with_abstract = []
    input_ids = []
    attention_masks = []

    for idx, (paper_id, point, label) in enumerate(zip(paper_ids, abstracts, labels)):
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

      paper_ids_with_abstracts.append(paper_id)
      labels_with_abstract.append(label)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # save the tensors to avoid re-computation
    torch.save(input_ids, f"inputs/{args.output_dir}/input_ids.pt")
    torch.save(attention_masks, f"inputs/{args.output_dir}/attention_masks.pt")
    with open(f"inputs/{args.output_dir}/paper_ids.pickle", 'wb') as handle:
      pickle.dump(paper_ids_with_abstracts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # overwrite labels only when its not hardcoded in preprocess_data_to_df()
    if all(labels_with_abstract) is not -1:
      with open(f"inputs/{args.output_dir}/labels.pickle", 'wb') as handle:
        pickle.dump(labels_with_abstract, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      with open(f"inputs/{args.output_dir}/dummy_labels.pickle", 'wb') as handle:
        pickle.dump(labels_with_abstract, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return input_ids, attention_masks, paper_ids_with_abstracts, labels_with_abstract
