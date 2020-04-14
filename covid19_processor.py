import glob
import json
import logging
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader, SequentialSampler

logger = logging.getLogger(__name__)
logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  level=logging.INFO)


class PaperAbstractDataset(Dataset):
  """
  returns the paper_id np array, input_ids tensor and attention_mask tensor
  """

  def __init__(self, dataset):
    self.dataset = dataset

  def __getitem__(self, index):
    paper_id = self.dataset[index]["paper_id"]
    input_id = self.dataset[index]["input_id"]
    attention_mask = self.dataset[index]["attention_mask"]
    return paper_id, input_id, attention_mask

  def __len__(self):
    return self.dataset.shape[0]


class Covid19Processor(object):
  def __init__(self):
    pass


  def get_vaccine_and_therapeutic_paper_ids(self, only_vaccine_file, only_therapeutic_file):
    """
    only_vaccine_file: a file containing a list of paper ids that only contain vaccine keywords
    only_therapeutic_file: a file containing a list of paper ids that only contain therapeutic keywords
    return set of vaccine only paper_ids and therapeutic only paper_ids
    """
    only_vaccine_file_names = set()
    only_therapeutic_file_names = set()

    with open(only_vaccine_file, "r") as f:
      only_vaccine_file_names.update(set([paper_id for paper_id in f.readlines()]))
    with open(only_therapeutic_file_names, "r") as f:
      only_therapeutic_file_names.update(set([paper_id for paper_id in f.readlines()]))
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
          filter_set.update(set([paper_id + ".json" for paper_id in f.readlines()]))
      json_file_names = list(filter(lambda x: os.path.split(x)[1] in filter_set, json_file_names))

    logger.info(" sample json file name: %s", str(json_file_names[0]))
    logger.info(" total json files available in dataset: %d", len(json_file_names))
    return json_file_names


  def preprocess_data_to_df(self, json_files, only_vaccine_file_name, only_therapeutic_file_name):
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
      labels.append(0 if paper_id in only_vaccine_file_name else 1)

    logger.info(" total valid abstracts: %s", valid_abstracts)
    logger.info(" empty_abstract_file_names: %d", len(empty_abstract_file_names))
    logger.info(" empty_body_file_names %d", len(empty_body_file_names))
    data = pd.DataFrame(data, columns=["paper_id", "abstract", "body_text"])
    logger.info(" shape of data: %s", str(data.shape))
    return train_test_split(data, labels, test_size=0.33)



  def get_dev_examples(self, data_dir, filename):
    """
    Returns the evaluation example from the data directory.
    Args:
    data_dir: Directory containing the data files used for training and evaluating.
    filename: the evaluation dataset filename.
    """
    if data_dir is None:
      data_dir = ""

    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
      input_data = json.load(reader)["data"]
    return self.create_examples(input_data, "dev")


  def get_train_examples(self, data_dir, filename):
    """
    Returns the training examples from the data directory.
    Args:
    data_dir: Directory containing the data files used for training and evaluating.
    filename: the train dataset filename.
    """
    if data_dir is None:
      data_dir = ""

    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
      input_data = json.load(reader)["data"]
    return self.create_examples(input_data, "train")


  def create_examples(self, input_data, set_type):
    is_training = set_type == "train"
    examples = []
    for entry in tqdm(input_data):
      title = entry["title"]
      for paragraph in entry["paragraphs"]:
        context_text = paragraph["context"]
        for qa in paragraph["qas"]:
          qas_id = qa["id"]
          question_text = qa["question"]
          start_position_character = None
          answer_text = None
          answers = []

          if "is_impossible" in qa:
            is_impossible = qa["is_impossible"]
          else:
            is_impossible = False

          if not is_impossible:
            if is_training:
              answer = qa["answers"][0]
              answer_text = answer["text"]
              start_position_character = answer["answer_start"]
            else:
              answers = qa["answers"]

          example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            context_text=context_text,
            answer_text=answer_text,
            start_position_character=start_position_character,
            title=title,
            is_impossible=is_impossible,
            answers=answers,
          )

          examples.append(example)
    return examples

  def convert_examples_to_features(self):