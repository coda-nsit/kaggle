import argparse
import glob
import pickle
import os
from queue import PriorityQueue
from sklearn.metrics.pairwise import cosine_similarity

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  level=logging.INFO)


def find_distance(func, list_1, list_2):
  return func(list_1, list_2)


def get_embeddings_for_seed_words(file_name):
  with open(file_name, "rb") as f:
    return pickle.load(f)


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--data_dir",
    default="word_embeddings/",
    type=str,
    required=True,
    help="The directory storing the word embeddings of the tokens (as python dictionary {token : embedding}) in pickle format")

  parser.add_argument(
    "--distance_metric",
    default="cosine",
    type=str,
    required=True,
    help="The distance metric to be used to find the similarity between the embeddings")

  parser.add_argument(
    "--k_similar",
    default=5,
    type=int,
    required=True,
    help="The number of similar words needed.")

  args = parser.parse_args()
  k = args.k_similar

  distance_metric = cosine_similarity
  if args.distance_metric == "cosine":
    distance_metric = cosine_similarity

  # stores only those tokens as the key which are in the priority queue
  token_to_embedding_pq_words = {}

  token_to_embeddings_files = glob.glob(args.data_dir + "word_embeddings_*.pickle")
  seed_to_embeddings = get_embeddings_for_seed_words(os.path.join(args.data_dir, "seed_embeddings_.pickle"))

  pq = PriorityQueue(k)

  '''
  for every token:
    find cosine similarity with the seed tokens
    keep the k most similar words
  '''
  for token_to_embeddings_file in token_to_embeddings_files:
    with open(token_to_embeddings_file, "rb") as f:
      token_to_embedding = pickle.load(f)

    for token, embedding in token_to_embedding.items():
      if pq.full() is True:
        most_far_word = pq.get()
        most_far_word_embedding = token_to_embedding_pq_words[most_far_word]
        token_to_embedding_pq_words.pop(most_far_word)

        cos_1 = cos_2 = 0

        for seed, seed_embedding in seed_to_embeddings:
          cos_1 += find_distance(distance_metric, seed_embedding, most_far_word_embedding)
          cos_2 += find_distance(distance_metric, seed_embedding, embedding)

        if cos_1 > cos_2:
          pq.put(most_far_word)
          token_to_embedding_pq_words[most_far_word] = most_far_word_embedding
        else:
          pq.put(token)
          token_to_embedding_pq_words[token] = embedding

      else:
        pq.put(token)
        token_to_embedding_pq_words[token] = embedding

  logger.info("closest tokens %s", str(token_to_embedding_pq_words.keys()))

if __name__ == "__main__":
  main()
