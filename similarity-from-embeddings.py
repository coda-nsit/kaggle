"""
python3 similarity-from-embeddings.py --data_dir word_embeddings/ --distance_metric cosine --k_similar 10 --output_dir similar_words

no need to run this file as a sbatch command. It doesn't make use of GPU
"""

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


# returns the distance of <token> from all the seed words
def distance_from_seeds(distance_metric, token, seed_to_embedding, token_to_embedding):
  dist = 0
  for seed, seed_embedding in seed_to_embedding.items():
    dist += distance_metric(seed_embedding.reshape(1, -1), token_to_embedding[token].reshape(1, -1))
  return dist


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--data_dir",
    default="word_embeddings/",
    type=str,
    required=True,
    help="The directory storing the word embeddings of the tokens (as python dictionary {token : embedding}) in pickle format")

  parser.add_argument(
    "--output_dir",
    default="similar_words/",
    type=str,
    required=True,
    help="The directory storing the word embeddings of the tokens which are most similar (as python dictionary {token : embedding}) in pickle format")

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
  seed_to_embedding = get_embeddings_for_seed_words(os.path.join(args.data_dir, "seed_embeddings_.pickle"))

  pq = PriorityQueue(k)

  logger.info("seed tokens are %s", str(list(seed_to_embedding.keys())))

  '''
  for every token:
    find cosine similarity with the seed tokens
    keep the k most similar words
  '''
  for token_to_embeddings_file in token_to_embeddings_files:
    with open(token_to_embeddings_file, "rb") as f:
      token_to_embedding = pickle.load(f)

    for token, embedding in list(token_to_embedding.items()):
      # don't consider seed words
      if token in seed_to_embedding:
        continue

      if pq.full() is True:
        most_far_word_distance, most_far_word = pq.get()
        token_to_embedding_pq_words.pop(most_far_word)

        dist_current_token = distance_from_seeds(distance_metric, token, seed_to_embedding, token_to_embedding)
        if dist_current_token > most_far_word_distance:
          pq.put((most_far_word_distance, most_far_word))
          token_to_embedding_pq_words[most_far_word] = token_to_embedding[most_far_word]
        else:
          pq.put((dist_current_token, token))
          token_to_embedding_pq_words[token] = embedding

      else:
        dist = distance_from_seeds(distance_metric, token, seed_to_embedding, token_to_embedding)
        pq.put((dist, token))
        token_to_embedding_pq_words[token] = embedding

  with open(f'{args.output_dir}/closest_word_to_embeddings.pickle', 'wb') as handle:
    pickle.dump(token_to_embedding_pq_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

  while pq.empty() is False:
    dist, word = pq.get()
    logger.info("%s %f", word, dist)

if __name__ == "__main__":
  main()
