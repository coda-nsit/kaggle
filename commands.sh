##### Notes:
# keep batch_size 8 for max_seq_length of 512
##### input file names
# input_ids_whole_dataset_biobert.pt
# attention_masks_whole_dataset_biobert.pt
# paper_ids_whole_dataset_biobert.pt
# input_ids_whole_dataset_bert.pt
# attention_masks_whole_dataset_bert.pt
# paper_ids_whole_dataset_bert.py
# input_ids_only_vaccines_biobert.pt
# attention_masks_only_vaccines_biobert.pt
# paper_ids_whole_dataset_biobert.py

##### commands
# whole dataset + biobert
python3 covid19-embeddings-simple-code.py \
    --have_input_data \
    --data_dir "whole_dataset_biobert" \
    --max_seq_length 512 \
    --batch_size 8 \
    --out_dir "whole_dataset_biobert" \
    --seed_words "vaccine" "vaccines" "vaccination" "vaccinations" \
    --model_path "/home/rbanerj8/biobert"
python3 similarity-from-embeddings.py \
  --data_dir "whole_dataset_biobert" \
  --distance_metric "cosine" \
  --output_file "closest_word_to_embeddings_whole_dataset_biobert" \
  --k_similar 1000
python3 find_document_similarity.py \
  --similar_tokens_to_embeddings "closest_word_to_embeddings_whole_dataset_biobert" \
  --data_dir "whole_dataset_biobert" \
  --model_path "/home/rbanerj8/biobert" \
  --output_file "whole_dataset_biobert" \
  --top_k 50

# only vaccines + biobert
python3 covid19-embeddings-simple-code.py \
    --have_input_data \
    --data_dir "only_vaccines_biobert" \
    --max_seq_length 512 \
    --batch_size 8 \
    --filter_file "Papers_with_only_vaccines_in_abstract.txt" \
    --out_dir "only_vaccines_biobert" \
    --seed_words "vaccine" "vaccines" "vaccination" "vaccinations" \
    --model_path "/home/rbanerj8/biobert"
python3 similarity-from-embeddings.py \
  --data_dir "only_vaccines_biobert" \
  --distance_metric "cosine" \
  --output_file "closest_word_to_embeddings_only_vaccines_biobert" \
  --k_similar 1000
python3 find_document_similarity.py \
  --similar_tokens_to_embeddings "closest_word_to_embeddings_only_vaccines" \
  --data_dir "only_vaccines_biobert" \
  --model_path "/home/rbanerj8/biobert" \
  --output_file "only_vaccines_biobert" \
  --top_k 50

# whole dataset + bert
python3 covid19-embeddings-simple-code.py \
    --have_input_data \
    --data_dir "whole_dataset" \
    --max_seq_length 512 \
    --batch_size 8 \
    --out_dir "whole_dataset" \
    --seed_words "vaccine" "vaccines" "vaccination" "vaccinations"
python3 similarity-from-embeddings.py \
  --data_dir "fine_tuned_result" \
  --distance_metric "cosine" \
  --output_file "closest_word_to_embeddings_only_covid_fine_tuned_bert" \
  --k_similar 1000
python3 find_document_similarity.py \
  --similar_tokens_to_embeddings "closest_word_to_embeddings_whole_dataset_bert" \
  --data_dir "whole_dataset" \
  --output_file "whole_dataset" \
  --top_k 50


##### finetuning
# fine-tuning the model
python3 vaccone_therapy_classification.py \
    --model_type "bert" \
    --model_name_or_path "bert-base-cased" \
    --filter_dir "filter_files" \
    --output_dir "fine_tuned_result" \
    --data_dir "extracted" \
    --do_train \
    --do_eval \
    --verbose_logging \
    --overwrite_cache \
    --per_gpu_train_batch_size 4 \
    --per_gpu_train_eval_size 4 \
    --save_steps 500 \
# get embeddings from the fine-tuned model
python3 embeddings_from_fine_tuned_model.py \
    --model_and_config_dir "fine_tuned_result" \
    --data_dir "extracted" \
    --output_dir "fine_tuned_result" \
    --seed_words "vaccine" "vaccines" "vaccination" "vaccinations" \
python3 similarity-from-embeddings.py \
  --data_dir "fine_tuned_result" \
  --distance_metric "cosine" \
  --output_file "closest_word_to_embeddings_only_covid_fine_tuned_bert" \
  --k_similar 1000
python3 find_document_similarity.py \
    --model_name "BertForSequenceClassification" \
    --similar_tokens_to_embeddings "closest_word_to_embeddings_only_covid_fine_tuned_bert" \
    --data_dir "fine_tuned_result" \
    --model_path "/home/rbanerj8/covid/fine_tuned_result" \
    --output_file "fine_tuned_result" \
    --top_k 50 \





