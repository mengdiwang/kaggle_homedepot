############# init_feature_extraction ##############
path_train = "../input/train.csv"
path_test = "../input/test.csv"
path_attr = "../input/attributes.csv"
path_product = "../input/product_descriptions.csv"
all_data_pickle = "all_data_corrected.p"
saved_features = "tf-idf_features_corrected.p"

################## train ####################
saved_models = "all_data_text_parsed.p"
train_tfidf_features = "tf-idf_features_corrected.p"
bullet_features = 'processing_text/df_attribute_bullets_processed.csv'

############# brand ###########
saved_models_csv = "all_data_corrected.csv"
material_df_csv = "processing_text/material_statistics.csv"
brand_df_csv = "processing_text/brand_statistics.csv"

############## parse text ##########
prased_features = "all_data_text_parsed.p"
