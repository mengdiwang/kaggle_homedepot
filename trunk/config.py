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


## parse color and bullet ###
df_all_text_color_bullet = "df_all_text_parsed_bullet_color.p"
df_all_text_bullet = "df_all_text_parsed_bullet.p"

df_materials_path = "processing_text/df_material_processed.csv"
df_attr_bullet_path = "processing_text/df_attribute_bullets_processed.csv"
df_attr_color_path = "processing_text/df_color_processed.csv"

droplist_for_split = ['search_term', 'product_title', 'product_description', 'product_info', 'attr', 'brand',
                      'search_term_parsed_woBrand','brands_in_search_term','search_term_parsed_woBM',
                      'materials_in_search_term','product_title_parsed_woBrand','brands_in_product_title',
                      'product_title_parsed_woBM','materials_in_product_title','search_term_for',
                      'search_term_for_stemmed','search_term_with','search_term_with_stemmed',
                      'product_title_parsed_without','product_title_without_stemmed',
                      'search_term_unstemmed','product_title_unstemmed','product_description_unstemmed',
                      'attribute_bullets','attribute_bullets_parsed','attribute_bullets_parsed_woBrand',
                      'brands_in_attribute_bullets','attribute_bullets_parsed_woBM','materials_in_attribute_bullets',
                      'attribute_bullets_stemmed','attribute_bullets_stemmed_woBM','attribute_bullets_stemmed_woBrand',
                      'word_in_bullets_string','word_in_bullets_string_only_string', 'product_color', 'brand_parsed',
                      'attribute_stemmed', 'value']

droplist_for_cust = ['id','relevance','search_term','product_title','product_description','product_info','attr','brand',
                     'tf-idf_term_title','tf-idf_term_desc','tf-idf_term_brand','search_term_parsed_woBrand',
                     'brands_in_search_term','search_term_parsed_woBM','materials_in_search_term',
                     'product_title_parsed_woBrand','brands_in_product_title','product_title_parsed_woBM',
                     'materials_in_product_title','search_term_for','search_term_for_stemmed','search_term_with',
                     'search_term_with_stemmed','product_title_parsed_without','product_title_without_stemmed',
                     'search_term_unstemmed','product_title_unstemmed','product_description_unstemmed',
                     'attribute_bullets','attribute_bullets_parsed','attribute_bullets_parsed_woBrand',
                     'brands_in_attribute_bullets','attribute_bullets_parsed_woBM','materials_in_attribute_bullets',
                     'attribute_bullets_stemmed','attribute_bullets_stemmed_woBM','attribute_bullets_stemmed_woBrand',
                     'word_in_bullets_string','word_in_bullets_string_only_string', 'product_color', 'brand_parsed',
                     'attribute_stemmed', 'value']
