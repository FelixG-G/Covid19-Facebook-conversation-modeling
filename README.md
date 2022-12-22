# Covid19-Facebook-conversation-modeling
Code and datasets for the "Modeling and Moderation of COVID-19 Social Network Chat" paper.

The code is in the 4 .py files.

The list of raw unaltered comments is in the "Data" folder, in the "comment_list.json" file.
The dictionary of the preprocessed version of the comments (that were used by the neural networks) used in the paper are in the "dict_preprocessed_comments.json" file, in the "Data" folder.

The dictionaries containing the values for the "Anger score", "Fear score", "Joy score", "Love score", "Sadness score", "Surprise score", "Sarcasm score", "Sentiment score" and "Toxicity score" can be found in the "Data\Features" folder, in the files with the name containing the feature name (ex.: the "Toxicity score" results are in the "dict_score_toxicity.json" file).
The final full dictionary containing every feature for every comment is in the "Data\Features" folder, in the "final_feature_dict.json" file.

The 12 trained HMMs that are presented in the paper can be found in the "Data\Trained models" folder, in their corresponding JSON file.

The emission probabilities of the 1st HMM (used to generate the heatmap presented in the paper) is found in the "Data" folder, in the "emission_probabilities_model1" Excel file.
